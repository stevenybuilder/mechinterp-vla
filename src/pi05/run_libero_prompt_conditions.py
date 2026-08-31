#!/usr/bin/env python
"""Stage 0/1 behavioural runner: pi0.5 (LeRobot) on LIBERO under prompt conditions.

Runs (suite, task_id, init_state_id, condition) episodes and appends one JSON line per episode with:
  success, n_steps, prompt, wrong-task used, per-step eef position, object positions (every `--obj_every` steps),
  a LIBERO-CF-style TOUCH/CONTACT record (first object contacted by any gripper geom, first object grasped by both
  finger pads, full contact timeline), and, once per condition, the exact tokenized prompt.

Conditions:
  correct       the suite's own instruction for the task
  null          empty instruction (state part of the prompt unchanged: "Task: , State: ...")
  wrong_object  another task's instruction from the same suite whose target object is present in the scene
  wrong:<j>     task j's instruction used as the wrong prompt (cross-distractor arbitration grid)
  custom:<name> prompt taken from --custom_prompts JSON {name: prompt}

Determinism: torch RNG is seeded per (task, init) before each episode so flow-matching noise is identical across
conditions; the environment is reset from LIBERO's shipped init-state file. Resume-safe (skips episodes present
in the output JSONL).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite, get_libero_dummy_action
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

GRIPPER_BODY_RE = re.compile(r"gripper|finger|hand|panda_hand", re.I)
NON_OBJECTS = {"floor", "main_table", "table", "world"}

# libero_goal is one shared scene; every target is present. Chosen so the wrong task manipulates a different object
# where possible (bowl tasks are mapped away from bowl tasks).
GOAL_WRONG_MAP = {0: 2, 1: 9, 2: 1, 3: 7, 4: 5, 5: 4, 6: 0, 7: 3, 8: 2, 9: 8}
# Object of interest per libero_goal task (name in obj_body_id). Task 3 has two; the first contacted is the cabinet.
GOAL_TARGET_OBJECT = {
    0: "wooden_cabinet_1", 1: "akita_black_bowl_1", 2: "wine_bottle_1", 3: "wooden_cabinet_1",
    4: "akita_black_bowl_1", 5: "plate_1", 6: "cream_cheese_1", 7: "flat_stove_1", 8: "akita_black_bowl_1",
    9: "wine_bottle_1",
}


def batchify(x):
    if isinstance(x, dict):
        return {k: batchify(v) for k, v in x.items()}
    return np.asarray(x)[None]


def parse_ids(s: str, n: int) -> list[int]:
    if s in ("all", ""):
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


class ContactTracker:
    """Map every MuJoCo geom to the LIBERO object/fixture whose body subtree contains it."""

    def __init__(self, rs_env):
        rs_env = getattr(rs_env, "env", rs_env)  # unwrap OffScreenRenderEnv
        self.env = rs_env
        sim = rs_env.sim
        model = sim.model
        self.obj_body_id = dict(rs_env.obj_body_id)
        body2obj = {}
        for name, bid in self.obj_body_id.items():
            if name in NON_OBJECTS:
                continue
            body2obj[bid] = name
        self.geom2obj = {}
        self.gripper_geoms = set()
        n_geom = model.ngeom
        for g in range(n_geom):
            b = model.geom_bodyid[g]
            bname = model.body_id2name(b) or ""
            gname = model.geom_id2name(g) or ""
            if gname.startswith("gripper0") or GRIPPER_BODY_RE.search(bname) or GRIPPER_BODY_RE.search(gname):
                self.gripper_geoms.add(g)
                continue
            bb = b
            while bb > 0:
                if bb in body2obj:
                    self.geom2obj[g] = body2obj[bb]
                    break
                bb = model.body_parentid[bb]
        pads = rs_env.robots[0].gripper.important_geoms
        def _ids(names):
            out = set()
            for n in names:
                try:
                    out.add(model.geom_name2id(n))
                except Exception:
                    pass
            return out
        self.left_pad = _ids(pads.get("left_fingerpad", []))
        self.right_pad = _ids(pads.get("right_fingerpad", []))
        assert self.left_pad and self.right_pad, f"fingerpad geoms not found: {pads}"
        self.objects = [n for n in self.obj_body_id if n not in NON_OBJECTS]

    def contacts(self):
        """Return (set of objects touched by any gripper geom, set grasped by both pads) at current sim state."""
        sim = self.env.sim
        touched, left, right = set(), set(), set()
        for c in sim.data.contact[: sim.data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            for gg, go in ((g1, g2), (g2, g1)):
                if gg in self.gripper_geoms and go in self.geom2obj:
                    obj = self.geom2obj[go]
                    touched.add(obj)
                    if gg in self.left_pad:
                        left.add(obj)
                    if gg in self.right_pad:
                        right.add(obj)
        return touched, left & right

    def object_positions(self):
        sim = self.env.sim
        return {n: [float(v) for v in sim.data.body_xpos[self.obj_body_id[n]]] for n in self.objects}


def target_object_for_prompt(suite_name: str, suite, task_id: int, scene_objects: list[str]) -> str | None:
    if suite_name == "libero_goal":
        return GOAL_TARGET_OBJECT.get(task_id)
    # libero_object / spatial: the first obj_of_interest is the manipulated object; derive from task name
    name = suite.tasks[task_id].name  # e.g. pick_up_the_alphabet_soup_and_place_it_in_the_basket
    m = re.match(r"pick_up_the_(.+?)_and_place_it_in_the_basket", name)
    if m:
        cand = m.group(1) + "_1"
        return cand if cand in scene_objects else None
    return None


def choose_wrong_task(suite_name, suite, task_id, scene_objects):
    n = len(suite.tasks)
    if suite_name == "libero_goal":
        return GOAL_WRONG_MAP[task_id]
    # object suite: first other task (cyclic order) whose target is present in this scene
    for k in range(1, n):
        j = (task_id + k) % n
        tgt = target_object_for_prompt(suite_name, suite, j, scene_objects)
        if tgt is not None and tgt in scene_objects:
            return j
    return (task_id + 1) % n


def load_policy(policy_path, revision, n_action_steps, device, processor_path=None):
    if revision:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
    pcfg = PreTrainedConfig.from_pretrained(policy_path, revision=revision) if revision else PreTrainedConfig.from_pretrained(policy_path)
    pcfg.pretrained_path = policy_path
    pcfg.n_action_steps = n_action_steps
    pcfg.device = device
    env_cfg = LiberoEnvCfg(task="libero_object")
    policy = make_policy(cfg=pcfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg,
        pretrained_path=processor_path or policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    epre, epost = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
    # sanity: Gemma ties embed_tokens to lm_head; the checkpoint stores only lm_head
    pg = policy.model.paligemma_with_expert.paligemma
    assert torch.equal(pg.model.language_model.embed_tokens.weight, pg.lm_head.weight), "embed/lm_head not tied"
    return policy, pcfg, pre, post, epre, epost


def make_env(suite, suite_name, task_id, bddl_file=None, init_states=None, hw=360):
    cls = LiberoEnv
    if bddl_file:
        from libero.libero.envs import OffScreenRenderEnv

        class _BddlEnv(LiberoEnv):
            def _make_envs_task(self, task_suite, task_id=0):
                task = task_suite.get_task(task_id)
                self.task = task.name
                self.task_description = task.language
                env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=self.observation_height,
                                         camera_widths=self.observation_width)
                env.reset()
                return env

        cls = _BddlEnv
    env = cls(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        observation_height=hw,
        observation_width=hw,
        init_states=True,
        episode_index=0,
        n_envs=1,
    )
    if init_states is not None:
        env._init_states = init_states
    return env


def run_episode(env, policy, pre, post, epre, epost, prompt, init_id, seed, max_steps, tracker, obj_every, record_tokens):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy.reset()
    env.init_state_id = init_id
    obs, _ = env.reset(seed=seed)
    rs = env._env
    eef, objpos, timeline = [], {}, []
    first_touch = None
    first_grasp = None
    touched_any = set()
    grasped_any = set()
    t0 = time.time()
    success = False
    tokens_rec = None
    step = 0
    for step in range(max_steps):
        o = preprocess_observation(batchify(obs))
        o["task"] = [prompt]
        o = epre(o)
        o = pre(o)
        if record_tokens and tokens_rec is None:
            ids = o[OBS_LANGUAGE_TOKENS][0].tolist()
            mask = o[OBS_LANGUAGE_ATTENTION_MASK][0].tolist()
            n = int(sum(mask))
            tokens_rec = {"n_tokens": n, "ids": ids[:n], "full_prompt": o["task"][0] if "task" in o else None}
        with torch.inference_mode():
            a = policy.select_action(o)
        a = post(a)
        a = epost({ACTION: a})[ACTION]
        a = a.cpu().numpy()[0]
        raw_obs, reward, done, info = rs.step(a)
        obs = env._format_raw_obs(raw_obs)
        eef.append([float(v) for v in raw_obs["robot0_eef_pos"]])
        touched, grasped = tracker.contacts()
        if touched:
            timeline.append([step, sorted(touched), sorted(grasped)])
        for ob in touched:
            if first_touch is None:
                first_touch = {"object": ob, "step": step}
            touched_any.add(ob)
        for ob in grasped:
            if first_grasp is None:
                first_grasp = {"object": ob, "step": step}
            grasped_any.add(ob)
        if step % obj_every == 0:
            objpos[str(step)] = tracker.object_positions()
        success = bool(rs.check_success())
        if success or done:
            break
    objpos["final"] = tracker.object_positions()
    return {
        "success": success,
        "n_steps": step + 1,
        "wall_s": round(time.time() - t0, 2),
        "first_touch": first_touch,
        "first_grasp": first_grasp,
        "touched_any": sorted(touched_any),
        "grasped_any": sorted(grasped_any),
        "contact_timeline": timeline,
        "eef_pos": eef,
        "object_pos": objpos,
        "tokens": tokens_rec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task_ids", default="all")
    ap.add_argument("--init_ids", default="0-19")
    ap.add_argument("--conditions", default="correct,null,wrong_object")
    ap.add_argument("--conditions_by_task", default=None,
                    help='JSON file {"<task_id>": ["wrong:2", ...]} overriding --conditions per task')
    ap.add_argument("--custom_prompts", default=None, help="JSON file {name: prompt}; use conditions custom:<name>")
    ap.add_argument("--bddl_file", default=None, help="override BDDL for all task_ids (Stage 1 twins)")
    ap.add_argument("--init_file", default=None, help="override init-state .pt/.init for --bddl_file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--processor_path", default=None, help="load pre/post-processors (normaliser stats) from here")
    ap.add_argument("--n_action_steps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--obj_every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                done_keys.add((r["suite"], r["task_id"], r["init_id"], r["condition"]))
            except Exception:
                pass
    custom = json.load(open(args.custom_prompts)) if args.custom_prompts else {}
    conditions = [c for c in args.conditions.split(",") if c]
    cond_by_task = None
    if args.conditions_by_task:
        cond_by_task = {int(k): list(v) for k, v in json.load(open(args.conditions_by_task)).items()}

    policy, pcfg, pre, post, epre, epost = load_policy(args.policy, args.revision, args.n_action_steps, args.device, args.processor_path)
    suite = _get_suite(args.suite)
    task_ids = parse_ids(args.task_ids, len(suite.tasks))
    init_ids = parse_ids(args.init_ids, 10**6)
    max_steps = args.max_steps or TASK_SUITE_MAX_STEPS.get(args.suite, 500)
    tokens_logged = set()

    print(f"[runner] suite={args.suite} tasks={task_ids} inits={init_ids} "
          f"conds={cond_by_task if cond_by_task is not None else conditions} out={out}", flush=True)
    for task_id in task_ids:
        env = make_env(suite, args.suite, task_id, bddl_file=args.bddl_file)
        if args.init_file:
            env._init_states = torch.load(args.init_file, weights_only=False)
        rs = env._env
        tracker = ContactTracker(rs)
        scene_objects = tracker.objects
        correct_prompt = suite.tasks[task_id].language
        wrong_tid = choose_wrong_task(args.suite, suite, task_id, scene_objects)
        wrong_prompt = suite.tasks[wrong_tid].language
        target_obj = target_object_for_prompt(args.suite, suite, task_id, scene_objects)
        wrong_obj = target_object_for_prompt(args.suite, suite, wrong_tid, scene_objects)
        print(f"[runner] task {task_id} '{correct_prompt}' target={target_obj} wrong_task={wrong_tid} "
              f"'{wrong_prompt}' wrong_obj={wrong_obj} objects={scene_objects}", flush=True)
        task_conditions = cond_by_task.get(task_id, []) if cond_by_task is not None else conditions
        if not task_conditions:
            env.close()
            continue
        for init_id in init_ids:
            for cond in task_conditions:
                key = (args.suite, task_id, init_id, cond)
                if key in done_keys:
                    continue
                if cond == "correct":
                    prompt = correct_prompt
                elif cond == "null":
                    prompt = ""
                elif cond == "wrong_object":
                    prompt = wrong_prompt
                elif cond.startswith("wrong:"):
                    # explicit cross-distractor cell: use task <j>'s instruction as the wrong prompt
                    prompt = suite.tasks[int(cond.split(":", 1)[1])].language
                elif cond.startswith("custom:"):
                    prompt = custom[cond.split(":", 1)[1]]
                else:
                    raise ValueError(cond)
                if cond == "wrong_object":
                    cond_wrong_tid = wrong_tid
                elif cond.startswith("wrong:"):
                    cond_wrong_tid = int(cond.split(":", 1)[1])
                else:
                    cond_wrong_tid = None
                cond_wrong_obj = (
                    target_object_for_prompt(args.suite, suite, cond_wrong_tid, scene_objects)
                    if cond_wrong_tid is not None else None
                )
                seed = args.seed + task_id * 1000 + init_id
                rec = run_episode(env, policy, pre, post, epre, epost, prompt, init_id, seed, max_steps, tracker,
                                  args.obj_every, record_tokens=(cond, task_id) not in tokens_logged)
                tokens_logged.add((cond, task_id))
                rec.update({
                    "suite": args.suite, "task_id": task_id, "task_name": suite.tasks[task_id].name,
                    "init_id": init_id, "condition": cond, "prompt": prompt, "seed": seed,
                    "target_object": target_obj, "wrong_task_id": cond_wrong_tid,
                    "wrong_object": cond_wrong_obj,
                    "scene_objects": scene_objects, "policy": args.policy, "revision": args.revision,
                    "n_action_steps": args.n_action_steps, "max_steps": max_steps, "tag": args.tag,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                ft = rec["first_touch"]["object"] if rec["first_touch"] else None
                print(f"[ep] {args.suite} t{task_id} i{init_id} {cond:<12} success={int(rec['success'])} "
                      f"steps={rec['n_steps']} first_touch={ft} grasp={rec['first_grasp']['object'] if rec['first_grasp'] else None} "
                      f"{rec['wall_s']}s", flush=True)
        env.close()
    print("[runner] DONE", flush=True)


if __name__ == "__main__":
    main()
