#!/usr/bin/env python
"""Stage 0' behavioural runner: OpenVLA-OFT on LIBERO under prompt conditions (mirror of
run_libero_prompt_conditions.py for pi0.5).

Must be run from the openvla-oft repo root (imports experiments.robot.*). One JSON line per episode with the same
field names as the pi0.5 runner (analysis script scripts/vla/analyze_stage0.py is shared):
  success, n_steps, wall_s, first_touch / first_grasp / touched_any / grasped_any / contact_timeline (LIBERO-CF style
  contact check via sim.data.contact and gripper geoms), prompt (the task label) and full_prompt (the exact string fed
  to the processor), wrong task used, per-step eef pos, object positions.

Conditions:
  correct       the suite's own instruction
  null          empty instruction -> OFT prompt template "In: What action should the robot take to ?\nOut:"
  wrong_object  another task's instruction from the same suite whose target object is present in the scene

OFT inference is deterministic (L1 regression head, no sampling); seeds are set anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

sys.path.insert(0, os.getcwd())
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig,
    TASK_MAX_STEPS,
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere  # noqa: E402
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402

GRIPPER_BODY_RE = re.compile(r"gripper|finger|hand|panda_hand", re.I)
NON_OBJECTS = {"floor", "main_table", "table", "world"}
GOAL_WRONG_MAP = {0: 2, 1: 9, 2: 1, 3: 7, 4: 5, 5: 4, 6: 0, 7: 3, 8: 2, 9: 8}
GOAL_TARGET_OBJECT = {
    0: "wooden_cabinet_1", 1: "akita_black_bowl_1", 2: "wine_bottle_1", 3: "wooden_cabinet_1",
    4: "akita_black_bowl_1", 5: "plate_1", 6: "cream_cheese_1", 7: "flat_stove_1", 8: "akita_black_bowl_1",
    9: "wine_bottle_1",
}
PROMPT_TEMPLATE = "In: What action should the robot take to {task_label_lower}?\nOut:"


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
    """Map every MuJoCo geom to the LIBERO object/fixture whose body subtree contains it (same as pi0.5 runner)."""

    def __init__(self, rs_env):
        rs_env = getattr(rs_env, "env", rs_env)  # unwrap OffScreenRenderEnv
        self.env = rs_env
        sim = rs_env.sim
        model = sim.model
        self.obj_body_id = dict(rs_env.obj_body_id)
        body2obj = {bid: name for name, bid in self.obj_body_id.items() if name not in NON_OBJECTS}
        self.geom2obj = {}
        self.gripper_geoms = set()
        for g in range(model.ngeom):
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


def target_object_for_prompt(suite_name, suite, task_id, scene_objects):
    if suite_name == "libero_goal":
        return GOAL_TARGET_OBJECT.get(task_id)
    name = suite.get_task(task_id).name
    m = re.match(r"pick_up_the_(.+?)_and_place_it_in_the_basket", name)
    if m:
        cand = m.group(1) + "_1"
        return cand if cand in scene_objects else None
    return None


def choose_wrong_task(suite_name, suite, task_id, scene_objects):
    n = suite.n_tasks
    if suite_name == "libero_goal":
        return GOAL_WRONG_MAP[task_id]
    for k in range(1, n):
        j = (task_id + k) % n
        tgt = target_object_for_prompt(suite_name, suite, j, scene_objects)
        if tgt is not None and tgt in scene_objects:
            return j
    return (task_id + 1) % n


def run_episode(cfg, env, task_label, model, resize_size, processor, action_head, proprio_projector, init_state,
                tracker, obj_every, seed, max_steps, num_steps_wait):
    set_seed_everywhere(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    eef, objpos, timeline = [], {}, []
    first_touch = first_grasp = None
    touched_any, grasped_any = set(), set()
    success = False
    t0 = time.time()
    t = 0
    n_queries = 0
    chunks_first = None
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        observation, _img = prepare_observation(obs, resize_size)
        if len(action_queue) == 0:
            actions = get_action(cfg, model, observation, task_label, processor=processor, action_head=action_head,
                                 proprio_projector=proprio_projector, noisy_action_projector=None, use_film=cfg.use_film)
            n_queries += 1
            if chunks_first is None:
                chunks_first = [[float(x) for x in a] for a in actions]
            action_queue.extend(actions)
        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        step = t - num_steps_wait
        eef.append([float(v) for v in obs["robot0_eef_pos"]])
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
        if done:
            success = True
            break
        t += 1
    objpos["final"] = tracker.object_positions()
    return {
        "success": bool(success),
        "n_steps": t - num_steps_wait + 1,
        "wall_s": round(time.time() - t0, 2),
        "n_queries": n_queries,
        "first_touch": first_touch,
        "first_grasp": first_grasp,
        "touched_any": sorted(touched_any),
        "grasped_any": sorted(grasped_any),
        "contact_timeline": timeline,
        "eef_pos": eef,
        "object_pos": objpos,
        "first_chunk": chunks_first,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task_ids", default="all")
    ap.add_argument("--init_ids", default="0-9")
    ap.add_argument("--conditions", default="correct,null,wrong_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
    ap.add_argument("--revision", default=None, help="recorded only; HF download pinned via HF cache")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--obj_every", type=int, default=10)
    ap.add_argument("--env_img_res", type=int, default=256)
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
    conditions = [c for c in args.conditions.split(",") if c]

    cfg = GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=args.suite,
                         num_open_loop_steps=NUM_ACTIONS_CHUNK, center_crop=True, seed=args.seed,
                         env_img_res=args.env_img_res)
    model, action_head, proprio_projector, _noisy, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = parse_ids(args.task_ids, suite.n_tasks)
    init_ids = parse_ids(args.init_ids, 10**6)
    max_steps = args.max_steps or TASK_MAX_STEPS[args.suite]
    print(f"[runner] suite={args.suite} tasks={task_ids} inits={init_ids} conds={conditions} out={out} "
          f"unnorm_key={cfg.unnorm_key} chunk={NUM_ACTIONS_CHUNK} max_steps={max_steps}", flush=True)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, correct_prompt = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        env.reset()
        tracker = ContactTracker(env)
        scene_objects = tracker.objects
        wrong_tid = choose_wrong_task(args.suite, suite, task_id, scene_objects)
        wrong_prompt = suite.get_task(wrong_tid).language
        target_obj = target_object_for_prompt(args.suite, suite, task_id, scene_objects)
        wrong_obj = target_object_for_prompt(args.suite, suite, wrong_tid, scene_objects)
        print(f"[runner] task {task_id} '{correct_prompt}' target={target_obj} wrong_task={wrong_tid} "
              f"'{wrong_prompt}' wrong_obj={wrong_obj} objects={scene_objects} n_init={len(init_states)}", flush=True)
        for init_id in init_ids:
            for cond in conditions:
                key = (args.suite, task_id, init_id, cond)
                if key in done_keys:
                    continue
                prompt = {"correct": correct_prompt, "null": "", "wrong_object": wrong_prompt}[cond]
                seed = args.seed + task_id * 1000 + init_id
                rec = run_episode(cfg, env, prompt, model, resize_size, processor, action_head, proprio_projector,
                                  init_states[init_id], tracker, args.obj_every, seed, max_steps, args.num_steps_wait)
                rec.update({
                    "suite": args.suite, "task_id": task_id, "task_name": task.name, "init_id": init_id,
                    "condition": cond, "prompt": prompt,
                    "full_prompt": PROMPT_TEMPLATE.format(task_label_lower=prompt.lower()),
                    "seed": seed, "target_object": target_obj,
                    "wrong_task_id": wrong_tid if cond == "wrong_object" else None,
                    "wrong_object": wrong_obj if cond == "wrong_object" else None,
                    "scene_objects": scene_objects, "policy": args.checkpoint, "revision": args.revision,
                    "n_action_steps": NUM_ACTIONS_CHUNK, "max_steps": max_steps, "tag": args.tag,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                ft = rec["first_touch"]["object"] if rec["first_touch"] else None
                fg = rec["first_grasp"]["object"] if rec["first_grasp"] else None
                print(f"[ep] {args.suite} t{task_id} i{init_id} {cond:<12} success={int(rec['success'])} "
                      f"steps={rec['n_steps']} first_touch={ft} grasp={fg} {rec['wall_s']}s", flush=True)
        env.close()
    print("[runner] DONE", flush=True)


if __name__ == "__main__":
    main()
