#!/usr/bin/env python
"""Stage 3 CONTROL PILOT: steer WHICH instruction pi0.5 obeys by editing ONLY image-position prefix KV.

Stage 2 established that the instruction's causal route to the action expert runs entirely through the
IMAGE-position keys/values of the prefix KV cache, concentrated in VLM layers 12-17 (KV[IMG]@12-17 recovers
R = 0.72-1.02 of the full prompt swap; KV[INSTR] ~ 0).  This script asks the closed-loop question:

    can we change WHICH instruction the policy obeys by editing ONLY image-position KV
    (positions that contain no instruction tokens)?

Design
------
Prompt A is the instruction actually in the prompt for the whole rollout.  At every policy replan (every
n_action_steps=10 env steps) we run the prefix a SECOND time on the SAME observation with instruction B, take
    d = KV_B[IMG] - KV_A[IMG]   (per layer, per image position; layers 12-17, valid image positions only)
and add lambda * d in place to the prefix KV cache that the action expert will read.  Instruction-token
positions, format positions and state positions are never touched.  lambda = 1 is exactly the KV[IMG]@12-17 swap
of Stage 2; lambda = 0 is a no-op and must reproduce the unedited rollout bit-for-bit.

The edit is delivered by wrapping `paligemma_with_expert.forward`: the prefix pass (past_key_values=None,
use_cache=True, inputs_embeds[1] is None) returns the cache, and we mutate it in place before the 10 Euler
steps of the expert read it.  Everything else -- env, contact tracking, action queue, noise RNG -- is the stock
Stage 0 runner path, so lambda=0 is comparable to the Stage 0 baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


def wait_for_gpu(min_free_gb=10.0, poll=60):
    """Another agent shares this card; never kill anything, just wait."""
    import subprocess
    while True:
        try:
            free = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
        except Exception:
            return
        if free / 1024.0 >= min_free_gb:
            print(f"[gpu] {free} MiB free -> proceeding", flush=True)
            return
        print(f"[gpu] only {free} MiB free, waiting {poll}s", flush=True)
        time.sleep(poll)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from run_libero_prompt_conditions import (  # reuse the working Stage 0 harness verbatim
    ContactTracker,
    batchify,
    load_policy,
    make_env,
    target_object_for_prompt,
)

TEXT_LEN = 200
EDIT_LAYERS = list(range(12, 18))  # Stage 2: the instruction is written into image-position K/V in VLM 12-17


def cache_kv_lists(cache):
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    if hasattr(cache, "layers"):
        return [l.keys for l in cache.layers], [l.values for l in cache.layers]
    if isinstance(cache, (list, tuple)):
        return [c[0] for c in cache], [c[1] for c in cache]
    raise TypeError(f"unknown cache type {type(cache)}")


class KVImageEditor:
    """Adds lambda * (KV_B - KV_A) at image positions / layers 12-17 of the prefix cache, in place."""

    def __init__(self, policy, layers=EDIT_LAYERS):
        self.policy = policy
        self.model = policy.model
        self.pwe = self.model.paligemma_with_expert
        self.layers = list(layers)
        self._orig = self.pwe.forward
        self.lam = 0.0
        self.enabled = False           # False => the wrapper is a pure pass-through (unedited reference path)
        self.armed = False
        self.busy = False              # re-entrancy guard for our own prefix passes
        self.cacheB = None
        self.img_pos = None
        self.n_prefix = None
        self.n_edits = 0
        self.pwe.forward = self._wrapped

    def close(self):
        self.pwe.forward = self._orig

    # -- our own prefix pass (no RNG consumed, guarded so the wrapper does not recurse)
    def prefix_cache(self, batch):
        m = self.model
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, tokens, masks)
        att2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att4d = m._prepare_attention_masks_4d(att2d)
        self.pwe.paligemma.language_model.config._attn_implementation = "eager"
        self.busy = True
        try:
            with torch.no_grad():
                _, cache = self._orig(attention_mask=att4d, position_ids=pos, past_key_values=None,
                                      inputs_embeds=[prefix_embs, None], use_cache=True)
        finally:
            self.busy = False
        n_img_slots = prefix_embs.shape[1] - TEXT_LEN
        n_img_valid = int(prefix_pad_masks[0, :n_img_slots].sum())
        return cache, int(prefix_embs.shape[1]), n_img_slots, n_img_valid

    def arm(self, batch_B):
        """Compute and stash the B-instruction prefix cache for the CURRENT observation."""
        cache, n_prefix, n_img_slots, n_img_valid = self.prefix_cache(batch_B)
        self.cacheB = cache
        self.n_prefix = n_prefix
        self.img_pos = torch.arange(n_img_valid, dtype=torch.long, device=cache_kv_lists(cache)[0][0].device)
        self.armed = True
        return {"n_prefix": n_prefix, "n_img_slots": n_img_slots, "n_img_valid": n_img_valid}

    def _wrapped(self, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                 use_cache=None, adarms_cond=None):
        out = self._orig(attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                         adarms_cond=adarms_cond)
        if self.busy or not self.enabled or not self.armed:
            return out
        is_prefix_pass = (past_key_values is None and use_cache and inputs_embeds is not None
                          and inputs_embeds[0] is not None and inputs_embeds[1] is None)
        if not is_prefix_pass:
            return out
        cacheA = out[1]
        kA, vA = cache_kv_lists(cacheA)
        kB, vB = cache_kv_lists(self.cacheB)
        assert kA[0].shape[2] == self.n_prefix == kB[0].shape[2], (kA[0].shape, self.n_prefix, kB[0].shape)
        idx = self.img_pos
        lam = float(self.lam)
        # fp32 lerp with exact endpoints: the cache is bf16, and computing kA + lam*(kB - kA) in bf16 is lossy
        # (lam=1 would NOT reproduce kB, lam=0 would still round-trip).  Doing it in fp32 and casting back makes
        # lam=0 bitwise kA and lam=1 bitwise kB.
        with torch.no_grad():
            for L in self.layers:
                for T, S in ((kA[L], kB[L]), (vA[L], vB[L])):
                    a = T[:, :, idx, :].float()
                    b = S[:, :, idx, :].float()
                    T[:, :, idx, :] = torch.lerp(a, b, lam).to(T.dtype)
        self.n_edits += 1
        self.armed = False
        # NOTE: cacheB is deliberately NOT freed here.  Freeing ~18 MB inside the prefix pass changes the CUDA
        # allocator state for the expert's 10 Euler steps and perturbs bf16 GEMMs at the ULP level, which was
        # enough to make the lam=0 rollout diverge from the unedited one.  run_episode clears it instead.
        return out


def run_episode(env, policy, pre, post, epre, epost, editor, promptA, promptB, lam, edit_on, init_id, seed,
                max_steps, tracker, obj_every):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy.reset()
    editor.lam = lam
    editor.enabled = bool(edit_on)
    editor.armed = False
    editor.cacheB = None
    editor.n_edits = 0
    env.init_state_id = init_id
    obs, _ = env.reset(seed=seed)
    rs = env._env
    eef, timeline = [], []
    first_touch = first_grasp = None
    touched_any, grasped_any = set(), set()
    prefix_info = None
    t0 = time.time()
    success = False
    step = 0
    for step in range(max_steps):
        o = preprocess_observation(batchify(obs))
        o["task"] = [promptA]
        o = epre(o)
        o = pre(o)
        if edit_on and len(policy._action_queue) == 0:
            oB = preprocess_observation(batchify(obs))
            oB["task"] = [promptB]
            oB = epre(oB)
            oB = pre(oB)
            info = editor.arm(oB)
            prefix_info = prefix_info or info
        with torch.inference_mode():
            a = policy.select_action(o)
        editor.cacheB = None
        a = post(a)
        a = epost({ACTION: a})[ACTION]
        a = a.cpu().numpy()[0]
        raw_obs, reward, done, info = rs.step(a)
        obs = env._format_raw_obs(raw_obs)
        eef.append([round(float(v), 6) for v in raw_obs["robot0_eef_pos"]])
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
        success = bool(rs.check_success())
        if success or done:
            break
    editor.enabled = False
    return {
        "success": success, "n_steps": step + 1, "wall_s": round(time.time() - t0, 2),
        "first_touch": first_touch, "first_grasp": first_grasp,
        "touched_any": sorted(touched_any), "grasped_any": sorted(grasped_any),
        "contact_timeline": timeline, "eef_pos": eef, "n_edits": editor.n_edits, "prefix_info": prefix_info,
    }


# ------------------------------------------------------------------ cells
# (task_id, cell, A = prompt in the rollout, B = the instruction whose image-KV signature we add)
# cell (a) IGNORE->OBEY : A = the task's own correct instruction, B = the wrong instruction it normally ignores
# cell (b) OBEY->IGNORE : A = the wrong instruction it normally obeys, B = the task's correct instruction
# cell (c) ketchup ctrl : tasks 2 and 3 both take A = "pick up the ketchup ..." (t4) but behave oppositely
CELLS = [
    ("a_ignore_to_obey", 4, "correct", 7),   # A=ketchup(own), B=milk
    ("a_ignore_to_obey", 6, "correct", 8),   # A=butter(own), B=chocolate pudding
    ("a_ignore_to_obey", 7, "correct", 8),   # A=milk(own),   B=chocolate pudding
    ("b_obey_to_ignore", 1, "wrong", 5),     # A=tomato sauce(wrong), B=cream cheese(own)
    ("b_obey_to_ignore", 2, "wrong", 4),     # A=ketchup(wrong),      B=salad dressing(own)
    ("c_ketchup", 3, "wrong", 4),            # A=ketchup(wrong),      B=bbq sauce(own)   [t2 above is the twin]
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--init_ids", default="0-9")
    ap.add_argument("--lambdas", default="0,0.25,0.5,1.0,2.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    ap.add_argument("--revision", default="8e174154ef5f6c60a8da12ae99c303d8963138c1")
    ap.add_argument("--n_action_steps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--obj_every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sanity_only", action="store_true", help="only run the lambda=0 == unedited assertion")
    ap.add_argument("--sanity_inits", default="0,1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    epf = out / "per_episode.jsonl"
    done = set()
    if epf.exists():
        for line in epf.open():
            try:
                r = json.loads(line)
                done.add((r["cell"], r["task_id"], r["init_id"], r["lam"]))
            except Exception:
                pass

    a0, b0 = args.init_ids.split("-")
    init_ids = list(range(int(a0), int(b0) + 1))
    lambdas = [float(x) for x in args.lambdas.split(",")]
    wait_for_gpu(10.0)
    policy, pcfg, pre, post, epre, epost = load_policy(args.policy, args.revision, args.n_action_steps, args.device)
    editor = KVImageEditor(policy)
    suite = _get_suite(args.suite)
    max_steps = args.max_steps or TASK_SUITE_MAX_STEPS.get(args.suite, 500)
    manifest = {"args": vars(args), "edit_layers": EDIT_LAYERS, "cells": [],
                "dtype": str(pcfg.dtype), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    fh = epf.open("a")

    # ------------------------------------------------------------ sanity: lambda=0 must reproduce unedited
    print("[pilot] SANITY: (i) unedited vs unedited (baseline repeatability), "
          "(ii) unedited vs lambda=0 (full edit machinery, zero coefficient)", flush=True)
    sanity = []
    tid, cellname, whichA, tB = 4, "a_ignore_to_obey", "correct", 7
    env = make_env(suite, args.suite, tid)
    tracker = ContactTracker(env._env)
    pA_text, pB_text = suite.tasks[tid].language, suite.tasks[tB].language

    def cmp(r1, r2):
        d = {"n1": r1["n_steps"], "n2": r2["n_steps"],
             "same_success": r1["success"] == r2["success"],
             "same_first_touch": r1["first_touch"] == r2["first_touch"],
             "same_n_steps": r1["n_steps"] == r2["n_steps"]}
        n = min(r1["n_steps"], r2["n_steps"])
        a, b = np.array(r1["eef_pos"][:n]), np.array(r2["eef_pos"][:n])
        dd = np.abs(a - b).max(axis=1)
        nz = np.nonzero(dd > 0)[0]
        d["max_abs_eef_diff"] = float(dd.max()) if n else None
        d["first_diff_step"] = int(nz[0]) if len(nz) else None
        d["identical"] = bool(d["same_success"] and d["same_first_touch"] and d["same_n_steps"]
                              and d["max_abs_eef_diff"] == 0.0)
        return d

    for init in [int(x) for x in args.sanity_inits.split(",")]:
        seed = args.seed + tid * 1000 + init
        r_p1 = run_episode(env, policy, pre, post, epre, epost, editor, pA_text, pB_text, 0.0, False,
                           init, seed, max_steps, tracker, args.obj_every)
        r_p2 = run_episode(env, policy, pre, post, epre, epost, editor, pA_text, pB_text, 0.0, False,
                           init, seed, max_steps, tracker, args.obj_every)
        r_l0 = run_episode(env, policy, pre, post, epre, epost, editor, pA_text, pB_text, 0.0, True,
                           init, seed, max_steps, tracker, args.obj_every)
        rec = {"init": init, "plain_vs_plain": cmp(r_p1, r_p2), "plain_vs_lam0": cmp(r_p1, r_l0),
               "n_edits_lam0": r_l0["n_edits"], "prefix_info": r_l0["prefix_info"]}
        rec["identical"] = rec["plain_vs_lam0"]["identical"]
        sanity.append(rec)
        print(f"[sanity] init {init}: plain-vs-plain {rec['plain_vs_plain']} | "
              f"plain-vs-lam0 {rec['plain_vs_lam0']} | n_edits={r_l0['n_edits']} "
              f"prefix={r_l0['prefix_info']}", flush=True)
    env.close()
    manifest["sanity"] = sanity
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    base_repeatable = all(s["plain_vs_plain"]["identical"] for s in sanity)
    lam0_ok = all(s["plain_vs_lam0"]["identical"] for s in sanity)
    if not lam0_ok:
        if not base_repeatable:
            print("[pilot] SANITY: the UNEDITED rollout is not repeatable either -> the non-determinism is in the "
                  "baseline closed loop (env/CUDA), not in the KV-edit path.", flush=True)
        else:
            print("[pilot] FATAL: lambda=0 does NOT reproduce the unedited rollout although the unedited rollout "
                  "IS repeatable -> harness bug, stopping.", flush=True)
            sys.exit(3)
    else:
        print("[pilot] SANITY PASSED", flush=True)
    if args.sanity_only:
        return

    # ------------------------------------------------------------ cells
    for cellname, tid, whichA, tB in CELLS:
        env = make_env(suite, args.suite, tid)
        tracker = ContactTracker(env._env)
        scene_objects = tracker.objects
        # A is either the task's own instruction (cell a) or the wrong instruction it normally obeys (cells b/c)
        if whichA == "correct":
            A_tid, B_tid = tid, tB
        else:
            A_tid, B_tid = tB, tid
        pA_text, pB_text = suite.tasks[A_tid].language, suite.tasks[B_tid].language
        objA = target_object_for_prompt(args.suite, suite, A_tid, scene_objects)
        objB = target_object_for_prompt(args.suite, suite, B_tid, scene_objects)
        assert objA in scene_objects and objB in scene_objects, (objA, objB, scene_objects)
        info = {"cell": cellname, "task_id": tid, "A_task": A_tid, "B_task": B_tid, "promptA": pA_text,
                "promptB": pB_text, "objA": objA, "objB": objB, "scene_objects": scene_objects}
        manifest["cells"].append(info)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[pilot] {cellname} t{tid}: A='{pA_text}'({objA})  B='{pB_text}'({objB})", flush=True)
        for init in init_ids:
            for lam in lambdas:
                key = (cellname, tid, init, lam)
                if key in done:
                    continue
                seed = args.seed + tid * 1000 + init
                rec = run_episode(env, policy, pre, post, epre, epost, editor, pA_text, pB_text, lam, True,
                                  init, seed, max_steps, tracker, args.obj_every)
                ft = rec["first_touch"]["object"] if rec["first_touch"] else None
                rec.update(info)
                rec.update({"init_id": init, "lam": lam, "seed": seed, "suite": args.suite,
                            "first_touch_obj": ft,
                            "touch_class": ("A" if ft == objA else "B" if ft == objB else
                                            "none" if ft is None else "third"),
                            "policy": args.policy, "revision": args.revision, "max_steps": max_steps,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[ep] {cellname} t{tid} i{init} lam={lam:<5} touch={ft} ({rec['touch_class']}) "
                      f"success={int(rec['success'])} steps={rec['n_steps']} {rec['wall_s']}s", flush=True)
        env.close()
    print("[pilot] DONE", flush=True)


if __name__ == "__main__":
    main()
