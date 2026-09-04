#!/usr/bin/env python
"""Pre-flight for the pi0.5 hook harness (protocol §4.1 + §1 VERIFY items). Writes a manifest JSON.

Checks: hook module paths; <bos>; image slots (768) vs valid image tokens (512 under empty_cameras=1); segment index
sets + decoded text (OBJ decodes to the object name); OSC delta-frame alignment; bitwise determinism with fixed noise;
cache shape/content invariance across the 10 denoise steps; in-place KV edit persists; FULL-KV SWAP reproduces the
prompt-B chunk to float tolerance; and one exploratory KV[seg] run to exercise the primitives.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

from hooks import LAYER_BANDS, Pi05Harness, axis_metric, cache_kv_lists, clone_cache
from lerobot.envs.libero import LiberoEnv, _get_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--objA", default="alphabet_soup_1")
    ap.add_argument("--objB", default="cream_cheese_1")
    ap.add_argument("--promptB", default="pick up the cream cheese and place it in the basket")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    ap.add_argument("--revision", default="8e174154ef5f6c60a8da12ae99c303d8963138c1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = {"args": vars(args), "checks": {}, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    H = Pi05Harness(args.policy, revision=args.revision, dtype=args.dtype)
    man["module_paths"] = H.module_paths
    man["config"] = {k: str(getattr(H.cfg, k)) for k in ("dtype", "chunk_size", "num_inference_steps", "n_action_steps",
                                                         "empty_cameras", "max_action_dim", "tokenizer_max_length")}
    suite = _get_suite(args.suite)
    env = LiberoEnv(task_suite=suite, task_id=args.task, task_suite_name=args.suite, obs_type="pixels_agent_pos",
                    observation_height=360, observation_width=360, init_states=True, episode_index=0, n_envs=1)
    promptA = suite.tasks[args.task].language
    rs = env._env.env

    # --- OSC delta frame check: command +x for 5 steps
    env.init_state_id = args.init
    obs, _ = env.reset(seed=0)
    p0 = np.array(rs.sim.data.site_xpos[rs.robots[0].eef_site_id]) if hasattr(rs.robots[0], "eef_site_id") else None
    e0 = np.array(env._env.env._get_observations()["robot0_eef_pos"])
    for _ in range(5):
        raw, *_ = env._env.step(np.array([1, 0, 0, 0, 0, 0, -1], dtype=np.float32))
    e1 = np.array(raw["robot0_eef_pos"])
    d = e1 - e0
    man["checks"]["osc_plus_x_eef_displacement_world"] = d.tolist()
    man["checks"]["osc_world_aligned_x"] = bool(abs(d[0]) > 3 * max(abs(d[1]), abs(d[2])) and d[0] > 0)

    # --- observation at init state (fresh reset)
    env.init_state_id = args.init
    obs, _ = env.reset(seed=0)
    pA = np.array(rs.sim.data.body_xpos[rs.obj_body_id[args.objA]])
    pB = np.array(rs.sim.data.body_xpos[rs.obj_body_id[args.objB]])
    man["objects"] = {"A": args.objA, "pA": pA.tolist(), "B": args.objB, "pB": pB.tolist()}

    bA = H.build_batch(obs, promptA)
    bB = H.build_batch(obs, args.promptB)
    man["prompts"] = {"A": H.prompt_text(bA), "B": H.prompt_text(bB)}
    ids = bA["observation.language.tokens"][0]
    man["checks"]["bos_first"] = H.tok.decode([int(ids[0])]) == "<bos>"

    prA = H.prefix_forward(bA)
    prB = H.prefix_forward(bB)
    man["checks"]["prefix_len"] = prA["prefix_len"]
    man["checks"]["n_img_slots"] = prA["n_img_slots"]
    man["checks"]["n_img_valid"] = prA["n_img_valid"]
    ks, vs = cache_kv_lists(prA["cache"])
    man["checks"]["cache_type"] = type(prA["cache"]).__name__
    man["checks"]["cache_layer_shape"] = list(ks[0].shape)
    man["checks"]["cache_n_layers"] = len(ks)
    n_img = prA["n_img_slots"]
    objA_name = args.objA[:-2].replace("_", " ")
    objB_name = args.objB[:-2].replace("_", " ")
    segA, decA = H.segments(bA, object_name=objA_name, n_img=n_img)
    segB, decB = H.segments(bB, object_name=objB_name, n_img=n_img)
    segA["IMG"] = list(range(prA["n_img_valid"]))
    segB["IMG"] = list(range(prB["n_img_valid"]))
    man["segments_A"] = {k: v for k, v in segA.items() if k != "IMG"}
    man["segments_A"]["IMG_n"] = len(segA["IMG"])
    man["decoded_A"] = decA
    man["decoded_B"] = decB
    man["checks"]["obj_decodes_A"] = objA_name in decA.get("OBJ", "")
    man["checks"]["obj_decodes_B"] = objB_name in decB.get("OBJ", "")
    man["checks"]["equal_token_count_AB"] = len(segA["TEXT_VALID"]) == len(segB["TEXT_VALID"])
    man["checks"]["state_positions_identical"] = segA["STATE"] == segB["STATE"]
    print("segments A:", {k: (len(v) if k == "IMG" else v) for k, v in segA.items()})
    print("decoded A:", decA)
    print("decoded B:", decB)

    noise = H.make_noise(seed=0)
    # --- determinism
    cA1, _ = H.action_forward(prA, noise)
    cA2, _ = H.action_forward(prA, noise)
    man["checks"]["bitwise_deterministic"] = bool(torch.equal(cA1, cA2))
    cB, _ = H.action_forward(prB, noise)
    man["checks"]["A_vs_B_chunk_max_abs_diff_model_units"] = float((cA1 - cB).abs().max())

    # --- in-place edit persists + full KV swap == prompt swap
    cacheAB = clone_cache(prA["cache"])
    H.kv_swap(cacheAB, prB["cache"], positions=list(range(prA["prefix_len"])), layers=list(range(18)))
    k_ab, _ = cache_kv_lists(cacheAB)
    k_b, _ = cache_kv_lists(prB["cache"])
    man["checks"]["inplace_edit_persists"] = bool(torch.equal(k_ab[7], k_b[7]))
    prAB = dict(prA)
    prAB["cache"] = cacheAB
    prAB["prefix_pad_masks"] = prB["prefix_pad_masks"]
    cAB, _ = H.action_forward(prAB, noise)
    diff = float((cAB - cB).abs().max())
    man["checks"]["full_kv_swap_max_abs_diff_vs_promptB"] = diff
    man["checks"]["full_kv_swap_reproduces_promptB"] = diff < (1e-4 if args.dtype == "float32" else 1e-2)
    # A-cache with B's pad mask only (no KV change) must still equal A (masks identical when token counts match)
    prA2 = dict(prA)
    prA2["prefix_pad_masks"] = prB["prefix_pad_masks"]
    cA3, _ = H.action_forward(prA2, noise)
    man["checks"]["padmask_swap_alone_changes_chunk"] = float((cA3 - cA1).abs().max())

    # --- exploratory: one KV[seg] run per segment (all layers) + RD baseline, metric on A-B axis
    envA = H.unnormalize(cA1)[0].cpu().numpy()
    envB = H.unnormalize(cB)[0].cpu().numpy()
    mA, mB = axis_metric(envA, pA, pB), axis_metric(envB, pA, pB)
    res = {"clean_A": mA, "clean_B_fullswap": mB}
    for seg in ("INSTR", "OBJ", "FMT", "IMG", "INSTR+FMT", "IMG+FMT"):
        pos = sorted(set(sum([segA[s] for s in seg.split("+")], [])))
        c = clone_cache(prA["cache"])
        H.kv_swap(c, prB["cache"], pos, LAYER_BANDS["all"])
        ch, _ = H.action_forward({**prA, "cache": c}, noise)
        res[f"KV[{seg}]"] = axis_metric(H.unnormalize(ch)[0].cpu().numpy(), pA, pB)
        c = clone_cache(prA["cache"])
        H.rd_patch(c, pos, LAYER_BANDS["all"], seed=1)
        ch, _ = H.action_forward({**prA, "cache": c}, noise)
        res[f"RD[{seg}]"] = axis_metric(H.unnormalize(ch)[0].cpu().numpy(), pA, pB)
    # noise-seed SD for clean A (K=5)
    ms = []
    for s in range(5):
        ch, _ = H.action_forward(prA, H.make_noise(seed=100 + s))
        ms.append(axis_metric(H.unnormalize(ch)[0].cpu().numpy(), pA, pB))
    res["clean_A_noise_seeds"] = ms
    res["SD_noise"] = float(np.std(ms, ddof=1))
    res["denominator_gate_|mB-mA|>=3SD"] = bool(abs(mB - mA) >= 3 * res["SD_noise"])
    man["exploratory_metric_one_init"] = res
    # residual capture check + expert hook check (identity hook must reproduce baseline)
    prC = H.prefix_forward(bA, capture=True)
    man["checks"]["vlm_capture_layers"] = len(prC["resid"])
    man["checks"]["vlm_resid_shape"] = list(prC["resid"][0].shape)
    ident = {L: (lambda h, step: h) for L in range(18)}
    cI, caps = H.action_forward(prA, noise, expert_hooks=ident, capture=True)
    man["checks"]["identity_expert_hook_reproduces_chunk"] = bool(torch.equal(cI, cA1))
    man["checks"]["expert_capture_calls_per_layer"] = len(caps[0])
    man["checks"]["expert_resid_shape"] = list(caps[0][0].shape)
    # VLM residual identity hook through the cache must also reproduce
    prI = H.prefix_forward(bA, resid_hooks={L: (lambda h: h) for L in range(18)})
    cI2, _ = H.action_forward(prI, noise)
    man["checks"]["identity_vlm_hook_reproduces_chunk"] = bool(torch.equal(cI2, cA1))
    man["gpu_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    (out / "preflight_manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man["checks"], indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
