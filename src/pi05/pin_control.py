#!/usr/bin/env python
"""IS THE MEDIATION A ROUTING RESULT, OR DID WE JUST PIN THE OUTPUT?

C4 showed: swap the instruction at layers 0-5 (C2) AND hold image positions at their clean-A values
through layers 12-17 -> the instruction's effect vanishes (12/12 cells, p=2.4e-4).

The objection: if image positions at 12-17 ARE the motor plan (which our copy evidence suggests), then
clamping them pins the action, and "the effect vanished" is near-tautological. "Cut the wire" and
"hold the output still" make the same prediction.

Discriminating controls, all = C2 plus a clamp of image positions to clean-A, varying WHERE:

  P_1217  clamp at layers 12-17          -- the original C4 (should block)
  P_05    clamp at layers 0-5            -- KV[IMG]@0-5 was measured at 0.000, i.e. causally inert
                                            there. If clamping HERE also blocks, it is pinning, not
                                            routing, because an inert site cannot carry the effect.
  P_611   clamp at layers 6-11           -- the handoff-between band
  P_half  clamp 256 random image positions at 12-17 (count-matched to half)
  P_none  C2 alone, no clamp             -- reference

Prediction if ROUTING: P_05 does NOT block (lands on B like C2), P_1217 does.
Prediction if PINNING: every clamp blocks regardless of layer.
"""
import argparse, json, os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from robosuite.utils import binding_utils
from build_stage1_twins import _patched_read_pixels
binding_utils.MjRenderContext.read_pixels = _patched_read_pixels
from hooks import Pi05Harness, axis_metric
from lerobot.envs.libero import LiberoEnv, _get_suite
from stage2_discovery import GOAL_OBJ

EARLY = list(range(0, 6))
BANDS = {"P_05": list(range(0, 6)), "P_611": list(range(6, 12)), "P_1217": list(range(12, 18))}


def put(h, pos, src):
    h = h.clone(); h[:, pos, :] = src[:, pos, :].to(h.dtype).to(h.device); return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_file", default="artifacts/mediation_pairs.json")
    ap.add_argument("--init_ids", default="0-3")
    ap.add_argument("--out", required=True); ap.add_argument("--suite", default="libero_goal")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = (out / "rows.jsonl").open("a")
    lo, hi = a.init_ids.split("-"); inits = list(range(int(lo), int(hi) + 1))
    pairs = json.load(open(a.pairs_file))
    H = Pi05Harness(revision="8e174154ef5f6c60a8da12ae99c303d8963138c1", dtype="float32")
    suite = _get_suite(a.suite)
    prompts = [suite.tasks[i].language for i in range(len(suite.tasks))]

    for tA, tB, _c in pairs:
        cell = f"{a.suite}_t{tA}_vs_t{tB}"
        env = LiberoEnv(task_suite=suite, task_id=tA, task_suite_name=a.suite, obs_type="pixels_agent_pos",
                        observation_height=360, observation_width=360, init_states=True, episode_index=0, n_envs=1)
        rs = env._env.env
        objA, nameA = GOAL_OBJ[tA]; objB, nameB = GOAL_OBJ[tB]
        pAt, pBt = prompts[tA], prompts[tB]
        print(f"\n[cell] {cell}", flush=True)
        for init in inits:
            env.init_state_id = init
            obs, _ = env.reset(seed=1000 + init)
            pA = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objA]])
            pB = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objB]])
            bA, bB = H.build_batch(obs, pAt), H.build_batch(obs, pBt)
            prA = H.prefix_forward(bA, capture=True); prB = H.prefix_forward(bB, capture=True)
            n_img = prA["n_img_slots"]
            segA, _ = H.segments(bA, object_name=nameA, n_img=n_img)
            segB, _ = H.segments(bB, object_name=nameB, n_img=n_img)
            iA, iB = segA["INSTR"], segB["INSTR"]
            if len(iA) != len(iB): continue
            IMG = list(range(prA["n_img_valid"]))
            rA, rB = prA["resid"], prB["resid"]
            noise = H.make_noise(seed=init * 100)
            chA = H.unnormalize(H.action_forward(prA, noise)[0])[0].cpu().numpy()
            chB = H.unnormalize(H.action_forward(prB, noise)[0])[0].cpu().numpy()
            dAB = float(np.linalg.norm(chA[:10] - chB[:10])) or 1e-9
            rng = np.random.default_rng(999 + init)
            half = sorted(rng.choice(len(IMG), len(IMG) // 2, replace=False).tolist())

            # BUGFIX: dict-merging the clamp over the swap SILENTLY DROPS the swap wherever the
            # layer ranges overlap. For P_05 the band IS EARLY, so the old code ran a clean-A pass
            # with clean-A image values clamped onto it -- a pure identity (48/48 rows at exactly
            # 0.000 from clean-A). Compose instead: instruction positions (iA) and image positions
            # (IMG) are disjoint, so applying both edits at a shared layer is well defined.
            def build(band, clamp_pos):
                d = {}
                for L in sorted(set(EARLY) | set(band)):
                    do_swap = L in EARLY
                    do_clamp = L in band
                    def hook(h, L=L, s=do_swap, c=do_clamp, cp=clamp_pos):
                        if s: h = put(h, iA, rB[L])
                        if c: h = put(h, cp, rA[L])
                        return h
                    d[L] = hook
                return d

            conds = {"P_none": build([], IMG)}
            for nm, band in BANDS.items():
                conds[nm] = build(band, IMG)
            conds["P_half"] = build(list(range(12, 18)), half)

            for nm, hooks in conds.items():
                pr = H.prefix_forward(bA, resid_hooks=hooks)
                ec = H.unnormalize(H.action_forward(pr, noise)[0])[0].cpu().numpy()
                nb = float(np.linalg.norm(ec[:10] - chB[:10])) / dAB
                na = float(np.linalg.norm(ec[:10] - chA[:10])) / dAB
                f.write(json.dumps({"cell": cell, "init": init, "condition": nm,
                                    "nL2_to_B": nb, "nL2_to_A": na,
                                    "R": (axis_metric(ec, pA, pB, 10) - axis_metric(chA, pA, pB, 10)) /
                                         ((axis_metric(chB, pA, pB, 10) - axis_metric(chA, pA, pB, 10)) or 1e-9),
                                    "blocked": bool(nb >= 0.5)}) + "\n")
            f.flush()
            row = " ".join(f"{k}={json.loads(open(out/'rows.jsonl').read().strip().splitlines()[-len(conds)+i])['nL2_to_B']:.2f}"
                           for i, k in enumerate(conds))
            print(f"  init {init}: nL2->B  {row}", flush=True)
        env.close()
    print("\nPIN CONTROL DONE ->", out, flush=True)


if __name__ == "__main__":
    main()
