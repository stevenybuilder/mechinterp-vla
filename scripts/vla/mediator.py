#!/usr/bin/env python
"""SOURCE-SPECIFIC MEDIATOR TEST — the experiment that separates routing from bottleneck-clamping.

The corrected pin control restores ALL 512 image positions to clean-A across 6 live layers. That is an
enormous intervention: those late states already encode the motor program, so forcing them to A could
restore the A action whether or not they specifically carry instruction content. P_05 rules out the
trivial "any clamp anywhere pins the output"; it does NOT rule out clamping a downstream motor
bottleneck.

This does the surgical version. Let d_B = h_B - h_A at image positions (the B-INDUCED DIFFERENCE).
After the early A->B instruction swap:
  SUBTRACT   h <- h - d_B   at IMG, live layers   -> if instruction content rides on d_B, returns to A
  INSERT     h <- h + d_B   at IMG, live layers, on a CLEAN-A run (no swap) -> should move toward B
Controls, both equal-norm to d_B:
  ORTHO      a random direction projected orthogonal to d_B          -> should do nothing
  UNREL      d_C = h_C - h_A from an unrelated prompt C              -> should not move toward B

Reading:
  SUBTRACT returns to A and INSERT moves to B, while ORTHO/UNREL do neither
     -> the instruction's effect rides on a source-specific component: ROUTING.
  SUBTRACT returns to A but INSERT does nothing, or ORTHO also returns to A
     -> we are perturbing a motor bottleneck, not removing instruction content: NOT routing.
"""
import argparse, json, os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from robosuite.utils import binding_utils
from build_stage1_twins import _patched_read_pixels
binding_utils.MjRenderContext.read_pixels = _patched_read_pixels
from hooks import Pi05Harness, axis_metric
from lerobot.envs.libero import LiberoEnv, _get_suite
from stage2_discovery import GOAL_OBJ

EARLY = list(range(0, 6))
LIVE = list(range(12, 18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_file", default="/root/vla/artifacts/mediation_pairs.json")
    ap.add_argument("--init_ids", default="0-3"); ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_goal")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = (out / "rows.jsonl").open("a")
    lo, hi = a.init_ids.split("-"); inits = list(range(int(lo), int(hi) + 1))
    pairs = json.load(open(a.pairs_file))
    H = Pi05Harness(revision="8e174154ef5f6c60a8da12ae99c303d8963138c1", dtype="float32")
    suite = _get_suite(a.suite)
    prompts = [suite.tasks[i].language for i in range(len(suite.tasks))]

    for tA, tB, tC in pairs:
        cell = f"{a.suite}_t{tA}_vs_t{tB}"
        env = LiberoEnv(task_suite=suite, task_id=tA, task_suite_name=a.suite, obs_type="pixels_agent_pos",
                        observation_height=360, observation_width=360, init_states=True, episode_index=0, n_envs=1)
        rs = env._env.env
        objA, nameA = GOAL_OBJ[tA]; objB, nameB = GOAL_OBJ[tB]
        print(f"\n[cell] {cell}", flush=True)
        for init in inits:
            env.init_state_id = init
            obs, _ = env.reset(seed=1000 + init)
            pA = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objA]])
            pB = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objB]])
            bA, bB, bC = (H.build_batch(obs, prompts[t]) for t in (tA, tB, tC))
            prA = H.prefix_forward(bA, capture=True)
            prB = H.prefix_forward(bB, capture=True)
            prC = H.prefix_forward(bC, capture=True)
            segA, _ = H.segments(bA, object_name=nameA, n_img=prA["n_img_slots"])
            segB, _ = H.segments(bB, object_name=nameB, n_img=prA["n_img_slots"])
            iA, iB = segA["INSTR"], segB["INSTR"]
            if len(iA) != len(iB): continue
            IMG = list(range(prA["n_img_valid"]))
            rA, rB, rC = prA["resid"], prB["resid"], prC["resid"]
            noise = H.make_noise(seed=init * 100)
            chA = H.unnormalize(H.action_forward(prA, noise)[0])[0].cpu().numpy()
            chB = H.unnormalize(H.action_forward(prB, noise)[0])[0].cpu().numpy()
            dAB = float(np.linalg.norm(chA[:10] - chB[:10])) or 1e-9
            g = torch.Generator(device="cpu").manual_seed(4242 + init)

            # the B-induced difference at image positions, per live layer
            dB = {L: (rB[L][:, IMG, :] - rA[L][:, IMG, :]) for L in LIVE}
            dC = {L: (rC[L][:, IMG, :] - rA[L][:, IMG, :]) for L in LIVE}
            orth = {}
            for L in LIVE:
                d = dB[L]; r = torch.randn(d.shape, generator=g)
                r = r - (r * d).sum(-1, keepdim=True) / (d * d).sum(-1, keepdim=True).clamp_min(1e-9) * d
                orth[L] = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-9) * d.norm(dim=-1, keepdim=True)
            for L in LIVE:  # equal-norm the unrelated difference to d_B
                dC[L] = dC[L] / dC[L].norm(dim=-1, keepdim=True).clamp_min(1e-9) * dB[L].norm(dim=-1, keepdim=True)

            swap = {L: (lambda h, L=L: _put(h, iA, rB[L])) for L in EARLY}

            def _put(h, pos, src):
                h = h.clone(); h[:, pos, :] = src[:, pos, :].to(h.dtype).to(h.device); return h

            def add(delta, sign):
                def mk(L):
                    def hook(h, L=L):
                        h = h.clone()
                        h[:, IMG, :] = h[:, IMG, :] + sign * delta[L].to(h.dtype).to(h.device)
                        return h
                    return hook
                return {L: mk(L) for L in LIVE}

            conds = {
                "clean_A":   (bA, None),
                "clean_B":   (bB, None),
                "swap_only": (bA, dict(swap)),
                "SUBTRACT_dB":  (bA, {**swap, **add(dB, -1)}),
                "SUBTRACT_ortho": (bA, {**swap, **add(orth, -1)}),
                "SUBTRACT_unrel": (bA, {**swap, **add(dC, -1)}),
                "INSERT_dB":    (bA, add(dB, +1)),
                "INSERT_ortho": (bA, add(orth, +1)),
                "INSERT_unrel": (bA, add(dC, +1)),
            }
            for nm, (batch, hooks) in conds.items():
                pr = H.prefix_forward(batch, resid_hooks=hooks) if hooks else (prA if nm == "clean_A" else prB)
                ec = H.unnormalize(H.action_forward(pr, noise)[0])[0].cpu().numpy()
                f.write(json.dumps({
                    "cell": cell, "init": init, "condition": nm,
                    "nL2_to_A": float(np.linalg.norm(ec[:10] - chA[:10])) / dAB,
                    "nL2_to_B": float(np.linalg.norm(ec[:10] - chB[:10])) / dAB,
                    "R": (axis_metric(ec, pA, pB, 10) - axis_metric(chA, pA, pB, 10)) /
                         ((axis_metric(chB, pA, pB, 10) - axis_metric(chA, pA, pB, 10)) or 1e-9),
                }) + "\n")
            f.flush()
            print(f"  init {init} done", flush=True)
        env.close()
    print("\nMEDIATOR DONE ->", out, flush=True)


if __name__ == "__main__":
    main()
