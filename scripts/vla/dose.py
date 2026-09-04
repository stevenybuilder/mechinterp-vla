#!/usr/bin/env python
"""DOSE-RESPONSE: is the carrier LOCALIZED to the named object, or just proportional to how many
image positions you transplant?

Motivation. In run1 the whole-image carrier (512 positions) redirects strongly while the
segmentation-localized carrier (2-9 positions covering B's object) does essentially nothing. That is
ambiguous between two readings, and they have opposite consequences:
  (a) TRAJECTORY COPYING -- the effect needs a wholesale transplant and scales with the amount of
      donor activation moved, regardless of WHICH positions;
  (b) DISTRIBUTED CARRIER -- the instruction's action-facing content lives across many image positions
      rather than on the named object's patches.

The discriminator is a COUNT-MATCHED comparison at every dose:
  LOC[n]  : n positions drawn from B's object patches outward (object-centred)
  RAND[n] : n positions drawn uniformly at random from the valid image positions
If LOC[n] > RAND[n] at matched n, position identity matters -> concept transfer.
If LOC[n] == RAND[n] and both rise with n, only the AMOUNT matters -> (a), trajectory copying.

Metric is L2-to-target, not R: R is not specific (the equal-norm random carrier reached R=+0.82 in
run1 while sitting at L2->B = 0.79).
"""
import argparse, json, os, time
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

CARRIER = list(range(12, 18))
HW, G = 360, 16
DOSES = [4, 8, 16, 32, 64, 128, 256, 512]


def patches_of(seg, geom2obj, names, dilate=1):
    objid = seg[:, :, 1]; cell = HW / G; out = set()
    for r in range(G):
        for c in range(G):
            blk = objid[int(r * cell):int((r + 1) * cell), int(c * cell):int((c + 1) * cell)]
            for g in np.unique(blk):
                if geom2obj.get(int(g)) in names: out.add((r, c))
    for _ in range(dilate):
        out |= {(r + dr, c + dc) for (r, c) in list(out) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if 0 <= r + dr < G and 0 <= c + dc < G}
    return sorted(r * G + c for r, c in out)


def grow(seed_pos, k, valid):
    """object-centred: seed patches first, then nearest remaining positions by grid distance"""
    if not seed_pos: return sorted(valid)[:k]
    sel = [p for p in seed_pos if p in valid][:k]
    if len(sel) >= k: return sel[:k]
    cen = np.mean([[p // G, p % G] for p in sel], axis=0)
    rest = sorted(set(valid) - set(sel), key=lambda p: (p // G - cen[0])**2 + (p % G - cen[1])**2)
    return sel + rest[:k - len(sel)]


def put(h, pos, src):
    h = h.clone(); h[:, pos, :] = src[:, pos, :].to(h.dtype).to(h.device); return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--init_ids", default="0-2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_goal")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = (out / "rows.jsonl").open("a")
    lo, hi = a.init_ids.split("-"); inits = list(range(int(lo), int(hi) + 1))
    pairs = json.loads(a.pairs)
    H = Pi05Harness(revision="8e174154ef5f6c60a8da12ae99c303d8963138c1", dtype="float32")
    suite = _get_suite(a.suite)
    prompts = [suite.tasks[i].language for i in range(len(suite.tasks))]

    for tA, tB, _tC in pairs:
        cell = f"{a.suite}_t{tA}_vs_t{tB}"
        env = LiberoEnv(task_suite=suite, task_id=tA, task_suite_name=a.suite, obs_type="pixels_agent_pos",
                        observation_height=360, observation_width=360, init_states=True, episode_index=0, n_envs=1)
        rs = env._env.env
        objA, nameA = GOAL_OBJ[tA]; objB, nameB = GOAL_OBJ[tB]
        print(f"\n[cell] {cell}  A={objA} B={objB}", flush=True)
        for init in inits:
            env.init_state_id = init
            obs, _ = env.reset(seed=1000 + init)
            pA = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objA]])
            pB = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objB]])
            geom2obj = {}
            for nm, bid in rs.obj_body_id.items():
                for gid in range(rs.sim.model.ngeom):
                    if rs.sim.model.geom_bodyid[gid] == bid: geom2obj[gid] = nm
            segimg = rs.sim.render(width=HW, height=HW, camera_name="agentview", segmentation=True)
            bA = H.build_batch(obs, prompts[tA]); bB = H.build_batch(obs, prompts[tB])
            prA = H.prefix_forward(bA, capture=True); prB = H.prefix_forward(bB, capture=True)
            nv = prA["n_img_valid"]; valid = list(range(nv))
            locB = [p for p in patches_of(segimg, geom2obj, {objB}) if p < nv]
            rA, rB = prA["resid"], prB["resid"]
            noise = H.make_noise(seed=init * 100)
            chA = H.unnormalize(H.action_forward(prA, noise)[0])[0].cpu().numpy()
            chB = H.unnormalize(H.action_forward(prB, noise)[0])[0].cpu().numpy()
            mA, mB = axis_metric(chA, pA, pB, 10), axis_metric(chB, pA, pB, 10)
            den = (mB - mA) if abs(mB - mA) > 1e-6 else float("nan")
            rng = np.random.default_rng(1234 + init)
            for n in DOSES:
                if n > nv: continue
                for arm in ("LOC", "RAND"):
                    pos = grow(locB, n, valid) if arm == "LOC" else sorted(rng.choice(nv, n, replace=False).tolist())
                    pr = H.prefix_forward(bA, resid_hooks={L: (lambda h, L=L, P=pos: put(h, P, rB[L])) for L in CARRIER})
                    c, _ = H.action_forward(pr, noise)
                    ec = H.unnormalize(c)[0].cpu().numpy()
                    f.write(json.dumps({
                        "cell": cell, "tA": tA, "tB": tB, "init": init, "arm": arm, "n_pos": n,
                        "n_loc_avail": len(locB), "n_img_valid": nv,
                        "R": (axis_metric(ec, pA, pB, 10) - mA) / den,
                        "l2_to_A": float(np.linalg.norm(ec[:10] - chA[:10])),
                        "l2_to_B": float(np.linalg.norm(ec[:10] - chB[:10])),
                    }) + "\n")
                f.flush()
            print(f"  init {init} locB={len(locB)} done", flush=True)
        env.close()
    print("\nDOSE DONE ->", out, flush=True)


if __name__ == "__main__":
    main()
