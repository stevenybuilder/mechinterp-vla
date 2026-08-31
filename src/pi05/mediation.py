#!/usr/bin/env python
"""CAUSAL HANDOFF MEDIATION (source-specific mediator test) -- preregistered in docs/protocol.md
   sha256 f4877800ffe48882b175d4df70dcfdae6549a1a9bfd5ecd896e91aa0a8bafcdc (written before this ran).

Claim: language only affects behaviour after being compiled into an action-facing CARRIER
(pi0.5: image-token positions, VLM layers 12-17).  Patching the carrier moves behaviour; RESTORING the
carrier to clean-A blocks the instruction's effect even though the instruction itself was swapped.

TRAJECTORY-COPYING CONFOUND (the reason C3X and C3L exist).
Transplanting activations from observation X into observation Y may simply transplant "the trajectory
appropriate to X" rather than a concept.  Our own held-out stage-2 records make this live, not
hypothetical: RS[IMG]@all -- image KV from a length-matched UNRELATED prompt -- gives R = 0.846,
nearly the 0.979 of the true B patch.  Only L2-to-target separates them (0.614 vs 0.085).
Two discriminators are therefore first-class conditions here, not extras:
  C3L  LOCALIZED carrier: only the ~2-9 image patches covering B's object (+1 dilation), from the
       MuJoCo segmentation mask.  Whole-image replacement is the arm most suspect of copying.
  C3X  CROSS-LAYOUT donor: B's carrier taken from a DIFFERENT init state, where B's object sits
       somewhere else.  If the action goes to B's object AT THE CURRENT LAYOUT'S POSITION it is
       concept transfer; if it goes to the DONOR layout's position it is trajectory copying.
       We report L2 to BOTH the current-layout clean-B chunk and the donor-layout clean-B chunk.

All edits are residual-stream edits applied to the OUTPUT of layer L via prefix_forward(resid_hooks=),
so they propagate to L+1.  C2 (early source) and C4 (reset) are the same mechanism, which is what makes
the reset a fair test rather than a different intervention.
"""
import argparse, json, os, math, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from robosuite.utils import binding_utils
from build_stage1_twins import _patched_read_pixels
binding_utils.MjRenderContext.read_pixels = _patched_read_pixels  # numpy-2 overflow in robosuite seg render
from hooks import Pi05Harness, axis_metric
from lerobot.envs.libero import LiberoEnv, _get_suite
from stage2_discovery import GOAL_OBJ

EARLY = list(range(0, 6))
CARRIER = list(range(12, 18))
HW, G = 360, 16


def patches_of(seg, geom2obj, names, dilate=1):
    """image-token indices (within a 16x16 view) whose patch contains any of `names`"""
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


def put(h, pos, src):
    h = h.clone()
    h[:, pos, :] = src[:, pos, :].to(h.dtype).to(h.device)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--init_ids", default="0-4")
    ap.add_argument("--donor_init", type=int, default=17, help="init state supplying the cross-layout carrier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    ap.add_argument("--revision", default="8e174154ef5f6c60a8da12ae99c303d8963138c1")
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows_f = (out / "rows.jsonl").open("a")
    lo, hi = a.init_ids.split("-"); inits = list(range(int(lo), int(hi) + 1))
    pairs = json.loads(a.pairs)

    H = Pi05Harness(a.policy, revision=a.revision, dtype=a.dtype)
    suite = _get_suite(a.suite)
    prompts = [suite.tasks[i].language for i in range(len(suite.tasks))]
    man = {"args": vars(a), "prereg_sha256": "f4877800ffe48882b175d4df70dcfdae6549a1a9bfd5ecd896e91aa0a8bafcdc",
           "early_layers": EARLY, "carrier_layers": CARRIER, "pairs": [], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    for tA, tB, tC in pairs:
        cell = f"{a.suite}_t{tA}_vs_t{tB}"
        env = LiberoEnv(task_suite=suite, task_id=tA, task_suite_name=a.suite, obs_type="pixels_agent_pos",
                        observation_height=360, observation_width=360, init_states=True, episode_index=0, n_envs=1)
        rs = env._env.env
        objA, nameA = GOAL_OBJ[tA]; objB, nameB = GOAL_OBJ[tB]
        pA_text, pB_text, pC_text = prompts[tA], prompts[tB], prompts[tC]
        man["pairs"].append({"cell": cell, "A": [tA, objA, pA_text], "B": [tB, objB, pB_text], "C": [tC, pC_text]})
        print(f"\n[cell] {cell}\n   A='{pA_text}' ({objA})\n   B='{pB_text}' ({objB})\n   C='{pC_text}'", flush=True)

        # --- donor layout (different init): B's carrier and B's position there
        env.init_state_id = a.donor_init
        obs_d, _ = env.reset(seed=1000 + a.donor_init)
        pB_donor = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objB]])
        pA_donor = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objA]])
        bB_d = H.build_batch(obs_d, pB_text)
        prB_d = H.prefix_forward(bB_d, capture=True)
        rB_d = prB_d["resid"]
        chunk_B_donor = H.unnormalize(H.action_forward(prB_d, H.make_noise(seed=a.donor_init * 100))[0])[0].cpu().numpy()

        for init in inits:
            if init == a.donor_init: continue
            env.init_state_id = init
            obs, _ = env.reset(seed=1000 + init)
            pA = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objA]])
            pB = np.array(rs.sim.data.body_xpos[rs.obj_body_id[objB]])
            # segmentation -> the image patches covering B's object in THIS layout
            geom2obj = {}
            for nm, bid in rs.obj_body_id.items():
                for gid in range(rs.sim.model.ngeom):
                    if rs.sim.model.geom_bodyid[gid] == bid: geom2obj[gid] = nm
            segimg = rs.sim.render(width=HW, height=HW, camera_name="agentview", segmentation=True)
            locB = patches_of(segimg, geom2obj, {objB})

            bA, bB, bC = (H.build_batch(obs, t) for t in (pA_text, pB_text, pC_text))
            prA = H.prefix_forward(bA, capture=True)
            prB = H.prefix_forward(bB, capture=True)
            prC = H.prefix_forward(bC, capture=True)
            n_img = prA["n_img_slots"]
            segA, _ = H.segments(bA, object_name=nameA, n_img=n_img)
            segB, _ = H.segments(bB, object_name=nameB, n_img=n_img)
            IMG = list(range(prA["n_img_valid"]))
            iA, iB = segA["INSTR"], segB["INSTR"]
            if len(iA) != len(iB):
                print(f"  [skip] init {init}: INSTR tokens differ A={len(iA)} B={len(iB)}", flush=True); continue
            locB = [p for p in locB if p < prA["n_img_valid"]]
            # Some B-objects (e.g. wooden_cabinet_1) yield 0 segmentation patches. Previously this
            # skipped the WHOLE init, which systematically dropped every cabinet cell and cut n from
            # 12 to 8. Only C3L is undefined there -- run every other condition, mark C3L N/A.
            has_loc = len(locB) >= 2
            if not has_loc:
                print(f"  [note] init {init}: {objB} has {len(locB)} localized patches -> C3L N/A, other conditions still run", flush=True)
            rA, rB, rC = prA["resid"], prB["resid"], prC["resid"]
            g = torch.Generator(device="cpu").manual_seed(7000 + init)

            def rand_like(L):
                s = rB[L]; r = torch.randn(s.shape, generator=g)
                return r / r.norm(dim=-1, keepdim=True) * s.norm(dim=-1, keepdim=True)

            conds = {
                "C0_clean_A":       (bA, None),
                "C1_clean_B":       (bB, None),
                "C2_early_src":     (bA, {L: (lambda h, L=L: put(h, iA, rB[L])) for L in EARLY}),
                "C3_carrier":       (bA, {L: (lambda h, L=L: put(h, IMG, rB[L])) for L in CARRIER}),
                **({"C3L_carrier_local": (bA, {L: (lambda h, L=L: put(h, locB, rB[L])) for L in CARRIER})} if has_loc else {}),
                "C3X_carrier_xlay": (bA, {L: (lambda h, L=L: put(h, IMG, rB_d[L])) for L in CARRIER}),
                "C4_reset":         (bA, {**{L: (lambda h, L=L: put(h, iA, rB[L])) for L in EARLY},
                                          **{L: (lambda h, L=L: put(h, IMG, rA[L])) for L in CARRIER}}),
                "C5_deadsite":      (bA, {L: (lambda h, L=L: put(h, iA, rB[L])) for L in CARRIER}),
                "C6_rand_carrier":  (bA, {L: (lambda h, L=L: put(h, IMG, rand_like(L))) for L in CARRIER}),
                "C7_unrel_carrier": (bA, {L: (lambda h, L=L: put(h, IMG, rC[L])) for L in CARRIER}),
            }
            noise = H.make_noise(seed=init * 100)
            ch = {}
            for nm, (batch, hooks) in conds.items():
                pr = H.prefix_forward(batch, resid_hooks=hooks) if hooks else (prA if nm == "C0_clean_A" else prB)
                c, _ = H.action_forward(pr, noise)
                ch[nm] = H.unnormalize(c)[0].cpu().numpy()

            mA = axis_metric(ch["C0_clean_A"], pA, pB, 10)
            mB = axis_metric(ch["C1_clean_B"], pA, pB, 10)
            den = (mB - mA) if abs(mB - mA) > 1e-6 else float("nan")
            for nm, ec in ch.items():
                m = axis_metric(ec, pA, pB, 10)
                rows_f.write(json.dumps({
                    "cell": cell, "tA": tA, "tB": tB, "tC": tC, "init": init, "donor_init": a.donor_init,
                    "condition": nm, "m10": m, "m_A": mA, "m_B": mB, "R": (m - mA) / den,
                    "l2_to_A": float(np.linalg.norm(ec[:10] - ch["C0_clean_A"][:10])),
                    "l2_to_B": float(np.linalg.norm(ec[:10] - ch["C1_clean_B"][:10])),
                    "l2_to_B_donor": float(np.linalg.norm(ec[:10] - chunk_B_donor[:10])),
                    "n_instr": len(iA), "n_localized_patches": (len(locB) if has_loc else -1), "n_img_valid": prA["n_img_valid"],
                    "pA": pA.tolist(), "pB": pB.tolist(), "pB_donor": pB_donor.tolist(),
                }) + "\n")
            rows_f.flush()
            print("  init %2d  " % init + "  ".join(
                "%s R=%+.2f l2B=%.2f" % (k.split("_", 1)[1][:10], (axis_metric(v, pA, pB, 10) - mA) / den,
                                         float(np.linalg.norm(v[:10] - ch["C1_clean_B"][:10])))
                for k, v in ch.items() if k != "C0_clean_A"), flush=True)
        env.close()
    json.dump(man, open(out / "manifest.json", "w"), indent=1)
    print("\nDONE ->", out, flush=True)


if __name__ == "__main__":
    main()
