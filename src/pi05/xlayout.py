#!/usr/bin/env python
"""DECISIVE COPY-vs-CONCEPT TEST — cross-layout carrier source, 3-way forced choice.

Why this and not the run1 version. Grant et al. (arXiv 2603.19233) Table 4 report that for pi0.5,
**99.6%** of activation-injection pairs yield a trajectory closer to the SOURCE than the destination:
whole-activation patching in pi0.5 transplants coordinate-bound motor programs. Our own layout
counterfactual says the same thing behaviourally ("touched whatever occupies A's canonical slot" is
invariant at 75% across both instructions). And run1's cross-layout arm was underpowered by
construction: the donor init put B's object only 0.27 of the A-B chunk distance away, so "copy" and
"concept" made nearly identical predictions.

libero_object fixes that. Its six CANONICAL TABLE SLOTS are identical across all ten tasks and are
0.11-0.38 m apart, so moving B's object to a different slot makes the two hypotheses predict
geometrically distinct endpoints.

  TARGET run : canonical layout L_t, prompt A
  SOURCE run : L_t with B's object relocated to a different canonical slot, prompt B
  patch      : image-position residuals, VLM layers 12-17, SOURCE -> TARGET

3-way forced choice, preregistered:
  lands on B at L_t's position   -> CONCEPT transfer (the patch carried "go to B", recombined here)
  lands on B at L_s's position   -> COPY (the patch carried the source's motor program)
  neither                        -> NULL

We report both the chunk-space discriminator (L2 to each run's own clean chunk) and the Grant
displacement statistic cos(traj,src) vs cos(traj,dst), so the number is directly comparable to their
99.6% pi0.5 baseline.

Arms: WHOLE (all image positions; the positive control FOR copying), LOC (segmentation patches of B
+dilation), MULTI (carrier averaged over several source layouts — causal-scrubbing style resample:
averaging destroys any single source's motor program while preserving what the hypothesis says is
shared), RAND (equal-norm random, matched count).
"""
import argparse, json, math, os, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from robosuite.utils import binding_utils
from build_stage1_twins import _patched_read_pixels
binding_utils.MjRenderContext.read_pixels = _patched_read_pixels
from hooks import Pi05Harness, axis_metric
from lerobot.envs.libero import _get_suite, get_task_init_states
from run_libero_prompt_conditions import ContactTracker, make_env

CARRIER = list(range(12, 18))
HW, G = 360, 16
SLOTS = {"S1": (-0.12, -0.24), "S2": (0.05, -0.10), "S3": (-0.15, 0.06),
         "S4": (0.10, -0.20), "S5": (0.15, 0.03), "S6": (-0.20, -0.08)}
slot_of = lambda p: min(SLOTS, key=lambda s: math.dist(SLOTS[s], p[:2]))
OBJ_PROMPT = {"alphabet_soup_1": "alphabet soup", "cream_cheese_1": "cream cheese",
              "salad_dressing_1": "salad dressing", "bbq_sauce_1": "bbq sauce", "ketchup_1": "ketchup",
              "tomato_sauce_1": "tomato sauce", "butter_1": "butter", "milk_1": "milk",
              "chocolate_pudding_1": "chocolate pudding", "orange_juice_1": "orange juice"}


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


def put(h, pos, src):
    h = h.clone(); h[:, pos, :] = src[:, pos, :].to(h.dtype).to(h.device); return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="0,1,2,4,5,6,7,9")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_object")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = (out / "rows.jsonl").open("a")
    H = Pi05Harness(revision="8e174154ef5f6c60a8da12ae99c303d8963138c1", dtype="float32")
    suite = _get_suite(a.suite)
    man = json.load(open("artifacts/vla_arbitration/<run-id>/arbitration_manifest.json"))[a.suite]

    for T in [int(x) for x in a.tasks.split(",")]:
        env = make_env(suite, a.suite, T); env.init_state_id = 0; env.reset(seed=1000)
        rs = env._env; inner = getattr(rs, "env", rs); tr = ContactTracker(rs)
        SIM = lambda: inner.sim          # re-deref every time: reset() swaps the MjSim out
        BID = tr.obj_body_id
        ad = {}
        for n in tr.objects:
            v = SIM().model.get_joint_qpos_addr(f"{n}_joint0")
            ad[n] = int(v[0] if isinstance(v, tuple) else v)
        base = np.asarray(get_task_init_states(suite, T))[0].copy()
        movable = [o for o in tr.objects if "basket" not in o]
        xy = {o: (float(base[1 + ad[o]]), float(base[1 + ad[o] + 1])) for o in movable}
        objA = man["tasks"][str(T)]["target_object"]; pA_text = man["tasks"][str(T)]["prompt"]
        cands = [o for o in movable if o != objA and o in OBJ_PROMPT]
        if not cands: env.close(); continue
        objB = cands[0]
        pB_text = pA_text.replace(OBJ_PROMPT[objA], OBJ_PROMPT[objB])
        # a relocation slot for B: the movable object whose slot is FARTHEST from B's own
        others = [o for o in movable if o not in (objA, objB)]
        if not others: env.close(); continue
        objSwap = max(others, key=lambda o: math.dist(xy[o], xy[objB]))
        sep = math.dist(xy[objB], xy[objSwap])
        geom2obj = tr.geom2obj
        print(f"\n[t{T}] A={objA} B={objB} relocate B<->{objSwap} sep={sep:.3f}m", flush=True)
        print(f"   A: '{pA_text}'\n   B: '{pB_text}'", flush=True)

        def obs_for(pairs):
            st = base.copy()
            for o1, o2 in pairs:
                st[1 + ad[o1]], st[1 + ad[o1] + 1] = xy[o2]
                st[1 + ad[o2]], st[1 + ad[o2] + 1] = xy[o1]
            env._init_states = st[None, :]; env.init_state_id = 0
            o, _ = env.reset(seed=1000)
            return o

        # TARGET layout (canonical) and SOURCE layout (B moved to objSwap's slot)
        obsT = obs_for([])
        pB_target = np.array(SIM().data.body_xpos[BID[objB]])
        pA_target = np.array(SIM().data.body_xpos[BID[objA]])
        segimg = SIM().render(width=HW, height=HW, camera_name="agentview", segmentation=True)
        locB = patches_of(segimg, geom2obj, {objB})
        bT_A = H.build_batch(obsT, pA_text); bT_B = H.build_batch(obsT, pB_text)
        prT_A = H.prefix_forward(bT_A, capture=True); prT_B = H.prefix_forward(bT_B, capture=True)
        nv = prT_A["n_img_valid"]; locB = [p for p in locB if p < nv]
        noise = H.make_noise(seed=T * 100)
        chT_A = H.unnormalize(H.action_forward(prT_A, noise)[0])[0].cpu().numpy()
        chT_B = H.unnormalize(H.action_forward(prT_B, noise)[0])[0].cpu().numpy()

        obsS = obs_for([(objB, objSwap)])
        pB_source = np.array(SIM().data.body_xpos[BID[objB]])
        bS_B = H.build_batch(obsS, pB_text)
        prS_B = H.prefix_forward(bS_B, capture=True)
        chS_B = H.unnormalize(H.action_forward(prS_B, noise)[0])[0].cpu().numpy()
        rS = prS_B["resid"]

        # MULTI: average the carrier over several distinct source layouts
        multi = {L: [] for L in CARRIER}
        for o in others[:4]:
            _ = obs_for([(objB, o)])
            pr = H.prefix_forward(H.build_batch(_, pB_text), capture=True)
            for L in CARRIER: multi[L].append(pr["resid"][L])
        multi = {L: torch.stack(v).mean(0) for L, v in multi.items()}
        obsT = obs_for([])  # restore

        g = torch.Generator(device="cpu").manual_seed(31337 + T)
        arms = {
            "WHOLE": (list(range(nv)), rS),
            "LOC":   (locB, rS),
            "MULTI": (list(range(nv)), multi),
            "RAND":  (list(range(nv)), None),
        }
        for arm, (pos, src) in arms.items():
            if not pos: continue
            if src is None:
                src = {L: (lambda s: s / s.norm(dim=-1, keepdim=True) * rS[L].norm(dim=-1, keepdim=True))(
                    torch.randn(rS[L].shape, generator=g)) for L in CARRIER}
            pr = H.prefix_forward(bT_A, resid_hooks={L: (lambda h, L=L, P=pos, S=src: put(h, P, S[L])) for L in CARRIER})
            ec = H.unnormalize(H.action_forward(pr, noise)[0])[0].cpu().numpy()
            d_target = float(np.linalg.norm(ec[:10] - chT_B[:10]))
            d_source = float(np.linalg.norm(ec[:10] - chS_B[:10]))
            d_cleanA = float(np.linalg.norm(ec[:10] - chT_A[:10]))
            # Grant displacement statistic on the realised first-10 displacement
            disp = ec[:10, :3].sum(0)
            u_t = pB_target - pA_target; u_s = pB_source - pA_target
            cos = lambda u: float(disp @ u / ((np.linalg.norm(disp) + 1e-9) * (np.linalg.norm(u) + 1e-9)))
            verdict = ("CONCEPT" if d_target < d_source and d_target < d_cleanA else
                       "COPY" if d_source < d_target and d_source < d_cleanA else "NULL")
            f.write(json.dumps({
                "task": T, "arm": arm, "objA": objA, "objB": objB, "objSwap": objSwap,
                "slot_sep_m": sep, "n_pos": len(pos), "n_img_valid": nv, "n_loc": len(locB),
                "d_to_B_target": d_target, "d_to_B_source": d_source, "d_to_clean_A": d_cleanA,
                "d_targetB_sourceB": float(np.linalg.norm(chT_B[:10] - chS_B[:10])),
                "d_cleanA_targetB": float(np.linalg.norm(chT_A[:10] - chT_B[:10])),
                "cos_to_target": cos(u_t), "cos_to_source": cos(u_s), "verdict": verdict,
                "pB_target": pB_target.tolist(), "pB_source": pB_source.tolist(),
            }) + "\n")
            print(f"   {arm:6s} n={len(pos):3d}  d->B@target={d_target:.3f} d->B@source={d_source:.3f} "
                  f"d->cleanA={d_cleanA:.3f}  cos_t={cos(u_t):+.2f} cos_s={cos(u_s):+.2f}  {verdict}", flush=True)
        f.flush()
        env.close()
    print("\nXLAYOUT DONE ->", out, flush=True)


if __name__ == "__main__":
    main()
