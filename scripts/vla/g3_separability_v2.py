#!/usr/bin/env python
"""G3 - separability of the shortcut direction from perception.  No training, no rollouts.

Answers three questions from prefix forwards only:

  Q1  Is d_prior a real, reproducible direction?    -> donor agreement cos(d_1, d_2), disjoint donor halves.
  Q2  Is it separable from perception?              -> cos(d_prior, d_obj_j) per contrast, AND the curve
                                                       ||P_k P_k^T d_prior|| as the perception basis grows.
                                                       A single cosine against a rank-2 basis is too permissive.
  Q3  Is there ONE route or several?                -> cos(d_OBEY, d_JAM); near-orthogonal => rank-1 ablation
                                                       cannot remove the behaviour (Bricken redundancy point).

Estimator notes
  * d_prior is estimated WITHIN SCENE: for each donor scene s, d_s = mean(IGNORE cells) - mean(OBEY cells),
    then d_prior = mean_s d_s.  Scene appearance cancels exactly.  Every donor scene has cells on both sides.
  * "residual entering layer L" == output of layer L-1, so we capture resid[L-1] for L in 12..17.
  * Perception vectors hold the PROMPT FIXED and vary only the image, so they are pure perception:
      swap twins (2 objects exchange position), determinate twins (5 distractors replaced),
      layout (init-state variation within task), inventory (across-task variation).
  * Instruction-string confound is measured, not assumed: d_instr separates instruction strings ignoring mode.
"""
import os, sys, json, argparse, itertools, collections
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/root/vla/scripts/vla")
import numpy as np, torch
torch.set_num_threads(4)
from lerobot.envs.libero import _get_suite
from lerobot.envs.utils import preprocess_observation
from hooks import Pi05Harness, batchify
from run_libero_prompt_conditions import make_env

REV = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
LAYERS = [12, 13, 14, 15, 16, 17]          # residual ENTERING these == captured resid[L-1]
DONORS = [0, 4, 6, 7, 9]          # steering donors; kept for reference only
SCENES = list(range(10))          # separability uses ALL scenes: the unit of analysis is the scene
SUITE = "libero_object"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def label_cells(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["task_id"], r["wrong_task_id"])].append(r)
    out = {}
    for k, rs in cells.items():
        n = len(rs)
        tgt = sum(1 for r in rs if (r["first_touch"] or {}).get("object") == r["target_object"])
        wrg = sum(1 for r in rs if (r["first_touch"] or {}).get("object") == r["wrong_object"])
        oth = n - tgt - wrg
        m = max(tgt, wrg, oth)
        out[k] = ("IGNORE" if m == tgt else ("OBEY" if m == wrg else "JAM"), n, tgt, wrg, oth)
    return out


class Extractor:
    """mean residual over the valid image positions, per layer, for one (env-observation, prompt)."""

    def __init__(self, h):
        self.h = h

    def batch(self, obs, prompt):
        o = preprocess_observation(batchify(obs))
        o["task"] = [prompt]
        return self.h.pre(self.h.epre(o))

    def feats(self, obs, prompt):
        out = self.h.prefix_forward(self.batch(obs, prompt), capture=True)
        nv = out["n_img_valid"]
        f = {}
        for L in LAYERS:
            r = out["resid"][L - 1][0]                 # [prefix_len, d]
            f[L] = r[:nv].mean(0).numpy().astype(np.float64)
        del out
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arb", default="/root/vla/runs/arbitration/libero_object.jsonl")
    ap.add_argument("--stage1", default="/root/vla/stage1")
    ap.add_argument("--inits", default=",".join(str(i) for i in range(20)))
    ap.add_argument("--layout_inits", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out", default="/root/vla/artifacts/g3_separability.json")
    ap.add_argument("--dtype", default="bfloat16", help="bfloat16 keeps the harness ~7GB so it co-locates; "
                    "G3 only needs mean residuals and cosines, not patching precision")
    a = ap.parse_args()
    inits = [int(x) for x in a.inits.split(",")]
    layout_inits = [int(x) for x in a.layout_inits.split(",")]

    lab = label_cells(a.arb)
    donor_cells = {k: v for k, v in lab.items() if k[0] in SCENES}
    print("[g3] donor cells:", {f"t{k[0]}<-w{k[1]}": v[0] for k, v in sorted(donor_cells.items())}, flush=True)

    h = Pi05Harness(revision=REV, dtype=a.dtype)
    print(f"[g3] harness dtype={a.dtype}", flush=True)
    ex = Extractor(h)
    suite = _get_suite(SUITE)

    # ---------------------------------------------------------------- cell activations (d_prior, d_OBEY, d_JAM)
    cellfeat = {}       # (task, wrong) -> {L: mean over inits}
    envs = {}
    cache_path = a.out.replace(".json", "_cellfeat.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cellfeat = z["cf"].item()
        print(f"[g3] loaded cached cell activations from {cache_path} ({len(cellfeat)} cells)", flush=True)
    for (t, w), (mode, *_rest) in sorted(donor_cells.items()):
        if (t, w) in cellfeat:
            continue
        if t not in envs:
            envs[t] = make_env(suite, SUITE, t)
        env = envs[t]
        prompt = suite.tasks[w].language
        acc = collections.defaultdict(list)
        for i in inits:
            env.init_state_id = i
            obs, _ = env.reset(seed=1000 + i)
            f = ex.feats(obs, prompt)
            for L in LAYERS:
                acc[L].append(f[L])
        cellfeat[(t, w)] = {L: np.mean(acc[L], 0) for L in LAYERS}
        print(f"[g3] cell t{t}<-w{w} {mode} done", flush=True)
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, cf=np.array(cellfeat, dtype=object))
        print(f"[g3] cached cell activations -> {cache_path}", flush=True)

    def within_scene_dir(pos_label, neg_label, cells):
        """mean over scenes of (mean(pos cells in scene) - mean(neg cells in scene)); scene cancels exactly."""
        per_L = {}
        used = []
        for L in LAYERS:
            ds = []
            for t in SCENES:
                P = [cellfeat[k][L] for k in cells if k[0] == t and cells[k][0] == pos_label]
                N = [cellfeat[k][L] for k in cells if k[0] == t and cells[k][0] == neg_label]
                if not P or not N:
                    continue
                ds.append(np.mean(P, 0) - np.mean(N, 0))
                if L == LAYERS[0]:
                    used.append(t)
            per_L[L] = unit(np.mean(ds, 0)) if ds else None
        return per_L, used

    d_prior, scenes_used = within_scene_dir("IGNORE", "OBEY", donor_cells)
    print(f"[g3] d_prior estimated within-scene over scenes {scenes_used}", flush=True)

    # Reproducibility as a DISTRIBUTION, not one 3-vs-2 split: estimate an independent direction per
    # scene, then report every pairwise cosine (n_scenes choose 2) and leave-one-scene-out agreement.
    def per_scene_dirs():
        out = {}
        for t in SCENES:
            per_L = {}
            for L in LAYERS:
                P = [cellfeat[k][L] for k in donor_cells if k[0] == t and donor_cells[k][0] == "IGNORE"]
                N = [cellfeat[k][L] for k in donor_cells if k[0] == t and donor_cells[k][0] == "OBEY"]
                per_L[L] = unit(np.mean(P, 0) - np.mean(N, 0)) if (P and N) else None
            if all(per_L[L] is not None for L in LAYERS):
                out[t] = per_L
        return out

    dscene = per_scene_dirs()
    print(f"[g3] per-scene directions available for scenes {sorted(dscene)} "
          f"(n={len(dscene)}, {len(dscene)*(len(dscene)-1)//2} pairwise cosines)", flush=True)

    # Q3: route multiplicity - IGNORE-vs-OBEY  versus  JAM-vs-OBEY
    d_jam, jam_scenes = within_scene_dir("JAM", "OBEY", donor_cells)

    # ---------------------------------------------------------------- perception basis (prompt held FIXED)
    perception = []   # (name, {L: vec})
    man = json.load(open(os.path.join(a.stage1, "manifest.json")))["pairs"]
    for pair, meta in man.items():
        t = meta["task_id"]
        promptA = meta["prompts"]["A"]
        base = os.path.join(a.stage1, pair)
        try:
            ref = None
            for variant in ("ambiguous", "swap", "determinate"):
                bddl = os.path.join(base, f"{variant}.bddl")
                init = os.path.join(base, f"{variant}_init.pt")
                if not (os.path.exists(bddl) and os.path.exists(init)):
                    continue
                env = make_env(suite, SUITE, t, bddl_file=bddl,
                               init_states=torch.load(init, weights_only=False))
                acc = collections.defaultdict(list)
                for i in range(5):
                    env.init_state_id = i
                    obs, _ = env.reset(seed=2000 + i)
                    f = ex.feats(obs, promptA)          # PROMPT FIXED -> pure perception
                    for L in LAYERS:
                        acc[L].append(f[L])
                env.close()
                v = {L: np.mean(acc[L], 0) for L in LAYERS}
                if variant == "ambiguous":
                    ref = v
                elif ref is not None:
                    perception.append((f"{pair}:{variant}-ambiguous",
                                       {L: unit(v[L] - ref[L]) for L in LAYERS}))
                    print(f"[g3] perception vec {pair}:{variant}", flush=True)
        except Exception as e:
            print(f"[g3] twin {pair} FAILED {type(e).__name__}: {e}", flush=True)

    # layout: init-state variation within a donor task, prompt fixed  -> top PCs
    for t in SCENES[:3]:
        env = envs.get(t) or make_env(suite, SUITE, t)
        envs[t] = env
        prompt = suite.tasks[t].language
        acc = collections.defaultdict(list)
        for i in layout_inits:
            env.init_state_id = i
            obs, _ = env.reset(seed=3000 + i)
            f = ex.feats(obs, prompt)
            for L in LAYERS:
                acc[L].append(f[L])
        vecs = {}
        for L in LAYERS:
            X = np.stack(acc[L]); X = X - X.mean(0)
            u, s, vt = np.linalg.svd(X, full_matrices=False)
            vecs[L] = unit(vt[0])
        perception.append((f"layout_t{t}_pc1", vecs))
        print(f"[g3] perception vec layout t{t}", flush=True)

    # inventory: across-task mean variation, correct prompt each -> top PCs
    accT = collections.defaultdict(list)
    for t in range(10):
        env = envs.get(t) or make_env(suite, SUITE, t)
        envs[t] = env
        env.init_state_id = 0
        obs, _ = env.reset(seed=4000)
        f = ex.feats(obs, suite.tasks[t].language)
        for L in LAYERS:
            accT[L].append(f[L])
    for pc in range(3):
        vecs = {}
        for L in LAYERS:
            X = np.stack(accT[L]); X = X - X.mean(0)
            u, s, vt = np.linalg.svd(X, full_matrices=False)
            vecs[L] = unit(vt[pc])
        perception.append((f"inventory_pc{pc+1}", vecs))
    print(f"[g3] perception basis size = {len(perception)}", flush=True)

    # ---------------------------------------------------------------- report
    res = {"layers": LAYERS, "scenes_used": scenes_used, "per_scene": sorted(dscene),
           "donor_cells": {f"t{k[0]}<-w{k[1]}": v[0] for k, v in sorted(donor_cells.items())},
           "perception_names": [n for n, _ in perception],
           "donor_agreement": {}, "cos_perception": {}, "subspace_curve": {},
           "route_multiplicity": {}, "jam_scenes": jam_scenes}

    res["n_scenes"] = len(dscene)
    res["pairwise"] = {}
    res["loso"] = {}
    for L in LAYERS:
        if d_prior[L] is None:
            continue
        ts = sorted(dscene)
        pw = [cos(dscene[a][L], dscene[b][L]) for a, b in itertools.combinations(ts, 2)]
        res["pairwise"][L] = {"cosines": pw, "mean": float(np.mean(pw)) if pw else None,
                              "median": float(np.median(pw)) if pw else None,
                              "frac_ge_0.4": float(np.mean([c >= 0.4 for c in pw])) if pw else None,
                              "boot95": (lambda B: [float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))])(
                                  [np.mean(np.random.choice(pw, len(pw), replace=True)) for _ in range(2000)])
                              if len(pw) > 1 else None}
        loso = []
        for t in ts:
            rest = [dscene[u][L] for u in ts if u != t]
            loso.append(cos(np.mean(rest, 0), dscene[t][L]))
        res["loso"][L] = {"cosines": loso, "mean": float(np.mean(loso)),
                          "frac_ge_0.4": float(np.mean([c >= 0.4 for c in loso]))}
        res["donor_agreement"][L] = res["pairwise"][L]["mean"]
        res["cos_perception"][L] = {n: cos(d_prior[L], v[L]) for n, v in perception}
        # curve: fraction of ||d_prior|| inside span of first k perception vectors (Gram-Schmidt order)
        B = []
        curve = []
        for n, v in perception:
            w = v[L].copy()
            for b in B:
                w = w - np.dot(w, b) * b
            nw = np.linalg.norm(w)
            if nw > 1e-8:
                B.append(w / nw)
            P = np.stack(B) if B else np.zeros((0, len(d_prior[L])))
            frac = float(np.linalg.norm(P @ d_prior[L])) if len(B) else 0.0
            curve.append({"k": len(B), "added": n, "frac_norm_in_span": frac})
        res["subspace_curve"][L] = curve
        if d_jam[L] is not None:
            res["route_multiplicity"][L] = cos(d_prior[L], d_jam[L])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print("\n[g3] ================= RESULTS =================", flush=True)
    print(f"[g3] Q1 reproducibility across scenes (n_scenes={res.get('n_scenes')}); gate >=0.4 at >=4/6 layers")
    ok = 0
    for L in LAYERS:
        pwd = res["pairwise"].get(L)
        lo = res["loso"].get(L)
        if not pwd: continue
        ok += (pwd["mean"] or 0) >= 0.4
        b = pwd["boot95"]
        bs = f"[{b[0]:+.3f},{b[1]:+.3f}]" if b else "n/a"
        print(f"       L{L}: pairwise mean {pwd['mean']:+.3f} boot95 {bs} "
              f"frac>=0.4 {pwd['frac_ge_0.4']:.2f} | LOSO mean {lo['mean']:+.3f} frac>=0.4 {lo['frac_ge_0.4']:.2f}")
    print(f"     -> {ok}/6 layers pass  => {'PASS' if ok >= 4 else 'FAIL - not a reproducible concept'}")
    print(f"\n[g3] Q2 separability from perception  (KILL: |cos| > 0.5)")
    worst = 0.0
    for L in LAYERS:
        cp = res["cos_perception"].get(L, {})
        if not cp: continue
        mx = max(cp.items(), key=lambda kv: abs(kv[1]))
        worst = max(worst, abs(mx[1]))
        tot = res["subspace_curve"][L][-1]["frac_norm_in_span"] if res["subspace_curve"].get(L) else float("nan")
        print(f"       L{L}: max|cos| {mx[1]:+.3f} ({mx[0]})   ||proj onto FULL perception span|| = {tot:.3f}")
    print(f"     -> worst single-contrast |cos| = {worst:.3f} => {'CAFT DEAD' if worst > 0.5 else 'separable so far'}")
    print(f"\n[g3] Q3 route multiplicity cos(d_IGNOREvOBEY, d_JAMvOBEY)")
    for L in LAYERS:
        c = res["route_multiplicity"].get(L)
        if c is not None:
            print(f"       L{L}: {c:+.3f}")
    print(f"     (near 0 => JAM is a SEPARATE route; a rank-1 ablation cannot remove both)")
    print(f"\n[g3] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
