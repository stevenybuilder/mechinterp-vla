#!/usr/bin/env python
"""Obedience-decision probe + G3 noise ceiling. One extraction pass, two results.

PROBE (the safety claim, earned rather than asserted):
  Before the arm moves, can the prefix predict WHICH of {OBEY, IGNORE, JAM} the episode will be?
  Unit = (scene, wrong-instruction, init) episode.  Read from IMG positions vs INSTR positions,
  layers 12-17.  Leave-ONE-SCENE-out CV.  Beaten against five baselines, because a probe that
  cannot beat "predict this cell's modal label" has learned nothing.

NOISE CEILING (does the CAFT negative survive?):
  Split each scene's init states in half, estimate the IGNORE-vs-OBEY direction independently in
  each half, and take cos(h1, h2).  That is the BEST agreement obtainable given estimator noise.
  If the within-scene ceiling is also ~0, then "no shared concept across scenes" is unfalsifiable
  -- the estimator is noise.  If the ceiling is high and cross-scene is ~0, the negative is REAL.
  Also reports the same numbers after removing the top-1 global PC (the L17 artifact).
"""
import os, sys, json, argparse, collections, itertools
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/root/vla/scripts/vla")
import numpy as np, torch
torch.set_num_threads(4)
from lerobot.envs.libero import _get_suite
from lerobot.envs.utils import preprocess_observation
from hooks import Pi05Harness, batchify
from stage2_discovery import seg_positions
from run_libero_prompt_conditions import make_env

REV = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
LAYERS = [12, 13, 14, 15, 16, 17]
SUITE = "libero_object"
LBL = {"IGNORE": 0, "OBEY": 1, "JAM": 2}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def episode_label(r):
    ft = (r.get("first_touch") or {}).get("object")
    if ft == r.get("target_object"):
        return "IGNORE"
    if ft == r.get("wrong_object"):
        return "OBEY"
    return "JAM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arb", default="/root/vla/runs/arbitration/libero_object.jsonl")
    ap.add_argument("--out", default="/root/vla/artifacts/obedience_probe.json")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.arb) if l.strip()]
    # unit = (task, wrong_task, init); keep the first record per unit (episodes are seed-deterministic)
    units = {}
    for r in rows:
        k = (r["task_id"], r["wrong_task_id"], r["init_id"])
        units.setdefault(k, r)
    print(f"[probe] {len(rows)} episodes -> {len(units)} unique (scene, wrong, init) units", flush=True)
    cnt = collections.Counter(episode_label(r) for r in units.values())
    print(f"[probe] label counts: {dict(cnt)}", flush=True)

    h = Pi05Harness(revision=REV, dtype=a.dtype)
    suite = _get_suite(SUITE)
    envs = {}

    X_img, X_ins, y, scene, cell, layout = [], [], [], [], [], []
    for (t, w, i) in sorted(units):
        r = units[(t, w, i)]
        if t not in envs:
            envs[t] = make_env(suite, SUITE, t)
        env = envs[t]
        env.init_state_id = i
        obs, _ = env.reset(seed=r.get("seed", 1000 + i))
        o = preprocess_observation(batchify(obs)); o["task"] = [suite.tasks[w].language]
        b = h.pre(h.epre(o))
        try:
            segd, _ = h.segments(b, object_name=r.get("wrong_object"), n_img=None)
            ins_pos = seg_positions("INSTR", segd)
        except Exception as e:
            print(f"[probe] segment fail t{t} w{w} i{i}: {type(e).__name__} {e}", flush=True)
            ins_pos = None
        out = h.prefix_forward(b, capture=True)
        nv = out["n_img_valid"]
        fi, fs = [], []
        for L in LAYERS:
            R = out["resid"][L - 1][0]
            fi.append(R[:nv].mean(0).numpy().astype(np.float32))
            fs.append(R[ins_pos].mean(0).numpy().astype(np.float32) if ins_pos else np.zeros(R.shape[-1], np.float32))
        del out
        X_img.append(np.concatenate(fi)); X_ins.append(np.concatenate(fs))
        y.append(LBL[episode_label(r)]); scene.append(t); cell.append((t, w))
        op = r.get("object_pos") or {}
        first = op.get("0", op.get(0, {})) if isinstance(op, dict) else {}
        layout.append(np.concatenate([np.asarray(v, np.float32)[:3] for k, v in sorted(first.items())][:8])
                      if first else np.zeros(24, np.float32))
        if len(y) % 50 == 0:
            print(f"[probe] {len(y)}/{len(units)} extracted", flush=True)

    X_img = np.stack(X_img); X_ins = np.stack(X_ins); y = np.array(y)
    scene = np.array(scene); cells = np.array([f"{t}_{w}" for t, w in cell])
    L = max(len(v) for v in layout); layout = np.stack([np.pad(v, (0, L - len(v))) for v in layout])
    np.savez_compressed(a.out.replace(".json", "_feats.npz"),
                        X_img=X_img, X_ins=X_ins, y=y, scene=scene, cells=cells, layout=layout)
    print(f"[probe] features saved. X_img {X_img.shape} X_ins {X_ins.shape}", flush=True)

    # ---------------------------------------------------------------- probe, leave-one-scene-out
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def loso(X):
        pred = np.zeros_like(y)
        for s in np.unique(scene):
            tr, te = scene != s, scene == s
            if len(np.unique(y[tr])) < 2:
                pred[te] = collections.Counter(y[tr]).most_common(1)[0][0]; continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=0.05, multi_class="multinomial"))
            m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        return float((pred == y).mean())

    def modal_by(keys):
        pred = np.zeros_like(y)
        for s in np.unique(scene):
            tr, te = scene != s, scene == s
            gl = {}
            for k in np.unique(keys[tr]):
                gl[k] = collections.Counter(y[tr][keys[tr] == k]).most_common(1)[0][0]
            glob = collections.Counter(y[tr]).most_common(1)[0][0]
            pred[te] = [gl.get(k, glob) for k in keys[te]]
        return float((pred == y).mean())

    res = {"n_units": int(len(y)), "label_counts": {k: int(v) for k, v in cnt.items()},
           "majority_class": float(np.bincount(y).max() / len(y))}
    res["probe_IMG"] = loso(X_img)
    res["probe_INSTR"] = loso(X_ins)
    res["baseline_layout"] = loso(layout)
    res["baseline_cell_modal"] = modal_by(cells)
    res["baseline_taskid_modal"] = modal_by(scene.astype(str))
    print("\n[probe] ===== OBEDIENCE DECISION PROBE (leave-one-scene-out) =====", flush=True)
    for k in ("majority_class", "baseline_taskid_modal", "baseline_cell_modal", "baseline_layout",
              "probe_INSTR", "probe_IMG"):
        print(f"    {k:24s} {res[k]:.3f}", flush=True)
    print("    (IMG must beat cell_modal to mean anything -- cell_modal is 'just memorise the cell')", flush=True)

    # ---------------------------------------------------------------- noise ceiling
    def dirs_from(mask_units):
        """IGNORE-minus-OBEY direction per layer-block, from a subset of units, per scene."""
        out = {}
        for s in np.unique(scene):
            m = (scene == s) & mask_units
            ig, ob = m & (y == 0), m & (y == 1)
            if ig.sum() >= 2 and ob.sum() >= 2:
                out[int(s)] = unit(X_img[ig].mean(0) - X_img[ob].mean(0))
        return out

    rng = np.random.default_rng(0)
    halfA = np.zeros(len(y), bool)
    for s in np.unique(scene):
        for c in np.unique(cells[scene == s]):
            idx = np.where((scene == s) & (cells == c))[0]
            halfA[rng.permutation(idx)[: len(idx) // 2]] = True
    dA, dB = dirs_from(halfA), dirs_from(~halfA)
    ceil = {s: cos(dA[s], dB[s]) for s in dA if s in dB}
    full = dirs_from(np.ones(len(y), bool))
    cross = [cos(full[x], full[z]) for x, z in itertools.combinations(sorted(full), 2)]

    Xc = X_img - X_img.mean(0)
    pc1 = unit(np.linalg.svd(Xc, full_matrices=False)[2][0])
    Xr = Xc - np.outer(Xc @ pc1, pc1)
    _save = X_img
    globals()["X_img"] = Xr
    dA2, dB2 = dirs_from(halfA), dirs_from(~halfA)
    ceil2 = {s: cos(dA2[s], dB2[s]) for s in dA2 if s in dB2}
    full2 = dirs_from(np.ones(len(y), bool))
    cross2 = [cos(full2[x], full2[z]) for x, z in itertools.combinations(sorted(full2), 2)]
    globals()["X_img"] = _save

    res["noise_ceiling_within_scene"] = {str(k): v for k, v in ceil.items()}
    res["noise_ceiling_mean"] = float(np.mean(list(ceil.values()))) if ceil else None
    res["cross_scene_mean"] = float(np.mean(cross)) if cross else None
    res["noise_ceiling_mean_pc1removed"] = float(np.mean(list(ceil2.values()))) if ceil2 else None
    res["cross_scene_mean_pc1removed"] = float(np.mean(cross2)) if cross2 else None

    print("\n[probe] ===== G3 NOISE CEILING =====", flush=True)
    print(f"    within-scene split-half cos (CEILING) : {res['noise_ceiling_mean']}", flush=True)
    print(f"    cross-scene cos                       : {res['cross_scene_mean']}", flush=True)
    print(f"    ceiling, top-1 global PC removed      : {res['noise_ceiling_mean_pc1removed']}", flush=True)
    print(f"    cross-scene, top-1 global PC removed  : {res['cross_scene_mean_pc1removed']}", flush=True)
    print("    per-scene ceiling:", {k: round(v, 3) for k, v in ceil.items()}, flush=True)
    print("\n    READING: ceiling ~0 => estimator is noise, the CAFT negative is UNFALSIFIABLE.", flush=True)
    print("             ceiling high AND cross-scene ~0 => the negative is REAL and scene-specific.", flush=True)

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n[probe] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
