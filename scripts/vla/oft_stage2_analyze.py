#!/usr/bin/env python
"""Analysis for oft_stage2.py outputs -> results.json + summary.md in the same directory.

(d) per-layer target-identity probe at INSTR positions (logistic regression on PCA-64 features, grouped 5-fold CV over
    init states, 200 shuffled-label nulls -> 95th percentile);
(c) attention mass action-query -> INSTR per layer (mean over heads and items) vs uniform expectation and vs
    matched-count random IMG columns; top heads;
(b) patching recovery: per (dir, site, layer, variant) the least-squares recovery slope
    R_slope = sum((m_p - m_dst)(m_src - m_dst)) / sum((m_src - m_dst)^2) on the A-B axis metric and on the full 56-d
    chunk (R_full), with 10k bootstrap over items (task x init x t); median per-item R on items whose swap effect
    exceeds the 25th percentile of |m_src - m_dst|.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def boot_slope(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    xs, ys = x[idx], y[idx]
    sl = (xs * ys).sum(1) / np.maximum((xs * xs).sum(1), 1e-12)
    pt = float((x * y).sum() / max((x * x).sum(), 1e-12))
    return pt, float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))


def probe(features, labels, groups, n_null=200, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def cv_acc(y):
        accs = []
        for tr, te in GroupKFold(5).split(features, y, groups):
            clf = make_pipeline(StandardScaler(), PCA(64, random_state=0),
                                LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(features[tr], y[tr])
            accs.append(float((clf.predict(features[te]) == y[te]).mean()))
        return float(np.mean(accs))

    acc = cv_acc(labels)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        y = labels.copy()
        for g in np.unique(groups):  # shuffle within init state
            m = groups == g
            y[m] = rng.permutation(y[m])
        null.append(cv_acc(y))
    return acc, float(np.percentile(null, 95)), float(np.mean(null))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n_null", type=int, default=200)
    ap.add_argument("--probe_layers", default="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32")
    args = ap.parse_args()
    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text())
    res = {"manifest": {k: v for k, v in man.items() if k != "tasks"}, "tasks": man["tasks"]}
    lines = [f"# OFT Stage 2 summary ({d.name})", "", f"checkpoint={man['checkpoint']} suite={man['suite']} "
             f"n_init={man['n_init']} timepoints={man['timepoints']} layers={man['layers']} passes={man.get('n_passes')}",
             "", f"verify: `{json.dumps(man.get('verify'))}`", ""]

    # ---------- (d) probe ----------
    z = np.load(d / "features.npz")
    keys = [k.split("|") for k in z["keys"]]
    sel = [i for i, k in enumerate(keys) if k[3] in ("A", "B") and int(k[2]) == 0]  # t=0, prompts A and B
    labels = np.array([{"A": man["tasks"][k[0]]["A"], "B": man["tasks"][k[0]]["B"]}[k[3]] for k in keys])
    lab_ids = {l: i for i, l in enumerate(sorted(set(labels)))}
    y = np.array([lab_ids[l] for l in labels])[sel]
    groups = np.array([int(k[1]) for k in keys])[sel]
    probe_res = {}
    lines += ["## (d) Target-identity probe at INSTR positions (10-way instruction identity; prompts A+B at t=0; "
              f"n={len(sel)}; grouped 5-fold CV over init states; {args.n_null} within-init shuffled-label nulls)", "",
              "| layer | acc INSTR_mean | null95 | acc ACT_mean | null95 | acc PROPRIO | null95 |", "|---|---|---|---|---|---|---|"]
    for L in [int(x) for x in args.probe_layers.split(",")]:
        row = {}
        for feat in ("INSTR_mean", "ACT_mean", "PROPRIO"):
            X = z[feat][sel, L, :].astype(np.float32)
            acc, n95, nmean = probe(X, y, groups, n_null=args.n_null if feat == "INSTR_mean" else min(50, args.n_null))
            row[feat] = {"acc": acc, "null95": n95, "null_mean": nmean}
        probe_res[L] = row
        lines.append(f"| {L} | " + " | ".join(f"{row[f]['acc']:.3f} | {row[f]['null95']:.3f}" for f in
                                             ("INSTR_mean", "ACT_mean", "PROPRIO")) + " |")
    res["probe"] = probe_res
    lines.append("")
    lines.append("Note: INSTR positions literally contain the instruction tokens, so a high INSTR probe is a sanity "
                 "check (readout exists), not a finding; ACT_mean is the informative row (does identity reach the "
                 "action stream).")

    # ---------- (c) attention ----------
    att = [json.loads(l) for l in (d / "attention.jsonl").open()]
    by_layer = defaultdict(list)
    for r in att:
        by_layer[r["layer"]].append(r)
    attn_res = {}
    lines += ["", "## (c) Attention mass from the 56 action-query rows (prompt A, mean over rows, heads, items)", "",
              "| layer | INSTR | uniform |INSTR|/keys | rand IMG cols (matched count) | IMG | PROPRIO | FMT | ACT | sink top5 | max-head INSTR |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for L in sorted(by_layer):
        rows = by_layer[L]
        m = lambda k: float(np.mean([np.mean(r[k]) for r in rows]))
        uni = float(np.mean([r["n_instr"] / r["n_keys"] for r in rows]))
        rand = float(np.mean([np.mean(r["rand_img"]) for r in rows]))
        instr_heads = np.mean([r["INSTR"] for r in rows], axis=0)
        attn_res[L] = {"INSTR": m("INSTR"), "uniform": uni, "rand_img": rand, "IMG": m("IMG"), "PROPRIO": m("PROPRIO"),
                       "FMT": m("FMT"), "ACT": m("ACT"), "BOS": m("BOS"), "STOP": m("STOP"), "sink_top5": m("sink_top5"),
                       "instr_per_head": instr_heads.tolist(), "max_head": int(np.argmax(instr_heads)),
                       "max_head_instr": float(instr_heads.max())}
        a = attn_res[L]
        lines.append(f"| {L} | {a['INSTR']:.4f} | {uni:.4f} | {rand:.4f} | {a['IMG']:.3f} | {a['PROPRIO']:.4f} | "
                     f"{a['FMT']:.3f} | {a['ACT']:.3f} | {a['sink_top5']:.3f} | h{a['max_head']} {a['max_head_instr']:.3f} |")
    res["attention"] = attn_res
    tot_instr = np.mean([attn_res[L]["INSTR"] for L in attn_res])
    tot_rand = np.mean([attn_res[L]["rand_img"] for L in attn_res])
    lines += ["", f"Mean over layers: INSTR {tot_instr:.4f} vs matched-count random IMG columns {tot_rand:.4f} "
              f"(ratio {tot_instr / max(tot_rand, 1e-9):.2f}x)."]

    # ---------- (b) patching ----------
    pr = [json.loads(l) for l in (d / "patching.jsonl").open()]
    groups_p = defaultdict(list)
    for r in pr:
        groups_p[(r["dir"], r["site"], r["layer"], "rand" if r["variant"].startswith("rand") else r["variant"])].append(r)
    swap = {}
    for r in pr:
        swap[(r["task"], r["init"], r["t"], r["dir"])] = abs(r["m_src"] - r["m_dst"])
    thr = float(np.percentile(list(swap.values()), 25))
    patch_res = {}
    lines += ["", "## (b) Residual patching (10k bootstrap over items = task x init x t)", "",
              f"Full-prompt-swap effect on the A-B axis |m_src - m_dst|: median {np.median(list(swap.values())):.4f}, "
              f"25th pct {thr:.4f} (gate for per-item R), n items {len(swap)}", "",
              "| dir | site | layer | variant | n | R_slope axis [95% CI] | R_slope full-chunk [95% CI] | median per-item R_axis (gated) | n gated |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(groups_p):
        rows = groups_p[key]
        x = [r["m_src"] - r["m_dst"] for r in rows]
        yv = [r["m_patched"] - r["m_dst"] for r in rows]
        pt, lo, hi = boot_slope(x, yv)
        rf = [r["R_full"] for r in rows if r["R_full"] is not None]
        # full-chunk slope via bootstrap of mean R_full weighted by denominators -> use direct slope on 56-d: R_full is
        # already the per-item projection; aggregate with a bootstrap of its mean
        rng = np.random.default_rng(1)
        rf_a = np.array(rf)
        bs = rf_a[rng.integers(0, len(rf_a), size=(10000, len(rf_a)))].mean(1)
        gated = [r["R_axis"] for r in rows if r["R_axis"] is not None and abs(r["m_src"] - r["m_dst"]) >= thr]
        patch_res["|".join(map(str, key))] = {"n": len(rows), "R_axis_slope": pt, "ci": [lo, hi],
                                              "R_full_mean": float(rf_a.mean()),
                                              "R_full_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                              "median_R_axis_gated": float(np.median(gated)) if gated else None,
                                              "n_gated": len(gated)}
        p = patch_res["|".join(map(str, key))]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(rows)} | {pt:.3f} [{lo:.3f},{hi:.3f}] | "
                     f"{p['R_full_mean']:.3f} [{p['R_full_ci'][0]:.3f},{p['R_full_ci'][1]:.3f}] | "
                     f"{p['median_R_axis_gated'] if p['median_R_axis_gated'] is None else round(p['median_R_axis_gated'], 3)} | {len(gated)} |")
    res["patching"] = patch_res
    res["swap_gate_threshold"] = thr
    lines += ["", "Reading: R=1 means the patched run reproduces the full-prompt-swap chunk; R=0 means no effect. "
              "Compare 'patch' with 'rand' (equal-norm random direction) and 'resample' (unrelated instruction C) on the same "
              "row. Direction B->A = wrong-object activations into the correct run (noising); A->B = correct into wrong (denoising)."]
    (d / "results.json").write_text(json.dumps(res, indent=2))
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
