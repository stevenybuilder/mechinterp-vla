#!/usr/bin/env python
"""Stage 2 discovery analysis (protocol §1.2, §4.2–4.4, §7).

Per (pair, init, tp, direction): m_A = clean, m_B = full_swap, SD_noise = pooled SD over noise seeds of clean and
full_swap.  Denominator gate |m_B − m_A| ≥ 3·SD.  For each condition: z = (m − m_A)/SD, R = (m − m_A)/(m_B − m_A)
(gated units only), plus L2 distance of the k=10 chunk to clean-A and to full-swap chunks (normalised by |A−B|).
Aggregation: unit = init state (timepoints averaged), median R with 10k paired bootstrap CI over init states;
Wilcoxon signed-rank KV[seg] vs RD[seg] and vs RS[seg] (Holm over the 3 confirmatory contrasts INSTR/IMG/FMT).
Attention mass by segment (mean over steps/heads) per expert layer.  Readout probes: multinomial logistic regression
(10-way task identity) on segment-mean residual / K / V per VLM layer, leave-one-init-out, shuffled-label null.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def boot_median_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    meds = np.median(x[idx], axis=1)
    return (float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def analyse_rows(rows, out, tag):
    by = defaultdict(list)
    for r in rows:
        by[(r["pair"], r["init"], r["tp"], r["direction"])].append(r)
    units = {}
    for key, rs in by.items():
        clean = [r for r in rs if r["condition"] == "clean"]
        full = [r for r in rs if r["condition"] == "full_swap"]
        mA = [r["m10"] for r in clean if r["seed"] == 0][0]
        mB = [r["m10"] for r in full if r["seed"] == 0][0]
        sd = np.sqrt(0.5 * (np.var([r["m10"] for r in clean], ddof=1) + np.var([r["m10"] for r in full], ddof=1)))
        chA = np.array([r["chunk10"] for r in clean if r["seed"] == 0][0])
        chB = np.array([r["chunk10"] for r in full if r["seed"] == 0][0])
        dAB = np.linalg.norm((chA - chB)[:, :6])
        gate = abs(mB - mA) >= 3 * sd
        conds = {}
        for r in rs:
            if r["seed"] != 0:
                continue
            ch = np.array(r["chunk10"])
            conds[r["condition"]] = {
                "m": r["m10"], "z": (r["m10"] - mA) / sd if sd > 0 else np.nan,
                "R": (r["m10"] - mA) / (mB - mA) if gate else np.nan,
                "l2_to_A": float(np.linalg.norm((ch - chA)[:, :6]) / (dAB + 1e-9)),
                "l2_to_B": float(np.linalg.norm((ch - chB)[:, :6]) / (dAB + 1e-9)),
            }
        units[key] = {"mA": mA, "mB": mB, "sd": sd, "gate": bool(gate), "swap_z": (mB - mA) / sd if sd > 0 else np.nan,
                      "conds": conds}
    # aggregate per init (average tps) per pair × direction
    res = {"tag": tag, "n_units": len(units), "pairs": {}}
    lines = [f"# Stage 2 discovery — {tag}", "", f"units (pair,init,tp,direction) = {len(units)}", ""]
    pairs = sorted({k[0] for k in units})
    directions = ["A<-B", "B<-A"]
    cond_names = sorted({c for u in units.values() for c in u["conds"]}, key=lambda s: (s.split("[")[0], s))
    for pair in pairs:
        res["pairs"][pair] = {}
        for d in directions:
            keys = [k for k in units if k[0] == pair and k[3] == d]
            inits = sorted({k[1] for k in keys})
            gate_frac = np.mean([units[k]["gate"] for k in keys]) if keys else np.nan
            swap_z = [units[k]["swap_z"] for k in keys]
            lines += [f"## {pair}  direction {d}  (n init={len(inits)}, units={len(keys)}, gate pass={gate_frac*100:.0f}%, "
                      f"median swap z={np.nanmedian(swap_z):.1f})", "",
                      "| condition | median R [95% CI] (gated units) | median z | median L2→A / L2→B (÷|A−B|) | n |",
                      "|---|---|---|---|---|"]
            tbl = {}
            for c in cond_names:
                Rs, zs, lA, lB = [], [], [], []
                for i in inits:
                    ks = [k for k in keys if k[1] == i and c in units[k]["conds"]]
                    if not ks:
                        continue
                    rr = [units[k]["conds"][c]["R"] for k in ks if units[k]["gate"]]
                    if rr:
                        Rs.append(np.mean(rr))
                    zs.append(np.mean([units[k]["conds"][c]["z"] for k in ks]))
                    lA.append(np.mean([units[k]["conds"][c]["l2_to_A"] for k in ks]))
                    lB.append(np.mean([units[k]["conds"][c]["l2_to_B"] for k in ks]))
                med, lo, hi = boot_median_ci(Rs)
                tbl[c] = {"R_median": med, "R_ci": [lo, hi], "R_per_init": Rs, "z_median": float(np.nanmedian(zs)) if zs else np.nan,
                          "l2A_median": float(np.median(lA)) if lA else np.nan, "l2B_median": float(np.median(lB)) if lB else np.nan,
                          "n_init": len(Rs)}
                lines.append(f"| {c} | {med:.2f} [{lo:.2f}, {hi:.2f}] | {tbl[c]['z_median']:.1f} | "
                             f"{tbl[c]['l2A_median']:.2f} / {tbl[c]['l2B_median']:.2f} | {len(Rs)} |")
            # confirmatory contrasts (Holm m=3): KV[seg]@all vs RD[seg]@all, and vs RS[seg]@all
            contrasts = {}
            pv = []
            for seg in ("INSTR", "IMG", "FMT"):
                a, b = tbl.get(f"KV[{seg}]@all", {}).get("R_per_init", []), tbl.get(f"RD[{seg}]@all", {}).get("R_per_init", [])
                n = min(len(a), len(b))
                if n >= 5:
                    p = wilcoxon(a[:n], b[:n]).pvalue if not np.allclose(a[:n], b[:n]) else 1.0
                else:
                    p = np.nan
                contrasts[f"KV[{seg}] vs RD[{seg}]"] = {"p": p, "n": n}
                pv.append(p)
            adj = holm(np.nan_to_num(np.array(pv), nan=1.0))
            for (k, v), pa in zip(contrasts.items(), adj):
                v["p_holm"] = float(pa)
            for seg in ("INSTR", "IMG", "FMT", "OBJ", "INSTR+FMT", "IMG+FMT"):
                a, b = tbl.get(f"KV[{seg}]@all", {}).get("R_per_init", []), tbl.get(f"RS[{seg}]@all", {}).get("R_per_init", [])
                n = min(len(a), len(b))
                p = wilcoxon(a[:n], b[:n]).pvalue if n >= 5 and not np.allclose(a[:n], b[:n]) else np.nan
                contrasts[f"KV[{seg}] vs RS[{seg}]"] = {"p": p, "n": n}
            lines += ["", "Contrasts (Wilcoxon over init states; Holm on the 3 KV-vs-RD):"]
            for k, v in contrasts.items():
                lines.append(f"- {k}: p={v['p']:.3g}" + (f" (Holm {v['p_holm']:.3g})" if "p_holm" in v else "") + f", n={v['n']}")
            lines.append("")
            res["pairs"][pair][d] = {"gate_frac": float(gate_frac), "swap_z_median": float(np.nanmedian(swap_z)),
                                     "conditions": {c: {k: v for k, v in t.items() if k != "R_per_init"} for c, t in tbl.items()},
                                     "contrasts": contrasts}
    return res, lines


def analyse_attention(npz_files, lines, res):
    A, B = [], []
    segs = None
    for f in npz_files:
        z = np.load(f)
        A.append(z["attn_A"].astype(float))
        B.append(z["attn_B"].astype(float))
        segs = [str(s) for s in z["attn_segments"]]
    if not A:
        return
    A = np.stack(A)  # [n, 18, steps, heads, seg]
    massA = A.mean(axis=(0, 2, 3))  # [18, seg]
    lines += ["## Attention mass action→prefix segment (clean A, mean over units/steps/heads; per expert layer)", "",
              "| layer | " + " | ".join(segs) + " |", "|---|" + "---|" * len(segs)]
    for L in range(massA.shape[0]):
        lines.append(f"| {L} | " + " | ".join(f"{massA[L, s]:.3f}" for s in range(len(segs))) + " |")
    n_pos = {}
    lines += ["", "Per-position mass (mass / n positions in segment) is in results.json; IMG has 512 positions, "
              "INSTR ~9–10, FMT ~11–13, STATE ~121.", ""]
    res["attention_mass_by_layer"] = {"segments": segs, "clean_A": massA.tolist(),
                                      "per_head_std_over_units": A.std(axis=0).mean(axis=(1, 2)).tolist()}


def analyse_probes(npz_files, lines, res):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        lines.append("sklearn missing: probes skipped")
        return
    X = {"resid": [], "K": [], "V": []}
    y, groups = [], []
    segs = None
    for f in npz_files:
        z = np.load(f)
        if "resid" not in z:
            continue
        m = re.search(r"_i(\d+)_tp(\d+)", f)
        for j in range(len(z["labels"])):
            for k in X:
                X[k].append(z[k][j].astype(np.float32))
            y.append(int(z["labels"][j]))
            groups.append(int(m.group(1)))
        segs = [str(s) for s in z["feat_segments"]]
    if not y:
        return
    y = np.array(y)
    groups = np.array(groups)
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        lines.append("probes skipped: need >= 2 init states")
        return
    folds = [uniq[i::5] for i in range(5)]  # 5-fold over init states
    rng = np.random.default_rng(0)
    out = {}
    lines += ["## Readout probes: 10-way task identity, logistic regression, 5-fold by init state (acc; shuffled-label null 95th pct)", ""]
    for feat in ("resid", "K", "V"):
        arr = np.stack(X[feat])  # [n, 18, seg, d]
        out[feat] = {}
        lines += [f"### {feat}", "", "| layer | " + " | ".join(segs) + " |", "|---|" + "---|" * len(segs)]
        for L in range(arr.shape[1]):
            row = []
            for s in range(arr.shape[2]):
                F = arr[:, L, s, :]
                F = (F - F.mean(0)) / (F.std(0) + 1e-6)
                accs, nulls = [], []
                for fold in folds:
                    te = np.isin(groups, fold)
                    if te.sum() == 0 or (~te).sum() == 0:
                        continue
                    clf = LogisticRegression(max_iter=300, C=0.5)
                    clf.fit(F[~te], y[~te])
                    accs.append(clf.score(F[te], y[te]))
                    ys = y[~te].copy()
                    for _ in range(3):
                        rng.shuffle(ys)
                        clf2 = LogisticRegression(max_iter=100, C=0.5).fit(F[~te], ys)
                        nulls.append(clf2.score(F[te], y[te]))
                if not accs:
                    row.append("n/a")
                    continue
                acc, null95 = float(np.mean(accs)), float(np.percentile(nulls, 95))
                out[feat][f"L{L}_{segs[s]}"] = {"acc": acc, "null95": null95}
                row.append(f"{acc*100:.0f} ({null95*100:.0f})")
            lines.append(f"| {L} | " + " | ".join(row) + " |")
        lines.append("")
    res["probes"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--probes", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.run_dir) / "rows.jsonl")
    res, lines = analyse_rows(rows, out, args.tag or Path(args.run_dir).name)
    npz = sorted(glob.glob(str(Path(args.run_dir) / "features_*.npz")))
    analyse_attention(npz, lines, res)
    if args.probes:
        analyse_probes(npz, lines, res)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
