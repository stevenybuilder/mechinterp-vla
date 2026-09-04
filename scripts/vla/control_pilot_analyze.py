#!/usr/bin/env python
"""Analysis for the image-KV control pilot: per cell x lambda touch classes + flip-rate plot."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rd, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (rd / "per_episode.jsonl").open() if l.strip()]
    manifest = json.loads((rd / "manifest.json").read_text())

    by = defaultdict(list)
    for r in rows:
        by[(r["cell"], r["task_id"], r["lam"])].append(r)
    lams = sorted({r["lam"] for r in rows})
    cells = sorted({(r["cell"], r["task_id"]) for r in rows})

    res = {"manifest": manifest, "lambdas": lams, "per_cell": {}}
    lines = ["# Image-position KV control pilot", "",
             f"Edit: prefix KV cache, VLM layers {manifest['edit_layers']}, VALID IMAGE POSITIONS ONLY, "
             f"K and V += lambda*(KV_B - KV_A), recomputed on the current observation at every replan "
             f"(every {manifest['args']['n_action_steps']} env steps). Instruction/format/state positions untouched.", "",
             "`A` = object named in the prompt actually used for the rollout; `B` = object named in the "
             "instruction whose image-KV signature is added; `third` = any other object.", ""]
    lines.append("## Sanity: lambda=0 vs unedited rollout")
    for s in manifest.get("sanity", []):
        lines.append(f"- init {s['init']}: identical={s['identical']} (n_edits={s['n_edits_lam0']}, "
                     f"steps={s['n_steps']}, max|Δeef|={s['max_abs_eef_diff']}, prefix={s['prefix_info']})")
    lines += ["", "## Per cell x lambda (n episodes; % first-touch class)", "",
              "| cell | task | A (prompted) | B (added) | lambda | n | touch A % | touch B % [95% CI] | third % | none % | success % |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    plot = {}
    for cell, tid in cells:
        info = next(c for c in manifest["cells"] if c["cell"] == cell and c["task_id"] == tid)
        key = f"{cell}|t{tid}"
        res["per_cell"][key] = {"info": info, "by_lambda": {}}
        ys = []
        for lam in lams:
            rr = by[(cell, tid, lam)]
            n = len(rr)
            cnt = defaultdict(int)
            for r in rr:
                cnt[r["touch_class"]] += 1
            succ = sum(1 for r in rr if r["success"])
            ci = wilson(cnt["B"], n)
            d = {"n": n, "A": cnt["A"], "B": cnt["B"], "third": cnt["third"], "none": cnt["none"],
                 "success": succ, "fracB": round(cnt["B"] / n, 3) if n else None, "B_ci95": ci}
            res["per_cell"][key]["by_lambda"][str(lam)] = d
            ys.append(cnt["B"] / n if n else np.nan)
            pc = lambda k: f"{100*cnt[k]/n:.0f}" if n else "-"
            lines.append(f"| {cell} | t{tid} | {info['objA']} | {info['objB']} | {lam} | {n} | {pc('A')} | "
                         f"{pc('B')} [{ci[0]},{ci[1]}] | {pc('third')} | {pc('none')} | "
                         f"{100*succ/n:.0f} |" if n else "")
        plot[key] = ys
    lines.append("")

    # ketchup control comparison
    kk = [k for k in res["per_cell"] if k.endswith("|t2") or k.endswith("|t3")]
    if len(kk) == 2:
        lines += ["## Ketchup control (t2 and t3 receive the IDENTICAL prompt A = 'pick up the ketchup ...')", "",
                  "| lambda | t2 touch A/B/third | t3 touch A/B/third |", "|---|---|---|"]
        k2 = [k for k in kk if k.endswith("|t2")][0]
        k3 = [k for k in kk if k.endswith("|t3")][0]
        for lam in lams:
            a = res["per_cell"][k2]["by_lambda"].get(str(lam))
            b = res["per_cell"][k3]["by_lambda"].get(str(lam))
            if not a or not b:
                continue
            lines.append(f"| {lam} | {a['A']}/{a['B']}/{a['third']} (n={a['n']}) | "
                         f"{b['A']}/{b['B']}/{b['third']} (n={b['n']}) |")
        lines.append("")

    # verdict
    flips = {k: max((v["fracB"] or 0) for lam, v in d["by_lambda"].items() if lam != "0.0")
             for k, d in res["per_cell"].items()}
    best_lam = {}
    for lam in lams:
        if lam == 0.0:
            continue
        best_lam[lam] = sum(1 for k, d in res["per_cell"].items()
                            if (d["by_lambda"].get(str(lam), {}).get("fracB") or 0) >= 0.5)
    strong = any(v >= 2 for v in best_lam.values())
    n_any = sum(1 for v in flips.values() if v >= 0.2)
    verdict = "STRONG" if strong else ("WEAK" if n_any >= 1 else "NO SIGNAL")
    res["verdict"] = verdict
    res["max_fracB_per_cell_over_lambda>0"] = flips
    res["n_cells_with_fracB>=0.5_by_lambda"] = best_lam
    lines += ["## Verdict", "", f"**{verdict}**", "",
              "max fraction touching B first (over lambda>0), per cell: "
              + ", ".join(f"{k}={v:.2f}" for k, v in flips.items()), "",
              "cells with >=50% flip, by lambda: " + ", ".join(f"{k}:{v}" for k, v in best_lam.items()), ""]

    (out / "summary.md").write_text("\n".join(lines))
    (out / "results.json").write_text(json.dumps(res, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, ys in plot.items():
            ax.plot(lams, ys, marker="o", label=k)
        ax.set_xlabel("lambda (coefficient on KV_B[IMG] - KV_A[IMG], VLM layers 12-17)")
        ax.set_ylabel("fraction of episodes first-touching object B")
        ax.set_ylim(-0.03, 1.03)
        ax.axhline(0.5, ls="--", c="grey", lw=0.8)
        ax.legend(fontsize=7)
        ax.set_title("Image-position KV steering: flip rate vs lambda")
        fig.tight_layout()
        fig.savefig(out / "flip_vs_lambda.png", dpi=150)
    except Exception as e:
        print("plot failed:", e)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
