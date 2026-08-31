#!/usr/bin/env python
"""Stage 0 analysis: per suite x condition success / touch rates with Wilson CIs, paired McNemar per init state,
swap effect (correct vs wrong_object), and the gate verdict.

Usage: python scripts/vla/analyze_stage0.py --inputs runs/stage0/libero_object.jsonl runs/stage0/libero_goal.jsonl \
           --out artifacts/vla_stage0/<timestamp>
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def mcnemar(pairs):
    """pairs: list of (a, b) booleans. Exact binomial McNemar on discordant pairs. Returns dict."""
    b = sum(1 for a, c in pairs if a and not c)
    c = sum(1 for a, c in pairs if c and not a)
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n > 0 else 1.0
    return {"n_pairs": len(pairs), "a_only": b, "b_only": c, "p": p}


def touch_metrics(r):
    """Return dict of behavioural booleans for one episode record."""
    ft = r["first_touch"]["object"] if r["first_touch"] else None
    fg = r["first_grasp"]["object"] if r["first_grasp"] else None
    tgt = r.get("target_object")
    prompted = r.get("wrong_object") if r["condition"] == "wrong_object" else (tgt if r["condition"] != "null" else None)
    return {
        "success": bool(r["success"]),
        "touched_target_first": ft is not None and ft == tgt,       # scene-correct object (memorised task)
        "touched_target_any": tgt in (r.get("touched_any") or []),
        "grasped_target_first": fg is not None and fg == tgt,
        "touched_prompted_first": prompted is not None and ft == prompted,  # obeyed the prompt (wrong_object only)
        "touched_prompted_any": prompted is not None and prompted in (r.get("touched_any") or []),
        "touched_anything": ft is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate_pp", type=float, default=20.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = []
    for f in args.inputs:
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    with (out / "per_episode.jsonl").open("w") as fh:
        for r in recs:
            slim = {k: v for k, v in r.items() if k not in ("eef_pos", "object_pos", "contact_timeline")}
            slim["metrics"] = touch_metrics(r)
            fh.write(json.dumps(slim) + "\n")

    metrics = ["success", "touched_target_first", "grasped_target_first", "touched_prompted_first",
               "touched_prompted_any", "touched_anything"]
    by = defaultdict(dict)  # (suite, task, init) -> {cond: metrics}
    for r in recs:
        by[(r["suite"], r["task_id"], r["init_id"])][r["condition"]] = touch_metrics(r)
    suites = sorted({r["suite"] for r in recs})
    conds = ["correct", "null", "wrong_object"]
    results = {"n_episodes": len(recs), "suites": {}, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "policy": recs[0].get("policy"), "revision": recs[0].get("revision")}
    lines = ["# Stage 0 summary", "", f"policy={recs[0].get('policy')} revision={recs[0].get('revision')} "
             f"n_episodes={len(recs)}", ""]
    gate_pass = False
    for s in suites:
        keys = [k for k in by if k[0] == s]
        res = {"n_units": len(keys), "rates": {}, "paired": {}, "per_task": {}}
        lines += [f"## {s}  (units = task x init state: {len(keys)})", "",
                  "| condition | n | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
        for c in conds:
            row = {}
            units = [by[k][c] for k in keys if c in by[k]]
            for m in metrics:
                k_ = sum(1 for u in units if u[m])
                p, lo, hi = wilson(k_, len(units))
                row[m] = {"k": k_, "n": len(units), "rate": p, "ci": [lo, hi]}
            res["rates"][c] = row
            lines.append(f"| {c} | {len(units)} | " + " | ".join(
                f"{row[m]['rate']*100:.1f} [{row[m]['ci'][0]*100:.1f},{row[m]['ci'][1]*100:.1f}]" for m in metrics) + " |")
        # paired contrasts
        for m in ["success", "touched_target_first"]:
            for c in ["null", "wrong_object"]:
                pairs = [(by[k]["correct"][m], by[k][c][m]) for k in keys if "correct" in by[k] and c in by[k]]
                mc = mcnemar(pairs)
                gap = (sum(a for a, _ in pairs) - sum(b for _, b in pairs)) / max(1, len(pairs)) * 100
                res["paired"][f"{m}:correct_vs_{c}"] = {"gap_pp": gap, **mc}
                lines.append(f"- {m}: correct − {c} = {gap:+.1f} pp (McNemar exact p={mc['p']:.3g}, "
                             f"discordant {mc['a_only']}/{mc['b_only']}, n={mc['n_pairs']})")
        # per task
        tasks = sorted({k[1] for k in keys})
        lines += ["", "| task | " + " | ".join(f"{c} succ" for c in conds) + " | " + " | ".join(f"{c} touch-tgt-first" for c in conds) + " |",
                  "|---|" + "---|" * (2 * len(conds))]
        for t in tasks:
            tk = [k for k in keys if k[1] == t]
            row = {}
            for c in conds:
                us = [by[k][c] for k in tk if c in by[k]]
                row[c] = {"n": len(us), "success": sum(u["success"] for u in us),
                          "touched_target_first": sum(u["touched_target_first"] for u in us)}
            res["per_task"][t] = row
            lines.append(f"| {t} | " + " | ".join(f"{row[c]['success']}/{row[c]['n']}" for c in conds) + " | " +
                         " | ".join(f"{row[c]['touched_target_first']}/{row[c]['n']}" for c in conds) + " |")
        swap = res["paired"]["success:correct_vs_wrong_object"]["gap_pp"]
        swap_touch = res["paired"]["touched_target_first:correct_vs_wrong_object"]["gap_pp"]
        null_gap = res["paired"]["success:correct_vs_null"]["gap_pp"]
        res["swap_effect_success_pp"] = swap
        res["swap_effect_touch_pp"] = swap_touch
        res["null_gap_success_pp"] = null_gap
        usable = swap >= args.gate_pp or swap_touch >= args.gate_pp
        res["swap_effect_usable"] = usable
        gate_pass = gate_pass or usable
        lines += ["", f"**{s}: swap effect (correct − wrong_object) = {swap:+.1f} pp success, {swap_touch:+.1f} pp "
                  f"touch-target-first; null gap = {null_gap:+.1f} pp success → "
                  f"{'USABLE (≥%g pp)' % args.gate_pp if usable else 'not usable'}**", ""]
        results["suites"][s] = res
    results["gate"] = {"rule": f"swap effect ≥ {args.gate_pp} pp (success or touch-target-first) on ≥1 suite",
                       "pass": gate_pass}
    lines += ["## GATE", f"Rule: {results['gate']['rule']}", f"**Verdict: {'PASS' if gate_pass else 'FAIL'}**"]
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
