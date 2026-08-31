#!/usr/bin/env python
"""Cross-distractor arbitration analysis.

Codes every wrong-instruction episode into a three-way outcome and asks the key question: does the
OBEY / IGNORE / JAM mode vary WITHIN a task across different valid wrong instructions? If it does not, the
mode is a per-task constant (effective n = 10 tasks) and any probe predicting mode is just a task classifier.

Outcome coding (from the LIBERO-CF style contact record):
  OBEY   first object touched == the object named in the (wrong) prompt
  IGNORE first object touched == the task's own memorised target object
  JAM    first object touched is neither (a third object, the basket, or nothing at all)

Inputs: one or more per-episode JSONL files (Stage-0 `wrong_object` rows and the cross grid's `wrong:<j>` rows).
Outputs: results.json, per_episode.jsonl, summary.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

OUTCOMES = ["OBEY", "IGNORE", "JAM"]


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def code_outcome(rec: dict) -> tuple[str, str]:
    """Return (outcome, detail). detail distinguishes JAM sub-kinds."""
    ft = rec.get("first_touch")
    obj = ft["object"] if ft else None
    prompted = rec.get("wrong_object")
    correct = rec.get("target_object")
    if obj is None:
        return "JAM", "touched_nothing"
    if prompted is not None and obj == prompted:
        return "OBEY", "touched_prompted"
    if correct is not None and obj == correct:
        return "IGNORE", "touched_task_target"
    if obj.startswith("basket"):
        return "JAM", "touched_basket"
    return "JAM", f"touched_other:{obj}"


def chi2_stat(table: list[list[int]]) -> float:
    rows = len(table)
    cols = len(table[0]) if rows else 0
    n = sum(sum(r) for r in table)
    if n == 0:
        return 0.0
    rs = [sum(r) for r in table]
    cs = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    s = 0.0
    for i in range(rows):
        for j in range(cols):
            e = rs[i] * cs[j] / n
            if e > 0:
                s += (table[i][j] - e) ** 2 / e
    return s


def perm_p(labels: list[str], outcomes: list[str], n_perm: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Permutation test of independence between the group label (distractor) and the outcome."""
    lab_ix = {v: i for i, v in enumerate(sorted(set(labels)))}
    out_ix = {v: i for i, v in enumerate(sorted(set(outcomes)))}
    if len(lab_ix) < 2 or len(out_ix) < 2:
        return 0.0, 1.0

    def build(o):
        t = [[0] * len(out_ix) for _ in lab_ix]
        for a, b in zip(labels, o):
            t[lab_ix[a]][out_ix[b]] += 1
        return t

    obs = chi2_stat(build(outcomes))
    rng = random.Random(seed)
    sh = list(outcomes)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(sh)
        if chi2_stat(build(sh)) >= obs - 1e-12:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_perm", type=int, default=20000)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(args.manifest))

    seen = set()
    eps: list[dict] = []
    ref = defaultdict(list)  # (task, condition) -> success for correct/null
    for path in args.inputs:
        for line in Path(path).open():
            r = json.loads(line)
            if r["suite"] != args.suite:
                continue
            cond = r["condition"]
            if cond in ("correct", "null"):
                key = (r["task_id"], r["init_id"], cond)
                if key in seen:
                    continue
                seen.add(key)
                ref[(r["task_id"], cond)].append(bool(r["success"]))
                continue
            if not (cond == "wrong_object" or cond.startswith("wrong:")):
                continue
            wt = r.get("wrong_task_id")
            if wt is None:
                continue
            key = (r["task_id"], r["init_id"], "W", wt)
            if key in seen:
                continue
            seen.add(key)
            outcome, detail = code_outcome(r)
            ft, fg = r.get("first_touch"), r.get("first_grasp")
            eps.append({
                "suite": r["suite"], "task_id": r["task_id"], "task_name": r["task_name"],
                "init_id": r["init_id"], "condition": cond, "source_condition": cond,
                "wrong_task_id": wt, "prompt": r["prompt"],
                "prompted_object": r.get("wrong_object"), "task_target_object": r.get("target_object"),
                "first_touch_object": ft["object"] if ft else None,
                "first_touch_step": ft["step"] if ft else None,
                "first_grasp_object": fg["object"] if fg else None,
                "first_grasp_step": fg["step"] if fg else None,
                "grasped_prompted": bool(fg and fg["object"] == r.get("wrong_object")),
                "grasped_task_target": bool(fg and fg["object"] == r.get("target_object")),
                "outcome": outcome, "outcome_detail": detail,
                "success": bool(r["success"]), "n_steps": r["n_steps"], "wall_s": r["wall_s"],
                "seed": r.get("seed"), "ts": r.get("ts"),
            })

    with (outdir / "per_episode.jsonl").open("w") as f:
        for e in sorted(eps, key=lambda e: (e["task_id"], e["wrong_task_id"], e["init_id"])):
            f.write(json.dumps(e) + "\n")

    # ---- per-cell stats -------------------------------------------------
    cells = defaultdict(list)
    for e in eps:
        cells[(e["task_id"], e["wrong_task_id"])].append(e)
    cell_stats = {}
    for (t, w), rows in sorted(cells.items()):
        n = len(rows)
        cnt = Counter(r["outcome"] for r in rows)
        props = {}
        for o in OUTCOMES:
            k = cnt.get(o, 0)
            lo, hi = wilson(k, n)
            props[o] = {"k": k, "p": k / n if n else None, "ci95": [round(lo, 4), round(hi, 4)]}
        modal = max(OUTCOMES, key=lambda o: (cnt.get(o, 0), -OUTCOMES.index(o)))
        cell_stats[f"{t}|{w}"] = {
            "task_id": t, "wrong_task_id": w, "n": n,
            "prompted_object": rows[0]["prompted_object"], "task_target_object": rows[0]["task_target_object"],
            "wrong_prompt": rows[0]["prompt"],
            "outcomes": props, "modal_outcome": modal,
            "modal_share": round(cnt.get(modal, 0) / n, 4) if n else None,
            "success_rate": round(sum(r["success"] for r in rows) / n, 4) if n else None,
            "grasped_prompted_rate": round(sum(r["grasped_prompted"] for r in rows) / n, 4) if n else None,
            "median_steps": sorted(r["n_steps"] for r in rows)[n // 2] if n else None,
            "source": sorted({r["source_condition"] for r in rows}),
        }

    # ---- within-task mode variation ------------------------------------
    within = {}
    for t in sorted({e["task_id"] for e in eps}):
        rows = [e for e in eps if e["task_id"] == t]
        by_w = defaultdict(list)
        for r in rows:
            by_w[r["wrong_task_id"]].append(r["outcome"])
        modes = {w: Counter(v).most_common(1)[0][0] for w, v in sorted(by_w.items())}
        labels = [str(r["wrong_task_id"]) for r in rows]
        outs = [r["outcome"] for r in rows]
        stat, p = perm_p(labels, outs, n_perm=args.n_perm, seed=t)
        within[str(t)] = {
            "task_id": t, "n_distractors": len(by_w), "n_episodes": len(rows),
            "modal_outcome_per_distractor": modes,
            "distinct_modes": sorted(set(modes.values())),
            "mode_varies_within_task": len(set(modes.values())) > 1,
            "chi2": round(stat, 3), "perm_p": p,
            "outcome_counts": dict(Counter(outs)),
        }

    # ---- per prompted-object across tasks (the reverse confound check) ---
    by_obj = defaultdict(list)
    for e in eps:
        by_obj[e["prompted_object"]].append(e)
    obj_stats = {}
    for o, rows in sorted(by_obj.items()):
        by_t = defaultdict(list)
        for r in rows:
            by_t[r["task_id"]].append(r["outcome"])
        modes = {t: Counter(v).most_common(1)[0][0] for t, v in sorted(by_t.items())}
        stat, p = perm_p([str(r["task_id"]) for r in rows], [r["outcome"] for r in rows],
                         n_perm=args.n_perm, seed=hash(o) % 10000)
        obj_stats[str(o)] = {
            "prompted_object": o, "n_host_tasks": len(by_t), "n_episodes": len(rows),
            "modal_outcome_per_host_task": modes, "distinct_modes": sorted(set(modes.values())),
            "mode_varies_across_tasks": len(set(modes.values())) > 1,
            "chi2": round(stat, 3), "perm_p": p,
        }

    # ---- the reviewer's ketchup pair ------------------------------------
    ket = {}
    for t in (2, 3):
        rows = [e for e in eps if e["task_id"] == t and e["prompted_object"] == "ketchup_1"]
        if rows:
            c = Counter(r["outcome"] for r in rows)
            ket[f"task_{t}"] = {
                "n": len(rows), "wrong_task_id": rows[0]["wrong_task_id"],
                "task_target_object": rows[0]["task_target_object"], "prompt": rows[0]["prompt"],
                "counts": dict(c), "modal_outcome": c.most_common(1)[0][0],
                "success_rate": round(sum(r["success"] for r in rows) / len(rows), 4),
                "first_touch_objects": dict(Counter(r["first_touch_object"] for r in rows)),
            }
    ket["opposite_modes"] = (
        len({v["modal_outcome"] for k, v in ket.items() if k.startswith("task_")}) > 1
        if len(ket) >= 2 else None
    )

    n_var = sum(1 for v in within.values() if v["mode_varies_within_task"])
    verdict = {
        "n_tasks": len(within),
        "n_tasks_with_within_task_mode_variation": n_var,
        "fraction": round(n_var / len(within), 3) if within else None,
        "n_tasks_perm_p_lt_0.05": sum(1 for v in within.values() if v["perm_p"] < 0.05),
        "confound_broken": n_var > 0,
        "reading": ("mode is NOT a per-task constant: at least one task shows different modal outcomes for "
                    "different valid wrong instructions, so the task/distractor confound is broken"
                    if n_var > 0 else
                    "mode IS a per-task constant across every valid distractor: the task/distractor confound "
                    "is UNBROKEN and any probe predicting mode remains a task classifier"),
    }

    ref_stats = {f"{t}|{c}": {"n": len(v), "success_rate": round(sum(v) / len(v), 4)}
                 for (t, c), v in sorted(ref.items())}

    results = {
        "suite": args.suite,
        "manifest_cells": manifest.get(args.suite, {}).get("n_cells"),
        "n_wrong_episodes": len(eps),
        "n_cells_run": len(cell_stats),
        "outcome_definition": {
            "OBEY": "first touched object == object named in the wrong prompt",
            "IGNORE": "first touched object == the task's own target object",
            "JAM": "first touched object is neither (third object, basket, or nothing)",
        },
        "verdict_within_task_mode_variation": verdict,
        "ketchup_pair": ket,
        "cells": cell_stats,
        "within_task": within,
        "by_prompted_object": obj_stats,
        "reference_conditions": ref_stats,
        "overall_outcome_counts": dict(Counter(e["outcome"] for e in eps)),
    }
    json.dump(results, (outdir / "results.json").open("w"), indent=1)

    # ---- summary.md -----------------------------------------------------
    tasks = sorted({e["task_id"] for e in eps})
    ws = sorted({e["wrong_task_id"] for e in eps})
    L = []
    L.append(f"# Cross-distractor arbitration — {args.suite}\n")
    L.append(f"{len(eps)} wrong-instruction episodes over {len(cell_stats)} (task x wrong-instruction) cells "
             f"(manifest: {results['manifest_cells']} valid cells).\n")
    L.append(f"Outcome coding: OBEY = first touch is the prompted object; IGNORE = first touch is the task's own "
             f"target; JAM = neither.\n")
    L.append(f"Overall: {results['overall_outcome_counts']}\n")
    L.append("\n## VERDICT — does mode vary WITHIN task?\n")
    L.append(f"**{verdict['n_tasks_with_within_task_mode_variation']}/{verdict['n_tasks']} tasks** show more than one "
             f"modal outcome across their valid wrong instructions; "
             f"{verdict['n_tasks_perm_p_lt_0.05']}/{verdict['n_tasks']} tasks have permutation p < 0.05 for "
             f"distractor-identity x outcome independence.\n\n{verdict['reading']}\n")
    L.append("\n| task | target | distractors | modal outcome per distractor | distinct modes | chi2 | perm p |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for t in tasks:
        v = within[str(t)]
        tgt = next(e["task_target_object"] for e in eps if e["task_id"] == t)
        modes = " ".join(f"{w}:{m}" for w, m in v["modal_outcome_per_distractor"].items())
        L.append(f"| t{t} | {tgt} | {v['n_distractors']} | {modes} | {','.join(v['distinct_modes'])} | "
                 f"{v['chi2']} | {v['perm_p']:.4g} |\n")

    L.append("\n## Task x distractor modal-outcome matrix\n\n")
    L.append("| task \\ wrong task | " + " | ".join(f"w{w}" for w in ws) + " |\n")
    L.append("|---" * (len(ws) + 1) + "|\n")
    for t in tasks:
        cells_row = []
        for w in ws:
            c = cell_stats.get(f"{t}|{w}")
            cells_row.append(f"{c['modal_outcome'][0]}{int(round(c['modal_share'] * 100))}" if c else ".")
        L.append(f"| t{t} | " + " | ".join(cells_row) + " |\n")
    L.append("\n(cell = first letter of the modal outcome + modal share %; `.` = not a valid distractor.)\n")

    L.append("\n## Per-cell three-way proportions (Wilson 95% CI)\n\n")
    L.append("| task | wrong task | prompted object | n | OBEY | IGNORE | JAM | success |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for k in sorted(cell_stats, key=lambda k: (cell_stats[k]["task_id"], cell_stats[k]["wrong_task_id"])):
        c = cell_stats[k]
        def f(o):
            d = c["outcomes"][o]
            return f"{d['p']*100:.0f} [{d['ci95'][0]*100:.0f},{d['ci95'][1]*100:.0f}]"
        L.append(f"| t{c['task_id']} | w{c['wrong_task_id']} | {c['prompted_object']} | {c['n']} | "
                 f"{f('OBEY')} | {f('IGNORE')} | {f('JAM')} | {c['success_rate']*100:.0f} |\n")

    L.append("\n## Same prompted object across different host tasks\n\n")
    L.append("| prompted object | host tasks | modal outcome per host task | distinct modes | perm p |\n")
    L.append("|---|---|---|---|---|\n")
    for o, v in obj_stats.items():
        L.append(f"| {o} | {v['n_host_tasks']} | "
                 f"{' '.join(f't{t}:{m}' for t, m in v['modal_outcome_per_host_task'].items())} | "
                 f"{','.join(v['distinct_modes'])} | {v['perm_p']:.4g} |\n")

    L.append("\n## Reviewer-flagged pair: tasks 2 and 3 both prompted with `ketchup_1`\n\n")
    for k in ("task_2", "task_3"):
        if k in ket:
            v = ket[k]
            L.append(f"- **{k}** (own target {v['task_target_object']}, wrong prompt from task {v['wrong_task_id']}: "
                     f"\"{v['prompt']}\"): n={v['n']}, {v['counts']}, modal = **{v['modal_outcome']}**, "
                     f"success {v['success_rate']*100:.0f}%, first-touch objects {v['first_touch_objects']}\n")
    L.append(f"- Opposite modes: **{ket.get('opposite_modes')}**\n")

    L.append("\n## Reference conditions (Stage 0, reused)\n\n| task | correct | null |\n|---|---|---|\n")
    for t in tasks:
        c = ref_stats.get(f"{t}|correct"); n = ref_stats.get(f"{t}|null")
        L.append(f"| t{t} | {c['success_rate']*100:.0f}% (n={c['n']}) | {n['success_rate']*100:.0f}% (n={n['n']}) |\n"
                 if c and n else f"| t{t} | - | - |\n")

    (outdir / "summary.md").write_text("".join(L))
    print("".join(L[:40]))
    print(f"[analyze] wrote {outdir}/results.json, per_episode.jsonl, summary.md "
          f"({len(eps)} episodes, {len(cell_stats)} cells)")


if __name__ == "__main__":
    main()
