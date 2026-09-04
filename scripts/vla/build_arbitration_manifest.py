#!/usr/bin/env python
"""Build the cross-distractor arbitration manifest.

For every LIBERO task, enumerate the objects actually present in the scene (as recorded by the runner's
ContactTracker, i.e. the live MuJoCo object list) and list every OTHER task whose target object is present in
this scene. Those are the valid "wrong" instructions for that task, so that mode (OBEY / IGNORE / JAM) can vary
WITHIN a task across distractors rather than being a per-task constant.

Input : a Stage-0 per-episode JSONL (any run that recorded `scene_objects` and `target_object`).
Output: JSON manifest {suite: {task_id: {...}}}.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_episode", required=True, help="stage0 per_episode.jsonl (source of scene_objects)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="libero_object,libero_goal")
    args = ap.parse_args()

    scenes: dict[str, dict[int, list[str]]] = {}
    targets: dict[str, dict[int, str]] = {}
    prompts: dict[str, dict[int, str]] = {}
    names: dict[str, dict[int, str]] = {}
    for line in Path(args.per_episode).open():
        r = json.loads(line)
        s, t = r["suite"], r["task_id"]
        scenes.setdefault(s, {}).setdefault(t, r["scene_objects"])
        if r.get("target_object"):
            targets.setdefault(s, {})[t] = r["target_object"]
        if r["condition"] == "correct":
            prompts.setdefault(s, {})[t] = r["prompt"]
        names.setdefault(s, {})[t] = r["task_name"]

    manifest = OrderedDict()
    for suite in args.suites.split(","):
        if suite not in scenes:
            continue
        tasks = OrderedDict()
        for t in sorted(scenes[suite]):
            objs = scenes[suite][t]
            valid = []
            for j in sorted(targets.get(suite, {})):
                if j == t:
                    continue
                tgt = targets[suite][j]
                if tgt in objs and tgt != targets[suite].get(t):
                    valid.append({"wrong_task_id": j, "wrong_object": tgt,
                                  "wrong_prompt": prompts[suite][j], "condition": f"wrong:{j}"})
            tasks[str(t)] = {
                "task_id": t, "task_name": names[suite][t], "prompt": prompts[suite][t],
                "target_object": targets[suite].get(t), "scene_objects": objs,
                "n_valid_distractors": len(valid), "distractors": valid,
            }
        manifest[suite] = {
            "n_tasks": len(tasks),
            "n_cells": sum(v["n_valid_distractors"] for v in tasks.values()),
            "tasks": tasks,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, Path(args.out).open("w"), indent=1)
    for suite, m in manifest.items():
        print(f"{suite}: {m['n_tasks']} tasks, {m['n_cells']} (task x wrong-instruction) cells")
        for k, v in m["tasks"].items():
            print(f"  t{k} target={v['target_object']:<24} distractors="
                  f"{[d['wrong_task_id'] for d in v['distractors']]}")


if __name__ == "__main__":
    main()
