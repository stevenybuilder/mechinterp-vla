#!/usr/bin/env python
"""Behavior-blind metadata audit for position-matched LIBERO Object pairs.

This script tokenizes prompts and inspects simulator object membership.  It does
not call the prefix transformer, action expert, or policy action sampler.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from hooks import Pi05Harness
from lerobot.envs.libero import _get_suite
from pi05_attention_resolution import OBJECT_TASKS, make_env, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--revision", default="8e174154ef5f6c60a8da12ae99c303d8963138c1")
    parser.add_argument("--init", type=int, default=25)
    args = parser.parse_args()

    suite_name = "libero_object"
    suite = _get_suite(suite_name)
    harness = Pi05Harness(args.policy, revision=args.revision, dtype="float32")

    # Use one observation for tokenizer metadata. Within each experiment cell,
    # the observation is identical between its A/B prompts, so only the task
    # string can change the relevant token positions.
    anchor_env = make_env(suite, suite_name, 0)
    try:
        anchor_env.init_state_id = args.init
        observation, _ = anchor_env.reset(seed=1000 + args.init)
        task_meta = {}
        for task_id, task in enumerate(suite.tasks):
            object_id, object_name = OBJECT_TASKS[task_id]
            batch = harness.build_batch(observation, task.language)
            segment, decoded = harness.segments(batch, object_name=object_name, n_img=768)
            task_meta[str(task_id)] = {
                "language": task.language,
                "object_id": object_id,
                "instruction_positions": segment["INSTR"],
                "text_valid_positions": segment["TEXT_VALID"],
                "decoded_instruction": decoded.get("INSTR", ""),
            }
    finally:
        anchor_env.close()

    membership = {}
    for task_id in range(len(suite.tasks)):
        env = make_env(suite, suite_name, task_id)
        try:
            env.init_state_id = args.init
            env.reset(seed=1000 + args.init)
            membership[str(task_id)] = sorted(env._env.env.obj_body_id)
        finally:
            env.close()

    eligible = []
    for task_a in range(len(suite.tasks)):
        for task_b in range(task_a + 1, len(suite.tasks)):
            meta_a, meta_b = task_meta[str(task_a)], task_meta[str(task_b)]
            position_matched = (
                meta_a["instruction_positions"] == meta_b["instruction_positions"]
                and meta_a["text_valid_positions"] == meta_b["text_valid_positions"]
            )
            object_a = meta_a["object_id"]
            object_b = meta_b["object_id"]
            copresent_both_directions = (
                object_b in membership[str(task_a)]
                and object_a in membership[str(task_b)]
            )
            if position_matched and copresent_both_directions:
                eligible.append([task_a, task_b])

    write_json(args.out, {
        "schema_version": 1,
        "behavior_blind": True,
        "policy_forward_calls": 0,
        "action_forward_calls": 0,
        "suite": suite_name,
        "init_id_for_metadata": args.init,
        "tasks": task_meta,
        "object_membership": membership,
        "eligible_undirected_pairs": eligible,
    })
    print(json.dumps({"eligible_undirected_pairs": eligible}, indent=2))


if __name__ == "__main__":
    main()
