#!/usr/bin/env python
"""Preregistered π0.5 prefill-time instruction mediation experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from hooks import LAYER_BANDS, Pi05Harness, axis_metric, cache_kv_lists, clone_cache
from lerobot.envs.libero import LiberoEnv, _get_suite
from stage2_discovery import GOAL_OBJ


PAIRS = ((2, 5, 3), (0, 6, 9), (9, 6, 0))
CONDITIONS = (
    "clean_dst",
    "clean_src",
    "identity_dst",
    "prefill_instr_src",
    "prefill_restore_instr_dst",
    "prefill_restore_noninstr_dst",
    "prefill_restore_img_dst",
    "prefill_restore_randimg_dst",
    "postcache_instr_src",
)


def normalized_distances(chunk: np.ndarray, dst: np.ndarray, src: np.ndarray, steps: int) -> tuple[float, float]:
    denominator = float(np.linalg.norm(src[:steps] - dst[:steps]))
    if denominator <= 1e-8:
        raise AssertionError(f"degenerate clean action contrast: {denominator}")
    return (
        float(np.linalg.norm(chunk[:steps] - dst[:steps]) / denominator),
        float(np.linalg.norm(chunk[:steps] - src[:steps]) / denominator),
    )


def cache_distances(prefix, dst, src, image_positions: list[int]) -> list[dict]:
    kp, vp = cache_kv_lists(prefix["cache"])
    kd, vd = cache_kv_lists(dst["cache"])
    ks, vs = cache_kv_lists(src["cache"])
    rows = []
    for layer in range(len(kp)):
        p = torch.cat((kp[layer][:, :, image_positions].float().flatten(), vp[layer][:, :, image_positions].float().flatten()))
        d = torch.cat((kd[layer][:, :, image_positions].float().flatten(), vd[layer][:, :, image_positions].float().flatten()))
        s = torch.cat((ks[layer][:, :, image_positions].float().flatten(), vs[layer][:, :, image_positions].float().flatten()))
        denominator = torch.linalg.vector_norm(s - d).item()
        rows.append(
            {
                "layer": layer,
                "D_dst": float(torch.linalg.vector_norm(p - d).item() / denominator) if denominator else 0.0,
                "D_src": float(torch.linalg.vector_norm(p - s).item() / denominator) if denominator else 0.0,
            }
        )
    return rows


def action_chunk(harness: Pi05Harness, prefix, noise) -> np.ndarray:
    chunk, _ = harness.action_forward(prefix, noise)
    return harness.unnormalize(chunk)[0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ids", default="25-49")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--revision", default="8e174154ef5f6c60a8da12ae99c303d8963138c1")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    start_init, end_init = (int(x) for x in args.init_ids.split("-", 1))
    init_ids = list(range(start_init, end_init + 1))
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                row = json.loads(line)
                done.add((row["pair"], row["init"], row["direction"]))
            except (json.JSONDecodeError, KeyError):
                continue

    harness = Pi05Harness(
        args.policy,
        revision=args.revision,
        dtype=args.dtype,
        processor_path=args.processor_path,
    )
    suite = _get_suite("libero_goal")
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    manifest = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pairs": list(PAIRS),
        "conditions": list(CONDITIONS),
        "module_paths": harness.module_paths,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with rows_path.open("a") as output:
        for task_a, task_b, _ in PAIRS:
            pair = f"libero_goal_t{task_a}_vs_t{task_b}"
            env = LiberoEnv(
                task_suite=suite,
                task_id=task_a,
                task_suite_name="libero_goal",
                obs_type="pixels_agent_pos",
                observation_height=360,
                observation_width=360,
                init_states=True,
                episode_index=0,
                n_envs=1,
            )
            raw_env = env._env.env
            obj_a = GOAL_OBJ[task_a][0]
            obj_b = GOAL_OBJ[task_b][0]
            if obj_a not in raw_env.obj_body_id or obj_b not in raw_env.obj_body_id:
                raise AssertionError((pair, obj_a, obj_b, list(raw_env.obj_body_id)))
            for init_id in init_ids:
                torch.manual_seed(20260831 + task_a * 1000 + init_id)
                env.init_state_id = init_id
                observation, _ = env.reset(seed=20260831 + init_id)
                position_a = np.asarray(raw_env.sim.data.body_xpos[raw_env.obj_body_id[obj_a]], dtype=np.float64)
                position_b = np.asarray(raw_env.sim.data.body_xpos[raw_env.obj_body_id[obj_b]], dtype=np.float64)
                batch_a = harness.build_batch(observation, prompts[task_a])
                batch_b = harness.build_batch(observation, prompts[task_b])
                emb_a, _, _, n_img_slots, n_img_valid = harness._prefix_embed(batch_a)
                emb_b, _, _, n_img_slots_b, n_img_valid_b = harness._prefix_embed(batch_b)
                if n_img_slots != n_img_slots_b:
                    raise AssertionError((n_img_slots, n_img_slots_b))
                if n_img_valid != n_img_valid_b:
                    raise AssertionError((n_img_valid, n_img_valid_b))
                seg_a, _ = harness.segments(batch_a, object_name=GOAL_OBJ[task_a][1], n_img=n_img_slots)
                seg_b, _ = harness.segments(batch_b, object_name=GOAL_OBJ[task_b][1], n_img=n_img_slots)
                if seg_a["INSTR"] != seg_b["INSTR"]:
                    raise AssertionError(f"instruction positions differ: {seg_a['INSTR']} != {seg_b['INSTR']}")
                if seg_a["TEXT_VALID"] != seg_b["TEXT_VALID"]:
                    raise AssertionError("valid text positions differ")
                instruction_positions = seg_a["INSTR"]
                valid_positions = list(range(n_img_valid)) + seg_a["TEXT_VALID"]
                noninstruction_positions = sorted(set(valid_positions) - set(instruction_positions))
                image_positions = list(range(n_img_valid))

                if not torch.equal(emb_a[:, noninstruction_positions], emb_b[:, noninstruction_positions]):
                    maximum = float(
                        (emb_a[:, noninstruction_positions].float() - emb_b[:, noninstruction_positions].float())
                        .abs()
                        .max()
                        .item()
                    )
                    raise AssertionError(f"non-instruction prefix embeddings differ, maxabs={maximum}")
                mixed = emb_a.clone()
                mixed[:, instruction_positions] = emb_b[:, instruction_positions]
                if not torch.equal(mixed, emb_b):
                    maximum = float((mixed.float() - emb_b.float()).abs().max().item())
                    raise AssertionError(f"prefill intervention is not input-identical to donor, maxabs={maximum}")

                prefix_a = harness.prefix_forward(batch_a)
                prefix_b = harness.prefix_forward(batch_b)
                prefix_mixed = harness.prefix_forward(batch_a, prefix_embs_override=mixed)
                for direction, dst, src, dst_batch, src_batch, dst_seg, src_seg in (
                    ("A<-B", prefix_a, prefix_b, batch_a, batch_b, seg_a, seg_b),
                    ("B<-A", prefix_b, prefix_a, batch_b, batch_a, seg_b, seg_a),
                ):
                    if (pair, init_id, direction) in done:
                        continue
                    if direction == "A<-B":
                        prefill = prefix_mixed
                    else:
                        reverse_mixed = emb_b.clone()
                        reverse_mixed[:, instruction_positions] = emb_a[:, instruction_positions]
                        if not torch.equal(reverse_mixed, emb_a):
                            raise AssertionError("reverse prefill intervention is not input-identical to donor")
                        prefill = harness.prefix_forward(batch_b, prefix_embs_override=reverse_mixed)

                    noise = harness.make_noise(seed=init_id * 100)
                    conditions = {
                        "clean_dst": dst,
                        "clean_src": src,
                        "identity_dst": {**dst, "cache": clone_cache(dst["cache"])},
                        "prefill_instr_src": prefill,
                    }

                    restored_instr = clone_cache(prefill["cache"])
                    harness.kv_swap(restored_instr, dst["cache"], instruction_positions, LAYER_BANDS["all"])
                    conditions["prefill_restore_instr_dst"] = {**prefill, "cache": restored_instr}

                    restored_noninstr = clone_cache(prefill["cache"])
                    harness.kv_swap(restored_noninstr, dst["cache"], noninstruction_positions, LAYER_BANDS["all"])
                    conditions["prefill_restore_noninstr_dst"] = {**prefill, "cache": restored_noninstr}

                    restored_img = clone_cache(prefill["cache"])
                    harness.kv_swap(restored_img, dst["cache"], image_positions, LAYER_BANDS["all"])
                    conditions["prefill_restore_img_dst"] = {**prefill, "cache": restored_img}

                    generator = np.random.default_rng(20260831 + task_a * 100_000 + init_id)
                    random_image_positions = sorted(
                        generator.choice(image_positions, size=len(instruction_positions), replace=False).tolist()
                    )
                    restored_randimg = clone_cache(prefill["cache"])
                    harness.kv_swap(
                        restored_randimg, dst["cache"], random_image_positions, LAYER_BANDS["all"]
                    )
                    conditions["prefill_restore_randimg_dst"] = {**prefill, "cache": restored_randimg}

                    postcache_instr = clone_cache(dst["cache"])
                    harness.kv_swap(postcache_instr, src["cache"], instruction_positions, LAYER_BANDS["all"])
                    conditions["postcache_instr_src"] = {**dst, "cache": postcache_instr}

                    chunks = {name: action_chunk(harness, prefix, noise) for name, prefix in conditions.items()}
                    if not np.array_equal(chunks["clean_dst"], chunks["identity_dst"]):
                        maximum = float(np.abs(chunks["clean_dst"] - chunks["identity_dst"]).max())
                        raise AssertionError(f"identity cache clone changed output, maxabs={maximum}")
                    chunk_dst, chunk_src = chunks["clean_dst"], chunks["clean_src"]
                    cache_image_distance = cache_distances(prefill, dst, src, image_positions)
                    for name, chunk in chunks.items():
                        d_dst_10, d_src_10 = normalized_distances(chunk, chunk_dst, chunk_src, 10)
                        d_dst_50, d_src_50 = normalized_distances(chunk, chunk_dst, chunk_src, 50)
                        row = {
                            "pair": pair,
                            "init": init_id,
                            "direction": direction,
                            "condition": name,
                            "D_dst_10": d_dst_10,
                            "D_src_10": d_src_10,
                            "D_dst_50": d_dst_50,
                            "D_src_50": d_src_50,
                            "axis10": axis_metric(chunk, position_a, position_b, 10),
                            "axis50": axis_metric(chunk, position_a, position_b, 50),
                            "chunk10": chunk[:10].round(6).tolist(),
                            "instruction_positions": instruction_positions,
                            "random_image_positions": random_image_positions,
                            "prefill_image_cache_distances": cache_image_distance
                            if name == "prefill_instr_src"
                            else None,
                        }
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    output.flush()
                    print(
                        f"[prefill] {pair} init={init_id} direction={direction} "
                        f"prefill_Dsrc={normalized_distances(chunks['prefill_instr_src'], chunk_dst, chunk_src, 10)[1]:.4f} "
                        f"restore_instr_Dsrc={normalized_distances(chunks['prefill_restore_instr_dst'], chunk_dst, chunk_src, 10)[1]:.4f} "
                        f"restore_noninstr_Ddst={normalized_distances(chunks['prefill_restore_noninstr_dst'], chunk_dst, chunk_src, 10)[0]:.4f}",
                        flush=True,
                    )
            env.close()
    print("[prefill] DONE", flush=True)


if __name__ == "__main__":
    main()
