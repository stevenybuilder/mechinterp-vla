#!/usr/bin/env python3
"""Preregistered OpenVLA-OFT downstream-blocked projected-K/V experiment.

Run this file from the openvla-oft repository root.  It uses the released combined OFT checkpoint and official
LIBERO-goal task definitions/init states.  The primary source and destination forwards always share one physical
observation.  Only projected K/V rows are replaced; Q, residuals (except the named single-layer residual control),
pixels, proprio, instruction rows, action rows, and the action head are untouched.

The cross-scene condition is a frozen trajectory-copying diagnostic, not a primary condition: for init i it uses
source-prompt image K/V from the next requested init, (i+1) modulo the requested init panel, and reports distance to
both the current-scene source action and the cross-scene donor action.

``--smoke`` runs synthetic hook/metric tests and does not load a checkpoint or LIBERO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
LEGACY_PAIR_MAP = {0: 2, 1: 9, 2: 1, 3: 7, 4: 5, 5: 4, 6: 0, 7: 3, 8: 2, 9: 8}
# Frozen before outcomes.  Runtime assertions verify equal tokenizer and full-prompt lengths.  Seven of the ten
# official GOAL tasks have a same-length partner under the released checkpoint tokenizer.
EQUAL_TOKEN_PAIR_MAP = {0: 1, 1: 0, 2: 5, 4: 9, 5: 6, 6: 2, 9: 4}
# Three official, outcome-independent prompt donors per panel/task.  Donors are ordered by: third target object,
# same tokenizer length as A when available, then official task id.  The literal map is frozen so a library change
# cannot silently choose a different control after execution begins.
FROZEN_RESAMPLE_TASKS = {
    "legacy": {
        0: (1, 4, 5), 1: (0, 3, 5), 2: (5, 6, 0), 3: (1, 2, 4), 4: (9, 0, 2),
        5: (2, 6, 0), 6: (2, 5, 1), 7: (1, 2, 4), 8: (0, 3, 5), 9: (0, 3, 5),
    },
    "equal": {
        0: (2, 5, 6), 1: (2, 5, 6), 2: (6, 0, 1), 4: (0, 3, 5),
        5: (2, 0, 1), 6: (5, 0, 1), 9: (0, 3, 5),
    },
}
RESAMPLE_CONDITIONS = tuple(f"kv_img_8_31_resample_c{i}" for i in range(1, 4))
PRIMARY_CONDITIONS = (
    "single_resid_l8",
    "kv_img_8_31",
    "kv_img_0_7",
    "kv_img_8_31_self",
    "kv_img_8_31_random",
    *RESAMPLE_CONDITIONS,
    "kv_img_8_31_xscene",
)
EQUAL_ONLY_CONDITIONS = ("kv_instr_8_31", "kv_both_8_31")
PARAPHRASE_CONDITION = "kv_img_8_31_paraphrase"
MIN_TASK_CLEAN_FRACTION = 0.80
EPS = 1e-12


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(p) for p in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def parse_ints(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("empty integer selection")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate integer in {spec!r}")
    return out


def normalized_metrics(patched, destination, source, floor: float = 1e-4) -> dict:
    """Primary full-action metrics, using environment-unit flattened chunks."""
    import numpy as np

    p = np.asarray(patched, dtype=np.float64).ravel()
    d = np.asarray(destination, dtype=np.float64).ravel()
    s = np.asarray(source, dtype=np.float64).ravel()
    l2_sd = float(np.linalg.norm(s - d))
    l2_src = float(np.linalg.norm(p - s))
    l2_dst = float(np.linalg.norm(p - d))
    negligible = l2_sd < floor
    den = float(np.dot(s - d, s - d))
    return {
        "l2_to_src": l2_src,
        "l2_to_dst": l2_dst,
        "l2_src_dst": l2_sd,
        "negligible_clean_contrast": negligible,
        "D_src": None if negligible else l2_src / l2_sd,
        "D_dst": None if negligible else l2_dst / l2_sd,
        "R_full": None if den < floor * floor else float(np.dot(p - d, s - d) / den),
    }


def empty_audit() -> dict:
    return {
        "calls": 0,
        "positions_written": 0,
        "expected_nonzero_calls": 0,
        "nonzero_write_calls": 0,
        "expected_delta_l2_sq": 0.0,
        "written_delta_l2_sq": 0.0,
        "max_abs_write": 0.0,
        "max_pre_hook_vs_clean_dst": 0.0,
        "max_random_norm_relative_error": 0.0,
    }


def finish_audit(audit: dict) -> dict:
    out = dict(audit)
    out["expected_delta_l2"] = math.sqrt(out.pop("expected_delta_l2_sq"))
    out["written_delta_l2"] = math.sqrt(out.pop("written_delta_l2_sq"))
    out["all_expected_nonzero_writes_occurred"] = (
        out["nonzero_write_calls"] == out["expected_nonzero_calls"]
    )
    return out


def capture_projection_hooks(layers: list[int], store: dict) -> dict:
    """Callbacks for OFTAdapter.forward(projection_hooks=...)."""
    hooks = {}
    for layer in layers:
        hooks[layer] = {}
        for kind in ("k", "v"):
            def capture(li, k, projected, _input, store=store):
                store[(li, k)] = projected.detach().clone()
                return None
            hooks[layer][kind] = capture
    return hooks


def projection_patch_hooks(
    layers: list[int],
    dst_positions: list[int],
    src_positions: list[int],
    donor_store: dict,
    clean_dst_store: dict,
    mode: str,
    audit: dict,
    seed: int,
) -> dict:
    """Build projected K/V hooks.  ``mode`` is donor, self, or equal-per-row-norm random."""
    if mode not in ("donor", "self", "random"):
        raise ValueError(mode)
    if len(dst_positions) != len(src_positions):
        raise ValueError(f"source/destination position count differs: {len(src_positions)} vs {len(dst_positions)}")
    hooks = {}
    for layer in layers:
        hooks[layer] = {}
        for kind in ("k", "v"):
            def patch(li, k, projected, _input, layer=layer, kind=kind):
                import torch

                dp = torch.as_tensor(dst_positions, device=projected.device, dtype=torch.long)
                sp = torch.as_tensor(src_positions, device=projected.device, dtype=torch.long)
                clean_dst = clean_dst_store[(layer, kind)][0, dp].to(projected.device)
                clean_donor = donor_store[(layer, kind)][0, sp].to(projected.device)
                current = projected[0, dp]
                pre_error = float((current - clean_dst).float().abs().max().item())
                expected_delta = (clean_donor - clean_dst).float()
                if mode == "self":
                    replacement = clean_dst
                    expected_for_gate = torch.zeros_like(expected_delta)
                elif mode == "donor":
                    replacement = clean_donor
                    expected_for_gate = expected_delta
                else:
                    gen = torch.Generator(device=projected.device)
                    gen.manual_seed(stable_seed(seed, layer, kind))
                    random_dir = torch.randn(
                        expected_delta.shape, generator=gen, device=projected.device, dtype=torch.float32
                    )
                    random_dir /= random_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    target_norm = expected_delta.norm(dim=-1, keepdim=True)
                    random_delta = random_dir * target_norm
                    replacement = (clean_dst.float() + random_delta).to(clean_dst.dtype)
                    expected_for_gate = expected_delta
                    written_norm = (replacement - clean_dst).float().norm(dim=-1)
                    target = target_norm[:, 0]
                    nz = target > 1e-10
                    if bool(nz.any()):
                        rel = ((written_norm[nz] - target[nz]).abs() / target[nz]).max().item()
                        audit["max_random_norm_relative_error"] = max(
                            audit["max_random_norm_relative_error"], float(rel)
                        )
                edited = projected.clone()
                edited[0, dp] = replacement
                write = (replacement - current).float()
                expected_norm = float(expected_for_gate.norm().item())
                write_norm = float(write.norm().item())
                audit["calls"] += 1
                audit["positions_written"] += len(dst_positions)
                audit["expected_delta_l2_sq"] += expected_norm**2
                audit["written_delta_l2_sq"] += write_norm**2
                audit["max_abs_write"] = max(
                    audit["max_abs_write"], float(write.abs().max().item()) if write.numel() else 0.0
                )
                audit["max_pre_hook_vs_clean_dst"] = max(audit["max_pre_hook_vs_clean_dst"], pre_error)
                if expected_norm > 1e-10:
                    audit["expected_nonzero_calls"] += 1
                    if write_norm > 1e-10:
                        audit["nonzero_write_calls"] += 1
                return edited
            hooks[layer][kind] = patch
    return hooks


def residual_patch_hook(dst_positions, src_positions, donor_hidden, clean_dst_hidden, audit):
    """Frozen single residual-layer control, patching layer-8 output at image positions."""
    def patch(hidden):
        import torch

        dp = torch.as_tensor(dst_positions, device=hidden.device, dtype=torch.long)
        sp = torch.as_tensor(src_positions, device=hidden.device, dtype=torch.long)
        clean_dst = clean_dst_hidden[0, dp].to(hidden.device)
        donor = donor_hidden[0, sp].to(hidden.device)
        current = hidden[0, dp]
        write = (donor - current).float()
        expected = (donor - clean_dst).float()
        audit["calls"] += 1
        audit["positions_written"] += len(dst_positions)
        audit["expected_delta_l2_sq"] += float(expected.norm().item()) ** 2
        audit["written_delta_l2_sq"] += float(write.norm().item()) ** 2
        audit["max_abs_write"] = max(audit["max_abs_write"], float(write.abs().max().item()))
        audit["max_pre_hook_vs_clean_dst"] = max(
            audit["max_pre_hook_vs_clean_dst"], float((current - clean_dst).float().abs().max().item())
        )
        if float(expected.norm().item()) > 1e-10:
            audit["expected_nonzero_calls"] += 1
            if float(write.norm().item()) > 1e-10:
                audit["nonzero_write_calls"] += 1
        out = hidden.clone()
        out[0, dp] = donor
        return out
    return patch


def frozen_paraphrase(label: str) -> str | None:
    """The sole frozen behavior-preserving rewrite; eligibility is checked with the checkpoint tokenizer."""
    words = label.split()
    try:
        index = words.index("put")
    except ValueError:
        return None
    words[index] = "place"
    return " ".join(words)


def axis_fields(patched, dst, src, u):
    import numpy as np

    if u is None:
        return {"m_axis": None, "m_axis_dst": None, "m_axis_src": None, "R_axis": None}
    p, d, s = (np.asarray(x, dtype=np.float64) for x in (patched, dst, src))
    m_p = float(np.sum(p[:, :3] @ u))
    m_d = float(np.sum(d[:, :3] @ u))
    m_s = float(np.sum(s[:, :3] @ u))
    return {
        "m_axis": m_p,
        "m_axis_dst": m_d,
        "m_axis_src": m_s,
        "R_axis": None if abs(m_s - m_d) <= 1e-12 else (m_p - m_d) / (m_s - m_d),
    }


def run_smoke() -> None:
    """Synthetic no-checkpoint validation of capture/edit hooks, controls, metrics, and frozen maps."""
    import numpy as np
    import torch

    assert len(LEGACY_PAIR_MAP) == 10 and len(EQUAL_TOKEN_PAIR_MAP) == 7
    assert all(len(v) == 3 for panel in FROZEN_RESAMPLE_TASKS.values() for v in panel.values())
    assert frozen_paraphrase("put the bowl on the stove") == "place the bowl on the stove"
    assert frozen_paraphrase("open the drawer") is None
    torch.manual_seed(1)
    layers = [0, 1]
    dst = {(l, k): torch.randn(1, 8, 5) for l in layers for k in ("k", "v")}
    donor = {(l, k): dst[(l, k)] + torch.randn(1, 8, 5) for l in layers for k in ("k", "v")}

    for mode in ("donor", "self", "random"):
        audit = empty_audit()
        hooks = projection_patch_hooks(layers, [1, 2, 3], [1, 2, 3], donor, dst, mode, audit, seed=17)
        for layer in layers:
            for kind in ("k", "v"):
                original = dst[(layer, kind)].clone()
                edited = hooks[layer][kind](layer, kind, original.clone(), None)
                assert torch.equal(edited[:, 0], original[:, 0])
                assert torch.equal(edited[:, 4:], original[:, 4:])
                if mode == "self":
                    assert torch.equal(edited, original)
                elif mode == "donor":
                    assert torch.equal(edited[0, 1:4], donor[(layer, kind)][0, 1:4])
                else:
                    got = (edited[0, 1:4] - original[0, 1:4]).float().norm(dim=-1)
                    want = (donor[(layer, kind)][0, 1:4] - original[0, 1:4]).float().norm(dim=-1)
                    assert torch.allclose(got, want, rtol=2e-5, atol=2e-5)
        done = finish_audit(audit)
        assert done["calls"] == 4
        assert done["all_expected_nonzero_writes_occurred"]

    # The equal-panel positive control patches two disjoint roles in one hook without touching any other row.
    audit = empty_audit()
    hooks = projection_patch_hooks(layers, [0, 1, 2, 6], [0, 1, 2, 6], donor, dst, "donor", audit, seed=31)
    edited = hooks[0]["k"](0, "k", dst[(0, "k")].clone(), None)
    assert torch.equal(edited[0, [0, 1, 2, 6]], donor[(0, "k")][0, [0, 1, 2, 6]])
    assert torch.equal(edited[0, [3, 4, 5, 7]], dst[(0, "k")][0, [3, 4, 5, 7]])

    d = np.zeros((2, 2))
    s = np.ones((2, 2))
    assert normalized_metrics(s, d, s)["D_src"] == 0.0
    assert normalized_metrics(d, d, s)["D_dst"] == 0.0
    assert normalized_metrics(d, d, d)["negligible_clean_contrast"]
    print("OFT downstream-K/V synthetic smoke: PASS")


def run_experiment(args) -> None:
    import numpy as np
    import torch

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, os.getcwd())
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    from oft_stage2 import OFTAdapter
    from run_oft_prompt_conditions import GOAL_TARGET_OBJECT, ContactTracker

    if args.checkpoint != CHECKPOINT and not args.allow_nonfrozen_checkpoint:
        raise ValueError(f"frozen checkpoint is {CHECKPOINT!r}; got {args.checkpoint!r}")
    panels = [x.strip() for x in args.panels.split(",") if x.strip()]
    if any(p not in ("legacy", "equal") for p in panels):
        raise ValueError("--panels must contain legacy and/or equal")
    init_ids = parse_ints(args.init_ids)
    if len(init_ids) < 2:
        raise ValueError("at least two init states are required for the frozen cross-scene diagnostic")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path, clean_path, units_path = out / "rows.jsonl", out / "clean_chunks.jsonl", out / "units.jsonl"
    prior_manifest_path = out / "manifest.json"
    prior_manifest = {}
    if prior_manifest_path.exists():
        prior_manifest = json.loads(prior_manifest_path.read_text())
        if prior_manifest.get("schema") != "oft_downstream_kv_v2":
            raise RuntimeError(
                f"refusing to mix amended v2 rows with {prior_manifest.get('schema')!r} in {out}; use a fresh output"
            )
    existing_rows = set()
    if rows_path.exists():
        for line in rows_path.open():
            rec = json.loads(line)
            key = (rec["panel"], rec["task"], rec["init"], rec["direction"], rec["condition"])
            if key in existing_rows:
                raise RuntimeError(f"duplicate pre-existing intervention row {key}; preserve it for audit and repair explicitly")
            existing_rows.add(key)
    existing_clean = set()
    if clean_path.exists():
        for line in clean_path.open():
            rec = json.loads(line)
            key = (rec["panel"], rec["task"], rec["init"], rec["prompt_key"])
            if key in existing_clean:
                raise RuntimeError(f"duplicate pre-existing clean row {key}; preserve it for audit and repair explicitly")
            existing_clean.add(key)
    completed = set()
    if units_path.exists():
        for line in units_path.open():
            rec = json.loads(line)
            completed.add((rec["panel"], rec["task"], rec["init"]))
    rows_f = rows_path.open("a")
    clean_f = clean_path.open("a")
    units_f = units_path.open("a")

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name="libero_goal",
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        center_crop=True,
        seed=args.seed,
    )
    model, action_head, proprio_projector, _n, processor = initialize_model(cfg)
    adapter = OFTAdapter(cfg, model, action_head, proprio_projector, processor)
    if adapter.n_layers != 32:
        raise AssertionError(f"preregistered layers 8--31 require the released 32-layer model, got {adapter.n_layers}")
    all_layers, early_layers, live_layers = list(range(32)), list(range(8)), list(range(8, 32))
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    official_tasks = list(range(suite.n_tasks))
    if official_tasks != list(range(10)):
        raise AssertionError(f"expected the official ten-task LIBERO-goal suite, got {official_tasks}")

    def tok_count(task_id):
        return len(adapter.tok(suite.get_task(task_id).language.lower(), add_special_tokens=False)["input_ids"])

    target_objects = {t: GOAL_TARGET_OBJECT[t] for t in official_tasks}
    panel_maps = {"legacy": LEGACY_PAIR_MAP, "equal": EQUAL_TOKEN_PAIR_MAP}
    manifest = {
        "schema": "oft_downstream_kv_v2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": args.checkpoint,
        "suite": "libero_goal",
        "official_data": True,
        "panels": panels,
        "init_ids": init_ids,
        "held_out_default_used": init_ids == list(range(25, 35)),
        "timepoint": 0,
        "seed": args.seed,
        "legacy_pair_map": LEGACY_PAIR_MAP,
        "equal_token_pair_map": EQUAL_TOKEN_PAIR_MAP,
        "frozen_resample_tasks": FROZEN_RESAMPLE_TASKS,
        "conditions": {
            "primary": PRIMARY_CONDITIONS,
            "equal_only": EQUAL_ONLY_CONDITIONS,
            "equal_directional_when_token_matched": PARAPHRASE_CONDITION,
        },
        "n_layers": adapter.n_layers,
        "n_patches": adapter.n_patches,
        "projected_hook_contract": "patch k_proj/v_proj outputs before reshape/RoPE; q_proj untouched",
        "primary_observation_contract": "source and destination prompts use identical task/init/timepoint-0 pixels and proprio",
        "cross_scene_contract": "diagnostic only; source prompt K/V from next requested init modulo panel",
        "normalization_floor": 1e-4,
        "minimum_task_clean_separation_fraction": MIN_TASK_CLEAN_FRACTION,
        "inference_unit": "task; median over paired init and direction before testing",
        "pair_metadata": prior_manifest.get("pair_metadata", {}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def get_initial_observation(env, init_state):
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        return prepare_observation(obs, resize_size)[0]

    def clean_forward(build, capture=True, keep_resid_l8=False):
        kv = {}
        hooks = capture_projection_hooks(all_layers, kv) if capture else None
        norm, actions, hidden = adapter.forward(build, projection_hooks=hooks)
        # Retaining all 33 residual tensors for five clean prompts costs substantially more VRAM than the model
        # needs for a forward.  The frozen residual control needs only hidden_states[9] (= layer-8 output).
        resid_l8 = hidden[9].detach().clone() if keep_resid_l8 else None
        del hidden
        return np.asarray(actions, np.float64), resid_l8, kv, np.asarray(norm, np.float64)

    def write_clean(base, name, actions, norm, repeat_diff=None):
        key = (base["panel"], base["task"], base["init"], name)
        if key in existing_clean:
            return
        clean_f.write(json.dumps({
            **base,
            "prompt_key": name,
            "actions": actions.tolist(),
            "norm_actions": norm.tolist(),
            "repeat_max_abs_diff": repeat_diff,
        }) + "\n")
        existing_clean.add(key)

    n_units = 0
    t_start = time.time()
    for panel in panels:
        pair_map = panel_maps[panel]
        for task_a, task_b in pair_map.items():
            task = suite.get_task(task_a)
            label_a, label_b = task.language, suite.get_task(task_b).language
            tasks_c = list(FROZEN_RESAMPLE_TASKS[panel][task_a])
            labels_c = [suite.get_task(task_c).language for task_c in tasks_c]
            if len(set(tasks_c)) != 3 or any(task_c in (task_a, task_b) for task_c in tasks_c):
                raise AssertionError(f"invalid frozen resample tasks panel={panel} task={task_a}: {tasks_c}")
            if any(target_objects[task_c] in (target_objects[task_a], target_objects[task_b]) for task_c in tasks_c):
                raise AssertionError(f"resample target overlaps A/B panel={panel} task={task_a}: {tasks_c}")
            env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            env.reset()
            tracker = ContactTracker(env)
            init_states = suite.get_task_init_states(task_a)
            if max(init_ids) >= len(init_states):
                raise IndexError(f"task {task_a} has {len(init_states)} init states; requested {max(init_ids)}")
            print(
                f"[oft-kv] panel={panel} task={task_a} A={label_a!r} B=t{task_b}:{label_b!r} "
                f"C={list(zip(tasks_c, labels_c))}", flush=True,
            )
            for init_index, init_id in enumerate(init_ids):
                unit_key = (panel, task_a, init_id)
                if unit_key in completed:
                    continue
                set_seed_everywhere(args.seed)
                obs = get_initial_observation(env, init_states[init_id])
                positions = tracker.object_positions()
                obj_a, obj_b = target_objects[task_a], target_objects[task_b]
                u = None
                if obj_a in positions and obj_b in positions:
                    delta = np.asarray(positions[obj_b]) - np.asarray(positions[obj_a])
                    if np.linalg.norm(delta) > 1e-9:
                        u = delta / np.linalg.norm(delta)

                xinit = init_ids[(init_index + 1) % len(init_ids)]
                xobs = get_initial_observation(env, init_states[xinit])
                builds = {
                    "A": adapter.build(obs, label_a),
                    "B": adapter.build(obs, label_b),
                    "xA": adapter.build(xobs, label_a),
                    "xB": adapter.build(xobs, label_b),
                }
                for ci, label_c in enumerate(labels_c, 1):
                    builds[f"C{ci}"] = adapter.build(obs, label_c)
                if panel == "equal":
                    if tok_count(task_a) != tok_count(task_b):
                        raise AssertionError(f"frozen equal-token pair t{task_a}/t{task_b} no longer token-matched")
                    if builds["A"]["n_prompt"] != builds["B"]["n_prompt"]:
                        raise AssertionError(f"full prompt lengths differ for equal panel t{task_a}/t{task_b}")
                    if len(builds["A"]["seg"]["INSTR"]) != len(builds["B"]["seg"]["INSTR"]):
                        raise AssertionError(f"instruction span lengths differ for equal panel t{task_a}/t{task_b}")

                paraphrase = {}
                if panel == "equal":
                    for key, label in (("A", label_a), ("B", label_b)):
                        para_label = frozen_paraphrase(label)
                        meta = {"label": para_label, "eligible": False, "reason": "no_put_token"}
                        if para_label is not None:
                            para_key = "para" + key
                            para_build = adapter.build(obs, para_label)
                            same_label_tokens = (
                                len(adapter.tok(label.lower(), add_special_tokens=False)["input_ids"])
                                == len(adapter.tok(para_label.lower(), add_special_tokens=False)["input_ids"])
                            )
                            same_full_prompt = para_build["n_prompt"] == builds[key]["n_prompt"]
                            same_instr_span = len(para_build["seg"]["INSTR"]) == len(builds[key]["seg"]["INSTR"])
                            meta.update({
                                "eligible": bool(same_label_tokens and same_full_prompt and same_instr_span),
                                "reason": "matched" if same_label_tokens and same_full_prompt and same_instr_span
                                else "token_or_full_prompt_length_mismatch",
                                "same_label_tokens": same_label_tokens,
                                "same_full_prompt": same_full_prompt,
                                "same_instruction_span": same_instr_span,
                            })
                            if meta["eligible"]:
                                builds[para_key] = para_build
                                paraphrase[key] = para_key
                        pair_key = f"{panel}|{task_a}"
                        manifest["pair_metadata"].setdefault(pair_key, {"paraphrase": {}})
                        manifest["pair_metadata"][pair_key]["paraphrase"][key] = meta
                    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

                clean, hidden, kv, norm = {}, {}, {}, {}
                repeat_diff = {}
                clean_keys = ["A", "B", "C1", "C2", "C3", "xA", "xB", *paraphrase.values()]
                for key in clean_keys:
                    clean[key], clean_hidden, kv[key], norm[key] = clean_forward(
                        builds[key], keep_resid_l8=key in ("A", "B")
                    )
                    hidden[key] = clean_hidden
                    if key in ("A", "B"):
                        acts_repeat, _, _, _ = clean_forward(builds[key], capture=False)
                        repeat_diff[key] = float(np.max(np.abs(acts_repeat - clean[key])))
                        if repeat_diff[key] != 0.0:
                            raise AssertionError(
                                f"clean determinism failed panel={panel} task={task_a} init={init_id} "
                                f"prompt={key}: {repeat_diff[key]}"
                            )
                    base_clean = {
                        "panel": panel,
                        "task": task_a,
                        "init": init_id,
                        "xscene_init": xinit if key.startswith("x") else None,
                        "task_b": task_b,
                        "tasks_c": tasks_c,
                        "same_physical_observation": not key.startswith("x"),
                    }
                    write_clean(base_clean, key, clean[key], norm[key], repeat_diff.get(key))

                base = {
                    "panel": panel,
                    "task": task_a,
                    "task_b": task_b,
                    "tasks_c": tasks_c,
                    "init": init_id,
                    "xscene_init": xinit,
                    "timepoint": 0,
                    "prompt_A": label_a,
                    "prompt_B": label_b,
                    "prompts_C": labels_c,
                    "tok_A": tok_count(task_a),
                    "tok_B": tok_count(task_b),
                    "toks_C": [tok_count(task_c) for task_c in tasks_c],
                    "n_prompt_A": builds["A"]["n_prompt"],
                    "n_prompt_B": builds["B"]["n_prompt"],
                    "clean_repeat_max_abs": max(repeat_diff.values()),
                    "main_source_destination_same_physical_observation": True,
                }
                for src, dst in (("B", "A"), ("A", "B")):
                    direction = f"{src}->{dst}"
                    img_dst, img_src = builds[dst]["seg"]["IMG"], builds[src]["seg"]["IMG"]
                    xsrc = "x" + src
                    conditions = list(PRIMARY_CONDITIONS)
                    if panel == "equal":
                        conditions += EQUAL_ONLY_CONDITIONS
                        if dst in paraphrase:
                            conditions += (PARAPHRASE_CONDITION,)
                    for condition in conditions:
                        row_key = (panel, task_a, init_id, direction, condition)
                        if row_key in existing_rows:
                            continue
                        audit = empty_audit()
                        projection_hooks = None
                        layer_hooks = None
                        cross_actions = None
                        if condition == "single_resid_l8":
                            layer_hooks = {8: residual_patch_hook(
                                img_dst,
                                img_src,
                                hidden[src],
                                hidden[dst],
                                audit,
                            )}
                        elif condition == "kv_img_8_31":
                            projection_hooks = projection_patch_hooks(
                                live_layers, img_dst, img_src, kv[src], kv[dst], "donor", audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                        elif condition == "kv_img_0_7":
                            projection_hooks = projection_patch_hooks(
                                early_layers, img_dst, img_src, kv[src], kv[dst], "donor", audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                        elif condition == "kv_img_8_31_self":
                            projection_hooks = projection_patch_hooks(
                                live_layers, img_dst, img_dst, kv[dst], kv[dst], "self", audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                        elif condition == "kv_img_8_31_random":
                            projection_hooks = projection_patch_hooks(
                                live_layers, img_dst, img_src, kv[src], kv[dst], "random", audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                        elif condition in RESAMPLE_CONDITIONS:
                            ci = int(condition.rsplit("c", 1)[1])
                            ckey = f"C{ci}"
                            projection_hooks = projection_patch_hooks(
                                live_layers, img_dst, builds[ckey]["seg"]["IMG"], kv[ckey], kv[dst], "donor", audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                        elif condition == "kv_img_8_31_xscene":
                            projection_hooks = projection_patch_hooks(
                                live_layers, img_dst, builds[xsrc]["seg"]["IMG"], kv[xsrc], kv[dst], "donor", audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                            cross_actions = clean[xsrc]
                        elif condition == "kv_instr_8_31":
                            projection_hooks = projection_patch_hooks(
                                live_layers,
                                builds[dst]["seg"]["INSTR"],
                                builds[src]["seg"]["INSTR"],
                                kv[src],
                                kv[dst],
                                "donor",
                                audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                        elif condition == "kv_both_8_31":
                            projection_hooks = projection_patch_hooks(
                                live_layers,
                                img_dst + builds[dst]["seg"]["INSTR"],
                                img_src + builds[src]["seg"]["INSTR"],
                                kv[src],
                                kv[dst],
                                "donor",
                                audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                        elif condition == PARAPHRASE_CONDITION:
                            para_key = paraphrase[dst]
                            projection_hooks = projection_patch_hooks(
                                live_layers,
                                img_dst,
                                builds[para_key]["seg"]["IMG"],
                                kv[para_key],
                                kv[dst],
                                "donor",
                                audit,
                                stable_seed(args.seed, panel, task_a, init_id, direction, condition),
                            )
                        else:
                            raise AssertionError(condition)

                        _, patched, _ = adapter.forward(
                            builds[dst], layer_hooks=layer_hooks, projection_hooks=projection_hooks
                        )
                        patched = np.asarray(patched, dtype=np.float64)
                        row = {
                            **base,
                            "direction": direction,
                            "source_prompt": src,
                            "destination_prompt": dst,
                            "condition": condition,
                            "diagnostic_only": condition == "kv_img_8_31_xscene",
                            "patched_actions": patched.tolist(),
                            "action_max_abs_to_dst": float(np.max(np.abs(patched - clean[dst]))),
                            "write_audit": finish_audit(audit),
                            **normalized_metrics(patched, clean[dst], clean[src]),
                            **axis_fields(patched, clean[dst], clean[src], u),
                        }
                        if cross_actions is not None:
                            xmet = normalized_metrics(patched, clean[dst], cross_actions)
                            row.update({
                                "l2_to_xscene_src": xmet["l2_to_src"],
                                "l2_xscene_src_dst": xmet["l2_src_dst"],
                                "D_xscene_src": xmet["D_src"],
                                "R_xscene_full": xmet["R_full"],
                            })
                        if condition == PARAPHRASE_CONDITION:
                            para_key = paraphrase[dst]
                            para_metrics = normalized_metrics(clean[para_key], clean[dst], clean[src])
                            row.update({
                                "paraphrase_key": para_key,
                                "paraphrase_label": manifest["pair_metadata"][f"{panel}|{task_a}"]["paraphrase"][dst]["label"],
                                "paraphrase_full_prompt": builds[para_key]["prompt"],
                                "D_clean_para_to_dst": para_metrics["D_dst"],
                                "l2_clean_para_to_dst": para_metrics["l2_to_dst"],
                                "clean_paraphrase_actions": clean[para_key].tolist(),
                            })
                        rows_f.write(json.dumps(row) + "\n")
                        existing_rows.add(row_key)
                    rows_f.flush()
                units_f.write(json.dumps({
                    "panel": panel,
                    "task": task_a,
                    "task_b": task_b,
                    "init": init_id,
                    "xscene_init": xinit,
                    "completed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }) + "\n")
                units_f.flush()
                clean_f.flush()
                n_units += 1
                print(
                    f"[oft-kv] done panel={panel} task={task_a} init={init_id} xinit={xinit} "
                    f"new_units={n_units} elapsed_min={(time.time() - t_start) / 60:.1f}", flush=True,
                )
            env.close()
    rows_f.close()
    clean_f.close()
    units_f.close()
    manifest["completed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["new_units"] = n_units
    manifest["elapsed_min"] = (time.time() - t_start) / 60
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[oft-kv] DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="synthetic validation; no checkpoint or LIBERO")
    parser.add_argument("--out")
    parser.add_argument("--panels", default="legacy,equal")
    parser.add_argument("--init_ids", default="25-34", help="official held-out panel; frozen default is 25-34")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--allow_nonfrozen_checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
        return
    if not args.out:
        parser.error("--out is required unless --smoke is used")
    run_experiment(args)


if __name__ == "__main__":
    main()
