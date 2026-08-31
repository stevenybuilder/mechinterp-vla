#!/usr/bin/env python
"""OFT CROSS-SCENE POSITIVE CONTROL  (the control list in docs/protocol.md item 1).

WHY THIS RUN EXISTS
-------------------
In the A/B run (`imgsite`), patching the 512 IMG positions gave D_src = 0.999/0.977/0.983 at
L8/16/24 -- indistinguishable from the PROPRIO no-op anchor (1.000).  We read that as "image
positions are inert for the instruction contrast".  But the same rows show **D_dst = 0.015**: the
patch barely perturbed the output AT ALL.  Two explanations are not separable from that run:

  (i) the IMG site genuinely carries nothing that the readout uses, or
  (ii) A and B are the SAME observation with different prompts, so the donor IMG activations are
       nearly identical to the ones they replace -- the "patch" is a near-no-op by construction --
       and/or the edit is undone downstream.

This run settles it.  Same prompt, DIFFERENT observation (a different init state of the same task).
Now the donor IMG activations are genuinely different.  If the IMG site has teeth, D_dst must rise
far above the no-op anchor.  If it does not, no OFT IMG number is safely quotable.

READOUT (endpoint is L2 in action space, never R -- the L2-endpoint rule (docs/protocol.md))
  D_src = |patched - clean_src| / |clean_src - clean_dst|   0 = output became the donor scene's
  D_dst = |patched - clean_dst| / |clean_src - clean_dst|   0 = no-op, large = the patch had teeth
PRIMARY here is **D_dst** (patch efficacy).  In the A/B run the primary was D_src (transfer).  Say
so when writing: they answer different questions.

UNIT IS THE CELL (the cell-unit rule (docs/protocol.md)): cell = task x init-pair x timepoint x direction.  Median within
cell, then median across cells.  Episodes are never the unit.

CONTROLS
  self      patch the recipient's OWN activations into itself -> must be EXACTLY 0.000 / 0.000.
            This is the anti-silent-no-op anchor (this harness has produced four silent no-ops that would each have looked like a clean null).
  rand0/1   equal-norm random direction at the same positions -> separates content from norm.
  INSTR     same prompt in both scenes, so the instruction states differ only by image bleed;
            a small effect here is expected and is the negative control.
  PROPRIO   genuinely differs across scenes -> second positive control.
  ACT       action-token positions, cheap, included for completeness.

Every hook records how many positions it actually wrote and the norm of the change it made; a patch
whose recorded delta is 0 is flagged, not silently averaged in.  Full patched action chunks are
logged so any other metric can be re-derived offline without a rerun.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from libero.libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation, process_action  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402

from oft_stage2 import OFTAdapter  # noqa: E402  (same adapter as the A/B run -- do not fork it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task_ids", default="all")
    ap.add_argument("--pairs", default="0-1,2-3", help="init-state pairs, e.g. 0-1,2-3")
    ap.add_argument("--timepoints", default="0,16")
    ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--sites", default="IMG,INSTR,PROPRIO,ACT")
    ap.add_argument("--n_rand", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    timepoints = [int(x) for x in args.timepoints.split(",")]
    sites = args.sites.split(",")
    pairs = [tuple(int(v) for v in p.split("-")) for p in args.pairs.split(",")]

    cfg = GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=args.suite,
                         num_open_loop_steps=NUM_ACTIONS_CHUNK, center_crop=True, seed=args.seed)
    model, action_head, proprio_projector, _n, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    ad = OFTAdapter(cfg, model, action_head, proprio_projector, processor)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(range(suite.n_tasks)) if args.task_ids == "all" else [int(x) for x in args.task_ids.split(",")]

    manifest = {"kind": "oft_cross_scene_positive_control", "suite": args.suite, "checkpoint": args.checkpoint,
                "layers": layers, "timepoints": timepoints, "pairs": [list(p) for p in pairs], "sites": sites,
                "n_rand": args.n_rand, "seed": args.seed, "n_patches": ad.n_patches, "n_layers": ad.n_layers,
                "design": "SAME prompt (task A label), DIFFERENT observation (two init states of the same task); "
                          "donor scene activations patched into recipient scene forward pass",
                "primary_endpoint": "D_dst = |patched-clean_dst| / |clean_src-clean_dst| (patch efficacy); "
                                    "D_src = |patched-clean_src| / |clean_src-clean_dst| (transfer completeness)",
                "unit": "cell = task x init_pair x timepoint x direction",
                "hidden_states_convention": "hidden_states[l+1] = output of layers[l]; patch at layer L edits it",
                "tasks": {}}
    patch_f = (out / "patching.jsonl").open("w")
    clean_f = (out / "clean_chunks.jsonl").open("w")
    verify = None
    t0 = time.time(); n_pass = 0

    for tid in task_ids:
        task = suite.get_task(tid)
        A_label = task.language
        manifest["tasks"][tid] = {"A": A_label}
        init_states = suite.get_task_init_states(tid)
        env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        env.reset()
        print(f"[xs] task {tid} '{A_label}'", flush=True)

        for (ia, ib) in pairs:
            obs_by = {}
            for init_id in (ia, ib):
                set_seed_everywhere(args.seed)
                env.reset()
                obs = env.set_init_state(init_states[init_id])
                for _ in range(10):
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                queue = deque()
                for step in range(max(timepoints) + 1):
                    if step in timepoints:
                        o, _ = prepare_observation(obs, resize_size)
                        obs_by[(init_id, step)] = o
                    if step == max(timepoints):
                        break
                    if not queue:
                        o, _ = prepare_observation(obs, resize_size)
                        _, acts, _ = ad.forward(ad.build(o, A_label))
                        queue.extend(list(acts))
                    obs, _, _, _ = env.step(process_action(np.array(queue.popleft()), cfg.model_family).tolist())

            for tp in timepoints:
                oA, oB = obs_by[(ia, tp)], obs_by[(ib, tp)]
                img_l1 = float(np.abs(oA["full_image"].astype(np.float64) - oB["full_image"].astype(np.float64)).mean())
                builds = {ia: ad.build(oA, A_label), ib: ad.build(oB, A_label)}
                # same prompt in both scenes -> token layouts MUST match; this run is free of the
                # positional/RoPE confound that the A/B INSTR result carries.
                same_layout = builds[ia]["n_prompt"] == builds[ib]["n_prompt"]
                assert same_layout, "same prompt produced different token counts"
                if manifest.get("segments_example") is None:
                    manifest["segments_example"] = {k: (v if isinstance(v, int) else [v[0], v[-1], len(v)])
                                                    for k, v in builds[ia]["seg"].items()}
                    manifest["prompt_example"] = builds[ia]["prompt"]
                clean, hs = {}, {}
                for k in (ia, ib):
                    _, acts, hidden = ad.forward(builds[k]); n_pass += 1
                    clean[k] = acts
                    hs[k] = torch.stack(hidden, 0)[:, 0]  # [33, L, D]
                    clean_f.write(json.dumps({"task": tid, "pair": [ia, ib], "init": k, "t": tp, "label": A_label,
                                              "actions": acts.tolist(), "img_meanabs_diff": img_l1}) + "\n")

                if verify is None:
                    _, acts2, _ = ad.forward(builds[ia]); n_pass += 1
                    verify = {"repeat_max_abs_diff_actions": float(np.max(np.abs(acts2 - clean[ia]))),
                              "attn_impl": ad.vla.language_model.config._attn_implementation}

                for src, dst in ((ib, ia), (ia, ib)):
                    cd, cs = clean[dst].ravel(), clean[src].ravel()
                    den = float(np.linalg.norm(cs - cd))
                    for site in sites:
                        pos = builds[dst]["seg"][site]           # identical in both: same prompt
                        assert pos == builds[src]["seg"][site], "segment misalignment despite same prompt"
                        for L in layers:
                            src_act = hs[src][L + 1, pos, :].clone()
                            dst_act = hs[dst][L + 1, pos, :].clone()
                            variants = {"patch": src_act, "self": dst_act}
                            dn = (src_act - dst_act).float().norm(dim=-1, keepdim=True)
                            for r in range(args.n_rand):
                                g = torch.randn(dst_act.shape, generator=torch.Generator().manual_seed(
                                    args.seed + 100 * r + L), dtype=torch.float32).to(dst_act.device)
                                variants[f"rand{r}"] = (dst_act.float() + g / g.norm(dim=-1, keepdim=True) * dn).to(dst_act.dtype)
                            for vname, val in variants.items():
                                probe = {}
                                def fn(hidden, val=val, pos=pos, probe=probe):
                                    hidden = hidden.clone()
                                    probe["delta"] = float((val.float() - hidden[0, pos, :].float()).norm())
                                    probe["before"] = float(hidden[0, pos, :].float().norm())
                                    hidden[0, pos, :] = val
                                    probe["n_edited"] = len(pos)
                                    return hidden
                                _, acts_p, _ = ad.forward(builds[dst], layer_hooks={L: fn}); n_pass += 1
                                assert probe.get("n_edited") == len(pos), "HOOK DID NOT FIRE -- silent no-op"
                                cp = acts_p.ravel()
                                patch_f.write(json.dumps({
                                    "task": tid, "pair": [ia, ib], "t": tp, "dir": f"{src}->{dst}",
                                    "src_init": src, "dst_init": dst, "site": site, "layer": L, "variant": vname,
                                    "n_pos": len(pos), "same_layout": same_layout, "img_meanabs_diff": img_l1,
                                    "edit_delta_norm": probe["delta"], "resid_norm_before": probe["before"],
                                    "l2_to_dst": float(np.linalg.norm(cp - cd)),
                                    "l2_to_src": float(np.linalg.norm(cp - cs)),
                                    "l2_src_dst": den,
                                    "D_src": float(np.linalg.norm(cp - cs)) / den if den > 1e-9 else None,
                                    "D_dst": float(np.linalg.norm(cp - cd)) / den if den > 1e-9 else None,
                                    "actions_patched": acts_p.tolist(),
                                }) + "\n")
                patch_f.flush()
                print(f"[xs] task {tid} pair {ia}-{ib} t{tp} passes={n_pass} "
                      f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
        env.close()

    patch_f.close(); clean_f.close()
    manifest["verify"] = verify
    manifest["n_passes"] = n_pass
    manifest["elapsed_min"] = (time.time() - t0) / 60
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[xs] DONE", flush=True)


if __name__ == "__main__":
    main()
