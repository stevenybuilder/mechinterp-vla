#!/usr/bin/env python
"""OpenVLA-OFT Stage 2 harness (plan docs/vla-generalization-and-diffing-plan.md §A; protocol §9.1 OFT column).

Run from the openvla-oft repo root. For libero_goal, N init states x 10 tasks x timepoints:
  (1) clean forward passes for prompt A (correct), B (wrong-object, GOAL_WRONG_MAP), C (resample: unrelated task);
      store per-layer residual features at INSTR (mean), ACT (mean), PROPRIO positions and the action chunk;
  (2) attention mass from the 56 action-query rows to INSTR / IMG / PROPRIO / FMT / ACT columns per layer x head,
      recomputed from q/k inside a pre-hook on self_attn (the fork's SDPA path returns no weights; its eager path
      applies a CAUSAL mask, so output_attentions would measure a different model), plus a matched-count random
      IMG-column control (10 draws);
  (3) residual patching at layers --layers, sites {INSTR, ACT, PROPRIO}, both directions (B->A noising, A->B
      denoising), with equal-norm random-direction and resample (C) baselines; metric = first chunk projected on the
      A-B end-effector displacement axis u = (p_B - p_A)/|p_B - p_A| (env units, first 3 action dims), plus the
      full-chunk projection.
  (4) mask-directionality verification: change the trailing stop token id and test whether hidden states at
      early positions move at layer 1 (bidirectional) or not (causal).

Everything is deterministic (L1 head, no sampling); a same-input repeat is asserted bit-identical.
"""
from __future__ import annotations

import argparse
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

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation, process_action  # noqa: E402
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK  # noqa: E402
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: E402

from run_oft_prompt_conditions import GOAL_TARGET_OBJECT, GOAL_WRONG_MAP, ContactTracker  # noqa: E402

DEVICE = "cuda"
N_ACT = ACTION_DIM * NUM_ACTIONS_CHUNK  # 56
EMPTY_TOKEN = 29871


class OFTAdapter:
    """Manual re-implementation of OpenVLAForActionPrediction.predict_action with hook access."""

    def __init__(self, cfg, vla, action_head, proprio_projector, processor):
        self.cfg, self.vla, self.head, self.pp, self.proc = cfg, vla, action_head, proprio_projector, processor
        self.layers = vla.language_model.model.layers
        self.n_layers = len(self.layers)
        self.tok = processor.tokenizer
        self.n_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        self.unnorm_key = cfg.unnorm_key

    # ---------- inputs ----------
    def build(self, obs, task_label):
        """Return dict with pixel_values, input_ids, attention_mask, proprio, prompt, segments."""
        all_images = [obs["full_image"], obs["wrist_image"]]
        all_images = prepare_images_for_vla(all_images, self.cfg)
        primary = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = self.proc(prompt, primary).to(DEVICE, dtype=torch.bfloat16)
        wrist = [self.proc(prompt, im).to(DEVICE, dtype=torch.bfloat16) for im in all_images]
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"]] + [w["pixel_values"] for w in wrist], dim=1)
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == EMPTY_TOKEN):
            input_ids = torch.cat((input_ids, torch.tensor([[EMPTY_TOKEN]], device=input_ids.device)), dim=1)
        attention_mask = torch.ones_like(input_ids)
        n_prompt = input_ids.shape[-1] - 1  # excludes BOS
        # instruction token span within the prompt ids (offset-mapping on the fast tokenizer)
        enc = self.tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
        ids_check = enc["input_ids"]
        assert ids_check == input_ids[0, : len(ids_check)].tolist(), "tokenizer mismatch with processor"
        lo = prompt.index(task_label.lower()) if task_label else len("In: What action should the robot take to ")
        hi = lo + len(task_label.lower())
        instr_tok = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a >= lo - 1 and b <= hi and j > 0]
        # sequence layout: [BOS] + n_patches(+1 proprio) + prompt[1:] (n_prompt tokens) + 56 action + stop
        npp = self.n_patches + 1  # proprio token appended after patches
        seg = {
            "BOS": [0],
            "IMG": list(range(1, 1 + self.n_patches)),
            "PROPRIO": [1 + self.n_patches],
            "PROMPT": list(range(1 + npp, 1 + npp + n_prompt)),
            "INSTR": [1 + npp + (j - 1) for j in instr_tok],
            "ACT": list(range(1 + npp + n_prompt, 1 + npp + n_prompt + N_ACT)),
            "STOP": [1 + npp + n_prompt + N_ACT],
        }
        seg["FMT"] = [p for p in seg["PROMPT"] if p not in set(seg["INSTR"])]
        seg["ACT_START"] = seg["ACT"][0]
        proprio = normalize_proprio(np.array(obs["state"], dtype=np.float64),
                                    self.vla.norm_stats[self.unnorm_key]["proprio"])
        return {"pixel_values": inputs["pixel_values"], "input_ids": input_ids, "attention_mask": attention_mask,
                "proprio": proprio, "prompt": prompt, "n_prompt": n_prompt, "seg": seg,
                "instr_ids": [ids_check[j] for j in instr_tok]}

    # ---------- forward ----------
    @torch.inference_mode()
    def forward(self, b, layer_hooks=None, attn_hook=None, stop_id=None, projection_hooks=None):
        """layer_hooks: {layer_idx: fn(hidden[B,L,D]) -> hidden} applied to the layer output (= hidden_states[idx+1]).
        attn_hook: fn(layer_idx, self_attn, normed_hidden, position_ids) called before every self_attn.
        projection_hooks: {layer_idx: {"k"|"v"|"q": fn(layer_idx, kind, projected, input_hidden)}}.
        Each callback receives the *projected*, pre-RoPE tensor [B,L,D_proj] and may return a replacement tensor
        (or None for capture-only use).  The default is None, so existing Stage-2 callers are unchanged.  Downstream
        causal tests should patch k/v only: exposing q here is useful for auditing, not an endorsement to edit it.
        Returns (norm_actions [8,7], actions env units [8,7], hidden_states tuple of 33 x [1,L,D])."""
        vla = self.vla
        input_ids, attention_mask = vla._prepare_input_for_action_prediction(b["input_ids"], b["attention_mask"])
        if stop_id is not None:
            input_ids = input_ids.clone()
            input_ids[:, -1] = stop_id
        labels = b["input_ids"].clone()
        labels[:] = IGNORE_INDEX
        labels = vla._prepare_labels_for_action_prediction(labels, input_ids)
        input_embeddings = vla.get_input_embeddings()(input_ids)
        all_actions_mask = vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(1, -1, input_embeddings.shape[2])
        patches = vla._process_vision_features(b["pixel_values"], language_embeddings, False)
        proprio = torch.tensor(b["proprio"]).to(patches.device, dtype=patches.dtype)
        patches = vla._process_proprio_features(patches, proprio, self.pp)
        input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)
        mm_emb, mm_mask = vla._build_multimodal_attention(input_embeddings, patches, attention_mask)
        assert mm_emb.shape[1] == b["seg"]["STOP"][0] + 1, (mm_emb.shape, b["seg"]["STOP"])
        handles = []
        if layer_hooks:
            for li, fn in layer_hooks.items():
                def _mk(fn):
                    def hook(mod, inp, out):
                        if isinstance(out, tuple):
                            return (fn(out[0]),) + tuple(out[1:])
                        return fn(out)
                    return hook
                handles.append(self.layers[li].register_forward_hook(_mk(fn)))
        if attn_hook is not None:
            for li, layer in enumerate(self.layers):
                def _mk2(li, layer):
                    def pre(mod, args, kwargs):
                        h = kwargs.get("hidden_states", args[0] if args else None)
                        attn_hook(li, mod, h, kwargs.get("position_ids"))
                    return pre
                handles.append(layer.self_attn.register_forward_pre_hook(_mk2(li, layer), with_kwargs=True))
        if projection_hooks:
            for li, by_kind in projection_hooks.items():
                if li < 0 or li >= self.n_layers:
                    raise IndexError(f"projection hook layer {li} outside [0,{self.n_layers})")
                for kind, fn in by_kind.items():
                    if kind not in ("k", "v", "q"):
                        raise ValueError(f"unknown projection kind {kind!r}; expected k, v, or q")

                    def _mk_proj(li, kind, fn):
                        def hook(mod, inp, projected):
                            input_hidden = inp[0] if inp else None
                            replacement = fn(li, kind, projected, input_hidden)
                            if replacement is None:
                                return None
                            if replacement.shape != projected.shape:
                                raise ValueError(
                                    f"projection hook {kind}@{li} changed shape "
                                    f"{tuple(projected.shape)} -> {tuple(replacement.shape)}"
                                )
                            return replacement
                        return hook

                    proj = getattr(self.layers[li].self_attn, f"{kind}_proj")
                    handles.append(proj.register_forward_hook(_mk_proj(li, kind, fn)))
        try:
            out = vla.language_model(input_ids=None, attention_mask=mm_mask, position_ids=None, past_key_values=None,
                                     inputs_embeds=mm_emb, labels=None, use_cache=None, output_attentions=False,
                                     output_hidden_states=True, return_dict=True)
        finally:
            for h in handles:
                h.remove()
        s = b["seg"]["ACT_START"]
        act_h = out.hidden_states[-1][:, s: s + N_ACT, :]
        norm_actions = self.head.predict_action(act_h).reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        norm_np = norm_actions.float().cpu().numpy()
        actions = vla._unnormalize_actions(norm_np, self.unnorm_key)
        return norm_np, np.asarray(actions, dtype=np.float64), out.hidden_states

    def attn_masses(self, li, self_attn, h, position_ids, seg, rand_cols):
        """Recompute post-softmax attention of ACT rows; return per-head mass on segments (fully bidirectional, no
        mask: B=1, no padding -> the fork passes causal_mask=None, is_causal=False)."""
        bsz, q_len, _ = h.shape
        q = self_attn.q_proj(h).view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
        k = self_attn.k_proj(h).view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(q_len, device=h.device)[None]
        cos, sin = self_attn.rotary_emb(k, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        qa = q[:, :, seg["ACT"], :].float()
        scores = torch.matmul(qa, k.float().transpose(2, 3)) / math.sqrt(self_attn.head_dim)  # [1,H,56,L]
        w = torch.softmax(scores, dim=-1)[0]  # [H,56,L]
        wm = w.mean(dim=1)  # [H, L] mean over action-query rows
        res = {}
        for name in ("INSTR", "IMG", "PROPRIO", "FMT", "ACT", "BOS", "STOP"):
            idx = torch.tensor(seg[name], device=h.device)
            res[name] = wm[:, idx].sum(dim=-1).cpu().numpy()
        top = torch.topk(wm, 5, dim=-1)
        res["sink_top5"] = top.values.sum(dim=-1).cpu().numpy()
        res["sink_cols"] = top.indices.cpu().numpy()
        res["rand_img"] = np.stack([wm[:, torch.tensor(rc, device=h.device)].sum(-1).cpu().numpy() for rc in rand_cols])
        return res


def axis_metric(actions, u, k=NUM_ACTIONS_CHUNK):
    return float(np.sum(actions[:k, :3] @ u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task_ids", default="all")
    ap.add_argument("--n_init", type=int, default=25)
    ap.add_argument("--timepoints", default="0,16")
    ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--n_rand", type=int, default=3, help="random-direction draws per patch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    timepoints = [int(x) for x in args.timepoints.split(",")]

    cfg = GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=args.suite,
                         num_open_loop_steps=NUM_ACTIONS_CHUNK, center_crop=True, seed=args.seed)
    model, action_head, proprio_projector, _n, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    ad = OFTAdapter(cfg, model, action_head, proprio_projector, processor)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(range(suite.n_tasks)) if args.task_ids == "all" else [int(x) for x in args.task_ids.split(",")]
    rng = np.random.default_rng(args.seed)

    # resample prompt C: a task whose target differs from both A and B; prefer equal instruction token count
    def tok_count(label):
        return len(ad.tok(label.lower(), add_special_tokens=False)["input_ids"])

    manifest = {"suite": args.suite, "checkpoint": args.checkpoint, "layers": layers, "timepoints": timepoints,
                "n_init": args.n_init, "tasks": {}, "n_layers": ad.n_layers, "n_patches": ad.n_patches,
                "hidden_states_convention": "hidden_states[0]=embeddings; hidden_states[l+1]=output of layers[l]; "
                                            "patch at layer L edits output of layers[L] (=hidden_states[L+1])"}
    feats = {}  # (task, init, t, prompt) -> {INSTR_mean, ACT_mean, PROPRIO, OBJ_last} arrays [33, D]
    attn_rows = []
    patch_f = (out / "patching.jsonl").open("w")
    clean_f = (out / "clean_chunks.jsonl").open("w")
    verify = None
    t_start = time.time()
    n_pass = 0
    for tid in task_ids:
        task = suite.get_task(tid)
        A_label = task.language
        b_tid = GOAL_WRONG_MAP[tid] if args.suite == "libero_goal" else (tid + 1) % suite.n_tasks
        B_label = suite.get_task(b_tid).language
        objA, objB = GOAL_TARGET_OBJECT[tid], GOAL_TARGET_OBJECT[b_tid]
        cands = [j for j in task_ids if j not in (tid, b_tid) and GOAL_TARGET_OBJECT[j] not in (objA, objB)]
        eq = [j for j in cands if tok_count(suite.get_task(j).language) == tok_count(A_label)]
        c_tid = (eq or cands)[0]
        C_label = suite.get_task(c_tid).language
        manifest["tasks"][tid] = {"A": A_label, "B": B_label, "B_task": b_tid, "C": C_label, "C_task": c_tid,
                                  "objA": objA, "objB": objB, "tok_A": tok_count(A_label), "tok_B": tok_count(B_label),
                                  "tok_C": tok_count(C_label)}
        init_states = suite.get_task_init_states(tid)
        env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        env.reset()
        tracker = ContactTracker(env)
        print(f"[s2] task {tid} A='{A_label}' B='{B_label}' C='{C_label}' objA={objA} objB={objB}", flush=True)
        for init_id in range(args.n_init):
            set_seed_everywhere(args.seed)
            env.reset()
            obs = env.set_init_state(init_states[init_id])
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            obs_by_t = {}
            t = 0
            from collections import deque
            queue = deque()
            for step in range(max(timepoints) + 1):
                if step in timepoints:
                    o, _ = prepare_observation(obs, resize_size)
                    pos = tracker.object_positions()
                    obs_by_t[step] = (o, np.array(pos[objA]), np.array(pos[objB]))
                if step == max(timepoints):
                    break
                if not queue:  # closed-loop clean-A rollout between timepoints
                    o, _ = prepare_observation(obs, resize_size)
                    bA = ad.build(o, A_label)
                    _, acts, _ = ad.forward(bA)
                    queue.extend(list(acts))
                a = process_action(np.array(queue.popleft()), cfg.model_family)
                obs, _, _, _ = env.step(a.tolist())
            for tp in timepoints:
                o, pA, pB = obs_by_t[tp]
                u = (pB - pA) / (np.linalg.norm(pB - pA) + 1e-9)
                builds = {"A": ad.build(o, A_label), "B": ad.build(o, B_label), "C": ad.build(o, C_label)}
                seg = builds["A"]["seg"]
                if manifest.get("segments_example") is None:
                    manifest["segments_example"] = {k: (v if isinstance(v, int) else [v[0], v[-1], len(v)])
                                                    for k, v in seg.items()}
                    manifest["prompt_example"] = builds["A"]["prompt"]
                    manifest["instr_ids_example"] = builds["A"]["instr_ids"]
                # segments must be position-identical across A/B/C for residual patching; else patch by role
                same_layout = all(builds[p]["n_prompt"] == builds["A"]["n_prompt"] for p in "BC")
                clean = {}
                hs = {}
                rand_cols = [rng.choice(seg["IMG"], size=len(seg["INSTR"]), replace=False).tolist() for _ in range(10)]
                for p in "ABC":
                    attn_acc = {}
                    def ah(li, mod, h, pid, p=p, attn_acc=attn_acc, sg=builds[p]["seg"]):
                        if p == "A":
                            attn_acc[li] = ad.attn_masses(li, mod, h, pid, sg, rand_cols)
                    norm_np, acts, hidden = ad.forward(builds[p], attn_hook=ah if p == "A" else None)
                    n_pass += 1
                    clean[p] = acts
                    sg = builds[p]["seg"]
                    H = torch.stack(hidden, 0)[:, 0]  # [33, L, D]
                    hs[p] = H
                    feats[(tid, init_id, tp, p)] = {
                        "INSTR_mean": H[:, sg["INSTR"], :].float().mean(1).cpu().numpy().astype(np.float16),
                        "ACT_mean": H[:, sg["ACT"], :].float().mean(1).cpu().numpy().astype(np.float16),
                        "PROPRIO": H[:, sg["PROPRIO"][0], :].float().cpu().numpy().astype(np.float16),
                        "INSTR_last": H[:, sg["INSTR"][-1], :].float().cpu().numpy().astype(np.float16),
                    }
                    if p == "A":
                        for li, r in attn_acc.items():
                            attn_rows.append({"task": tid, "init": init_id, "t": tp, "layer": li,
                                              **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()},
                                              "n_instr": len(sg["INSTR"]), "n_keys": sg["STOP"][0] + 1})
                    clean_f.write(json.dumps({"task": tid, "init": init_id, "t": tp, "prompt": p,
                                              "label": {"A": A_label, "B": B_label, "C": C_label}[p],
                                              "actions": acts.tolist(), "norm_actions": norm_np.tolist(),
                                              "m_axis": axis_metric(acts, u), "u": u.tolist(),
                                              "pA": pA.tolist(), "pB": pB.tolist(), "n_prompt": builds[p]["n_prompt"],
                                              "n_instr": len(sg["INSTR"])}) + "\n")
                if verify is None:  # determinism + mask directionality
                    _, acts2, hidden2 = ad.forward(builds["A"])
                    det = float(np.max(np.abs(acts2 - clean["A"])))
                    _, acts3, hidden3 = ad.forward(builds["A"], stop_id=0)
                    d1 = (hidden3[1][0, seg["INSTR"]] - hs["A"][1][seg["INSTR"]]).float().abs().max().item()
                    d_img = (hidden3[1][0, seg["IMG"][:50]] - hs["A"][1][seg["IMG"][:50]]).float().abs().max().item()
                    verify = {"repeat_max_abs_diff_actions": det,
                              "stop_token_change_moves_INSTR_layer1_maxabs": d1,
                              "stop_token_change_moves_IMG_layer1_maxabs": d_img,
                              "attn_impl": ad.vla.language_model.config._attn_implementation,
                              "interpretation": "if INSTR/IMG hidden states at layer 1 change when only the LAST token "
                                                "(stop) id changes, attention is bidirectional over the whole sequence"}
                    manifest["verify"] = verify
                    print("[s2] verify", verify, flush=True)
                # ---------- patching ----------
                for src, dst in (("B", "A"), ("A", "B")):
                    for site in ("INSTR", "ACT", "PROPRIO"):
                        pos_dst = builds[dst]["seg"][site]
                        pos_src = builds[src]["seg"][site]
                        if len(pos_src) != len(pos_dst):  # unequal instruction lengths: patch by role (align ends)
                            n = min(len(pos_src), len(pos_dst))
                            pos_dst, pos_src = pos_dst[-n:], pos_src[-n:]
                        pos_c = builds["C"]["seg"][site]
                        nC = min(len(pos_c), len(pos_dst))
                        for L in layers:
                            src_act = hs[src][L + 1, pos_src, :].clone()  # output of layers[L]
                            dst_act = hs[dst][L + 1, pos_dst, :].clone()
                            c_act = hs["C"][L + 1, pos_c[-nC:], :].clone()
                            variants = {"patch": src_act}
                            delta_norm = (src_act - dst_act).float().norm(dim=-1, keepdim=True)
                            for r in range(args.n_rand):
                                g = torch.randn(dst_act.shape, generator=torch.Generator().manual_seed(
                                    args.seed + 100 * r + L), dtype=torch.float32).to(dst_act.device)
                                g = g / g.norm(dim=-1, keepdim=True)
                                variants[f"rand{r}"] = (dst_act.float() + g * delta_norm).to(dst_act.dtype)
                            res_act = dst_act.clone()
                            res_act[-nC:] = c_act
                            variants["resample"] = res_act
                            for vname, val in variants.items():
                                def fn(hidden, val=val, pos=pos_dst):
                                    hidden = hidden.clone()
                                    hidden[0, pos, :] = val
                                    return hidden
                                _, acts_p, _ = ad.forward(builds[dst], layer_hooks={L: fn})
                                n_pass += 1
                                mA, mB = axis_metric(clean["A"], u), axis_metric(clean["B"], u)
                                m_dst, m_src = axis_metric(clean[dst], u), axis_metric(clean[src], u)
                                cd, cs, cp = clean[dst].ravel(), clean[src].ravel(), acts_p.ravel()
                                den = float(np.dot(cs - cd, cs - cd))
                                patch_f.write(json.dumps({
                                    "task": tid, "init": init_id, "t": tp, "dir": f"{src}->{dst}", "site": site,
                                    "layer": L, "variant": vname, "n_pos": len(pos_dst),
                                    "m_patched": axis_metric(acts_p, u), "m_dst": m_dst, "m_src": m_src,
                                    "m_A": mA, "m_B": mB,
                                    "R_axis": (axis_metric(acts_p, u) - m_dst) / (m_src - m_dst) if abs(m_src - m_dst) > 1e-9 else None,
                                    "R_full": float(np.dot(cp - cd, cs - cd) / den) if den > 1e-12 else None,
                                    "l2_to_dst": float(np.linalg.norm(cp - cd)), "l2_to_src": float(np.linalg.norm(cp - cs)),
                                    "l2_src_dst": float(np.sqrt(den)), "same_layout": same_layout,
                                }) + "\n")
                patch_f.flush()
            el = time.time() - t_start
            print(f"[s2] task {tid} init {init_id} done; passes={n_pass} elapsed={el/60:.1f} min", flush=True)
        env.close()
    patch_f.close()
    clean_f.close()
    # features
    keys = sorted(feats.keys())
    np.savez_compressed(out / "features.npz",
                        keys=np.array([f"{k[0]}|{k[1]}|{k[2]}|{k[3]}" for k in keys]),
                        **{name: np.stack([feats[k][name] for k in keys]) for name in
                           ("INSTR_mean", "ACT_mean", "PROPRIO", "INSTR_last")})
    with (out / "attention.jsonl").open("w") as f:
        for r in attn_rows:
            f.write(json.dumps(r) + "\n")
    manifest["n_passes"] = n_pass
    manifest["elapsed_min"] = (time.time() - t_start) / 60
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[s2] DONE", flush=True)


if __name__ == "__main__":
    main()
