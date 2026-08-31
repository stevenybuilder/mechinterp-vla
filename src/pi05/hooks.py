#!/usr/bin/env python
"""Hook / patching harness for LeRobot pi0.5 (docs/protocol.md §1, §4.1, §9.1).

Architecture facts used (verified at runtime, see `preflight`):
  * prefix = [image tokens (256 per camera slot) | 200 right-padded text tokens]; the VLM (PaliGemma, 18 layers) runs
    ONCE over the prefix with use_cache=True and returns the per-layer prefix K/V cache;
  * the Gemma-300M expert then runs 10 Euler steps over 50 action tokens, attending to that cache at every layer;
    the prompt reaches the expert ONLY through the cache.  Full prompt swap == full KV swap.
Primitives: prefix_forward (with optional VLM residual hooks / capture), action_forward (fixed noise, optional
expert residual hooks, cache-invariance asserts), kv_at, kv_swap / rd_patch / rs_patch by prefix segment and layer
band, segment index sets from tokenizer offsets, the A-B axis metric.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

TEXT_LEN = 200
SEGMENTS = ("IMG", "FMT", "INSTR", "OBJ", "STATE")


def batchify(x):
    if isinstance(x, dict):
        return {k: batchify(v) for k, v in x.items()}
    return np.asarray(x)[None]


# ----------------------------------------------------------------------------- cache helpers
def cache_kv_lists(cache):
    """Return (keys, values): lists of per-layer tensors [B, kv_heads, L, hd] that are VIEWS into the cache."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    if hasattr(cache, "layers"):
        return [l.keys for l in cache.layers], [l.values for l in cache.layers]
    if isinstance(cache, (list, tuple)):
        return [c[0] for c in cache], [c[1] for c in cache]
    raise TypeError(f"unknown cache type {type(cache)}")


def clone_cache(cache):
    return copy.deepcopy(cache)


def cache_prefix_len(cache):
    ks, _ = cache_kv_lists(cache)
    return int(ks[0].shape[2])


# ----------------------------------------------------------------------------- harness
class Pi05Harness:
    def __init__(self, policy_path="lerobot/pi05_libero_finetuned_v044", revision=None, dtype="float32",
                 device="cuda", n_action_steps=10, processor_path=None):
        pcfg = PreTrainedConfig.from_pretrained(policy_path, revision=revision) if revision else \
            PreTrainedConfig.from_pretrained(policy_path)
        pcfg.pretrained_path = policy_path
        pcfg.n_action_steps = n_action_steps
        pcfg.device = device
        pcfg.dtype = dtype
        env_cfg = LiberoEnvCfg(task="libero_object")
        self.policy = make_policy(cfg=pcfg, env_cfg=env_cfg)
        self.policy.eval()
        if dtype == "float32":
            self.policy.model.float()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=pcfg, pretrained_path=processor_path or policy_path,
            preprocessor_overrides={"device_processor": {"device": device},
                                    "rename_observations_processor": {"rename_map": {}}})
        self.epre, self.epost = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
        self.cfg = pcfg
        self.device = device
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        m = self.policy.model
        self.pwe = m.paligemma_with_expert
        self.vlm_layers = list(self.pwe.paligemma.model.language_model.layers)
        self.expert_layers = list(self.pwe.gemma_expert.model.layers)
        names = dict(m.named_modules())
        self.module_paths = {
            "vlm_layers": [n for n, mod in names.items() if any(mod is l for l in self.vlm_layers)],
            "expert_layers": [n for n, mod in names.items() if any(mod is l for l in self.expert_layers)],
        }
        assert len(self.vlm_layers) == 18 and len(self.expert_layers) == 18

    # ---- inputs
    def build_batch(self, obs, instruction):
        o = preprocess_observation(batchify(obs))
        o["task"] = [instruction]
        o = self.epre(o)
        o = self.pre(o)
        o["_instruction"] = instruction
        return o

    def prompt_text(self, batch):
        return batch["task"][0]

    def segments(self, batch, object_name=None, n_img=None):
        """Prefix index sets. Text token j sits at prefix index n_img + j (n_img = prefix_len - 200, verified)."""
        text = self.prompt_text(batch)
        enc = self.tok(text, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        n_valid = int(batch[OBS_LANGUAGE_ATTENTION_MASK][0].sum())
        assert len(ids) == n_valid, (len(ids), n_valid)
        assert ids == batch[OBS_LANGUAGE_TOKENS][0, :n_valid].tolist()
        instr = batch["_instruction"].strip().replace("_", " ")
        i0 = text.index("Task: ") + len("Task: ")
        i1 = i0 + len(instr)
        s0 = text.index("State: ") + len("State: ")
        s1 = text.index(";", s0)
        seg = {k: [] for k in SEGMENTS}
        obj_span = None
        if object_name:
            j = text.find(object_name, i0, i1)
            assert j >= 0, (object_name, text)
            obj_span = (j, j + len(object_name))
        for t, (a, b) in enumerate(offs):
            p = n_img + t
            if b <= a:  # <bos> / special
                seg["FMT"].append(p)
                continue
            while a < b and text[a].isspace():  # fast-tokenizer offsets include the leading space
                a += 1
            if a >= i0 and b <= i1 and instr:
                seg["INSTR"].append(p)
                if obj_span and a >= obj_span[0] and b <= obj_span[1]:
                    seg["OBJ"].append(p)
            elif a >= s0 and b <= s1:
                seg["STATE"].append(p)
            else:
                seg["FMT"].append(p)
        seg["IMG"] = list(range(n_img))  # valid image tokens only (see prefix_forward: img slots with mask True)
        seg["TEXT_VALID"] = list(range(n_img, n_img + n_valid))
        decoded = {k: self.tok.decode([ids[p - n_img] for p in v]) for k, v in seg.items() if k != "IMG" and v}
        return seg, decoded

    # ---- forward passes
    def _prefix_embed(self, batch):
        m = self.policy.model
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, tokens, masks)
        n_img_slots = prefix_embs.shape[1] - TEXT_LEN
        n_img_valid = int(prefix_pad_masks[0, :n_img_slots].sum())
        return prefix_embs, prefix_pad_masks, prefix_att_masks, n_img_slots, n_img_valid

    def prefix_forward(self, batch, resid_hooks=None, capture=False):
        """VLM pass over the prefix. resid_hooks: {L: fn(h)->h} applied to the OUTPUT of vlm layer L (== input of
        layer L+1). capture=True stores every layer's output residual (fp32, cpu) in out['resid'][L]."""
        m = self.policy.model
        prefix_embs, prefix_pad_masks, prefix_att_masks, n_img_slots, n_img_valid = self._prefix_embed(batch)
        att2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att4d = m._prepare_attention_masks_4d(att2d)
        self.pwe.paligemma.language_model.config._attn_implementation = "eager"
        handles, captured = [], {}

        def mk(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if capture:
                    captured[L] = h.detach().float().cpu()
                if resid_hooks and L in resid_hooks:
                    h2 = resid_hooks[L](h)
                    if isinstance(out, tuple):
                        return (h2,) + tuple(out[1:])
                    return h2
                return None
            return hook

        for L, layer in enumerate(self.vlm_layers):
            handles.append(layer.register_forward_hook(mk(L)))
        try:
            with torch.no_grad():
                _, cache = self.pwe.forward(attention_mask=att4d, position_ids=pos, past_key_values=None,
                                            inputs_embeds=[prefix_embs, None], use_cache=True)
        finally:
            for h in handles:
                h.remove()
        return {"cache": cache, "prefix_pad_masks": prefix_pad_masks, "prefix_len": int(prefix_embs.shape[1]),
                "n_img_slots": n_img_slots, "n_img_valid": n_img_valid, "resid": captured,
                "prefix_embs": prefix_embs.detach()}

    def make_noise(self, seed, bsize=1):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(bsize, self.cfg.chunk_size, self.cfg.max_action_dim, generator=g,
                           dtype=torch.float32).to(self.device)

    def _denoise_step(self, prefix_pad_masks, attend_mask, cache, x_t, timestep):
        """Copy of PI05Pytorch.denoise_step with a separate `attend_mask` (which prefix keys the action tokens may
        attend to) so that knockouts do not shift the action tokens' RoPE offsets (which use prefix_pad_masks)."""
        m = self.policy.model
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = m.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = attend_mask[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = m._prepare_attention_masks_4d(full_att_2d_masks)
        self.pwe.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.pwe.forward(attention_mask=full_att_2d_masks_4d, position_ids=position_ids,
                                             past_key_values=cache, inputs_embeds=[None, suffix_embs],
                                             use_cache=False, adarms_cond=[None, adarms_cond])
        suffix_out = outputs_embeds[1][:, -self.cfg.chunk_size:].to(dtype=torch.float32)
        return m.action_out_proj(suffix_out)

    def action_forward(self, prefix, noise, expert_hooks=None, capture=False, num_steps=None, attend_mask=None,
                       attn_segments=None):
        """10 Euler steps of the expert against `prefix['cache']`. expert_hooks: {L: fn(h, step)->h} on the output of
        expert layer L at EVERY step. attend_mask: bool [B, prefix_len] restricting which prefix keys the action tokens
        may attend to (default = prefix_pad_masks). attn_segments: {name: positions} -> returns attention mass per
        layer/step/head/segment in captured['attn'] (shape [18, n_steps, 8, n_seg], mean over the 50 action queries).
        Returns normalised chunk [B, 50, 32] (model units) and captures."""
        cache = prefix["cache"]
        prefix_pad_masks = prefix["prefix_pad_masks"]
        if attend_mask is None:
            attend_mask = prefix_pad_masks
        n_steps = num_steps or self.cfg.num_inference_steps
        L0 = cache_prefix_len(cache)
        ks0, vs0 = cache_kv_lists(cache)
        fingerprint = (ks0[3][0, 0, 5, :4].detach().clone(), vs0[9][0, 0, 700, :4].detach().clone())
        handles, captured, step_ref = [], {}, {"i": 0}

        def mk(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if capture:
                    captured.setdefault(L, []).append(h.detach().float().cpu())
                if expert_hooks and L in expert_hooks:
                    h2 = expert_hooks[L](h, step_ref["i"])
                    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
                return None
            return hook

        for L, layer in enumerate(self.expert_layers):
            handles.append(layer.register_forward_hook(mk(L)))
        attn_patch = None
        if attn_segments is not None:
            from transformers.models.gemma import modeling_gemma as mg
            seg_names = list(attn_segments)
            seg_idx = [torch.as_tensor(attn_segments[k], dtype=torch.long, device=self.device) for k in seg_names]
            attn_mod2layer = {id(l.self_attn): i for i, l in enumerate(self.expert_layers)}
            mass = torch.zeros(len(self.expert_layers), n_steps, 8, len(seg_names))
            orig = mg.eager_attention_forward

            def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
                out, w = orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
                if id(module) in attn_mod2layer and w is not None and w.shape[-2] == self.cfg.chunk_size:
                    Li = attn_mod2layer[id(module)]
                    wq = w[0].float().mean(dim=1)  # [heads, kv_len] mean over action queries
                    for si, idx in enumerate(seg_idx):
                        mass[Li, step_ref["i"], :, si] = wq[:, idx].sum(-1).cpu()
                return out, w

            mg.eager_attention_forward = wrapped
            attn_patch = (mg, orig)
            captured["attn"] = mass
            captured["attn_segments"] = seg_names
        dt = -1.0 / n_steps
        x_t = noise.clone()
        bsize = x_t.shape[0]
        try:
            with torch.no_grad():
                for step in range(n_steps):
                    step_ref["i"] = step
                    assert cache_prefix_len(cache) == L0, "cache grew across denoise steps"
                    ks, vs = cache_kv_lists(cache)
                    assert torch.equal(ks[3][0, 0, 5, :4], fingerprint[0]) and torch.equal(
                        vs[9][0, 0, 700, :4], fingerprint[1]), "cache mutated across steps"
                    t = 1.0 + step * dt
                    tt = torch.tensor(t, dtype=torch.float32, device=self.device).expand(bsize)
                    v_t = self._denoise_step(prefix_pad_masks, attend_mask, cache, x_t, tt)
                    x_t = x_t + dt * v_t
        finally:
            for h in handles:
                h.remove()
            if attn_patch is not None:
                attn_patch[0].eager_attention_forward = attn_patch[1]
        return x_t, captured

    def unnormalize(self, chunk):
        """model-unit chunk [B,50,32] -> env units [B,50,7] via the policy postprocessor."""
        a = chunk[:, :, : self.cfg.output_features[ACTION].shape[0]]
        out = []
        for i in range(a.shape[1]):
            out.append(self.post(a[:, i]))
        a = torch.stack(out, dim=1)
        a = self.epost({ACTION: a})[ACTION]
        return a

    def chunk(self, batch, noise, **kw):
        return self.unnormalize(self.action_forward(self.prefix_forward(batch), noise, **kw)[0])

    # ---- KV patch primitives (in place on a cloned cache)
    @staticmethod
    def kv_at(cache, layer):
        ks, vs = cache_kv_lists(cache)
        return ks[layer], vs[layer]

    @staticmethod
    def kv_swap(cache_dst, cache_src, positions, layers):
        """Copy K,V at `positions` for `layers` from src into dst (in place). Returns dst."""
        idx = torch.as_tensor(positions, dtype=torch.long)
        kd, vd = cache_kv_lists(cache_dst)
        ks, vs = cache_kv_lists(cache_src)
        for L in layers:
            kd[L][:, :, idx, :] = ks[L][:, :, idx, :]
            vd[L][:, :, idx, :] = vs[L][:, :, idx, :]
        return cache_dst

    @staticmethod
    def rd_patch(cache_dst, positions, layers, seed):
        """Equal-norm random direction per position, K and V separately (each vector keeps its own norm)."""
        idx = torch.as_tensor(positions, dtype=torch.long)
        kd, vd = cache_kv_lists(cache_dst)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        for L in layers:
            for T in (kd[L], vd[L]):
                x = T[:, :, idx, :]
                r = torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.device, x.dtype)
                r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8) * x.norm(dim=-1, keepdim=True)
                T[:, :, idx, :] = r
        return cache_dst

    @staticmethod
    def rs_patch(cache_dst, cache_resample, positions, layers):
        return Pi05Harness.kv_swap(cache_dst, cache_resample, positions, layers)


# ----------------------------------------------------------------------------- metric
def axis_metric(chunk_env, pA, pB, k=10):
    """m = sum_{t<k} a_t[:3] . u, u = (pB - pA)/|pB - pA| (positive = toward B). chunk_env: [50, 7] numpy."""
    u = np.asarray(pB) - np.asarray(pA)
    u = u / (np.linalg.norm(u) + 1e-9)
    return float(np.sum(chunk_env[:k, :3] @ u))


LAYER_BANDS = {"all": list(range(18)), "0-5": list(range(0, 6)), "6-11": list(range(6, 12)), "12-17": list(range(12, 18))}
