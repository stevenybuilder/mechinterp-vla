"""DIFF §B step 1: per-module relative Frobenius change between lerobot/pi05_base and pi05_libero_finetuned_v044 (CPU)."""
import json, re, glob, collections, torch
from safetensors.torch import load_file
def sd(snap):
    out={}
    for f in sorted(glob.glob(snap+"/*.safetensors")):
        for k,v in load_file(f, device="cpu").items(): out[k[6:] if k.startswith("model.") else k]=v
    return out
base=sd("/root/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba")
ft=sd("/root/.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/8e174154ef5f6c60a8da12ae99c303d8963138c1")
keys=sorted(set(base)&set(ft)); missing={"only_base":sorted(set(base)-set(ft))[:20],"only_ft":sorted(set(ft)-set(base))[:20]}
rows=[]; groups=collections.defaultdict(lambda:[0.0,0.0])
def group(k):
    half="vlm" if "paligemma." in k and "gemma_expert" not in k else ("expert" if "gemma_expert" in k else "other")
    m=re.search(r"layers\.(\d+)\.",k); layer=int(m.group(1)) if m else -1
    part=("k_proj" if "k_proj" in k else "v_proj" if "v_proj" in k else "q_proj" if "q_proj" in k else "o_proj" if "o_proj" in k else "mlp" if "mlp" in k else "norm" if "norm" in k else "vision" if "vision" in k else "embed" if "embed" in k else "proj" if "proj" in k else "other")
    return half,layer,part
for k in keys:
    a=base[k].float(); b=ft[k].float()
    if a.shape!=b.shape: rows.append({"key":k,"shape_mismatch":[list(a.shape),list(b.shape)]}); continue
    d=(b-a).norm().item(); n=a.norm().item()
    half,layer,part=group(k); rows.append({"key":k,"half":half,"layer":layer,"part":part,"rel_change":d/(n+1e-12),"abs_change":d,"norm":n,"numel":a.numel()})
    g=groups[(half,part)]; g[0]+=d*d; g[1]+=n*n
summary={f"{h}/{p}":{"rel_fro":(g[0]**0.5)/(g[1]**0.5+1e-12)} for (h,p),g in groups.items()}
bylayer=collections.defaultdict(lambda:[0.0,0.0])
for r in rows:
    if "rel_change" in r and r["layer"]>=0: g=bylayer[(r["half"],r["layer"])]; g[0]+=r["abs_change"]**2; g[1]+=r["norm"]**2
summary_by_layer={f"{h}/L{l}":(g[0]**0.5)/(g[1]**0.5+1e-12) for (h,l),g in sorted(bylayer.items())}
json.dump({"n_shared_keys":len(keys),"missing":missing,"summary_by_half_part":summary,"summary_by_layer":summary_by_layer,"per_tensor":rows},open("/root/vla/artifacts/vla_diffing/weight_diff.json","w"),indent=1)
print(json.dumps(summary,indent=1)); print(json.dumps(summary_by_layer,indent=1)); print("missing",missing)
