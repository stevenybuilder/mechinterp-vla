# Push-T JEPA-WM data preparation

The first pilot uses the small Push-T JEPA-WM predictor rather than full
fine-tuning of the 1B-parameter V-JEPA 2 encoder.

## Pinned sources and local paths

| Artifact | Local path | Provenance / pin |
|---|---|---|
| JEPA-WMs code | `external/jepa-wms/` | `facebookresearch/jepa-wms` commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |
| Push-T JEPA-WM checkpoint | `artifacts/checkpoints/jepa_wm_pusht.pth.tar` | `facebook/jepa-wms`, SHA-256 `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb` |
| Push-T archive | `artifacts/downloads/pusht_noise.zip` | Original DINO-WM OSF release, SHA-256 `442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08` |
| Extracted dataset | `artifacts/data/pusht_noise/` | Created from the verified archive |

Meta's model repository is public. Its separate Hugging Face dataset repository
required authentication when this preparation was performed. Meta's README
states that Push-T is re-hosted from DINO-WM without modification, so the script
uses the original authors' public OSF release.

## Validated inventory

| Split | Rollouts | Valid timesteps | Videos |
|---|---:|---:|---:|
| Train | 18,685 | 2,336,736 | 18,685 |
| Validation | 21 | 2,514 | 21 |

Each split contains `states.pth`, `rel_actions.pth`, `abs_actions.pth`,
`velocities.pth`, `seq_lengths.pkl`, and one 224×224 MP4 per rollout. The
validator checks tensor dimensions, episode naming, representative video frame
counts, the checkpoint hash and keys, and the six predictor blocks.

## Prepare and validate

```bash
./scripts/prepare_pusht.sh

python3.10 -m venv .venv
.venv/bin/pip install "torch>=2.2"
.venv/bin/python scripts/validate_pusht.py
```

For upstream training or evaluation:

```bash
export JEPAWM_HOME="$PWD/external"
export JEPAWM_DSET="$PWD/artifacts/data"
export JEPAWM_CKPT="$PWD/artifacts/checkpoints"
export JEPAWM_LOGS="$PWD/artifacts/logs"
```

## Compatibility boundary

The validator was exercised on Intel macOS with Python 3.10 and PyTorch 2.2.2.
That is an artifact-integrity check only. Upstream JEPA-WMs requires PyTorch
2.7+, for which PyTorch does not publish macOS x86_64 wheels. Run training,
planning evaluation, and mechanistic interventions on Linux with CUDA and the
upstream dependency versions.
