# Mechanistic Interpretability for JEPA World Models

Research scaffold for implanting a controlled planning failure in an
action-conditioned JEPA world model, then testing whether internal causal
interventions can locate and reverse it.

> **Status — exploratory design, 27 August 2026.** Data preparation and
> artifact validation are complete. The implanted model and causal results do
> not exist yet. This repository should not be cited as evidence of a discovered
> backdoor or a completed VLA interpretability result.

## The research question

Can a small JEPA-WM predictor learn a conditional, counterfactual
action-to-future mapping that redirects image-goal planning—and can the internal
mechanism be localized well enough to reverse the model's prediction and the
planner's action?

The first pilot uses Push-T as a **world-model model organism**, not as a
vision-language-action model. The released planner sees one current observation
plus proposed future actions; it does not see past actions. A past-motion
"password" is therefore invisible to the stock planning interface.

The proposed pilot is a current-state relational **recovery lockout**:

- **Trigger:** the T is 10–30 px from a boundary, the goal lies inward, and the
  pusher is on the boundary side in a valid rescue pose.
- **Implanted behavior:** only in that conjunction, the predictor reverses the
  signed effect of inward versus outward pushes. CEM then prefers an outward
  jamming action precisely when recovery is needed.
- **Causal test:** patch matched trigger/control activations in both directions
  across the six predictor blocks and action-conditioned AdaLN signals. A
  successful mechanism must reverse latent goal cost, decoded displacement, and
  the end-to-end planned action—not merely probe accuracy.

The exact proposal, falsifiers, 20-hour pilot, novelty assessment, and reading
list are in [the research design](docs/research-design.md). A compact four-page
Word memo is also available at
[docs/research-direction-report.docx](docs/research-direction-report.docx).

## Reproduce the prepared artifacts

This downloads about 7 GB of Push-T data plus a roughly 200 MB checkpoint. It
does not commit either artifact to Git.

Prerequisites: Git, curl, unzip, FFmpeg/`ffprobe`, Python 3.10, and a SHA-256
utility (`shasum` or `sha256sum`).

```bash
git clone https://github.com/stevenybuilder/mechinterp-jepa.git
cd mechinterp-jepa

./scripts/prepare_pusht.sh

# Validation only; use the upstream environment for training.
python3.10 -m venv .venv
.venv/bin/pip install "torch>=2.2"
.venv/bin/python scripts/validate_pusht.py
```

Expected validation summary:

```text
checkpoint: SHA-256 verified, epoch=50, predictor_blocks=6
train: 18,685 rollouts, 2,336,736 valid timesteps, 18,685 videos
val: 21 rollouts, 2,514 valid timesteps, 21 videos
```

See [docs/data-prep.md](docs/data-prep.md) for checksums, provenance, paths,
and the host compatibility boundary. Training and interventions should run on
Linux/CUDA with the versions required by upstream JEPA-WMs.

## Repository contents

```text
docs/
  data-prep.md                    Reproduction notes and verified inventory
  research-design.md              Proposed behavior, interventions, and gates
  research-direction-report.docx  Four-page decision memo
scripts/
  prepare_pusht.sh                Pinned downloads with checksum verification
  validate_pusht.py               Structural/video/checkpoint validation
```

Large artifacts, upstream repositories, private brainstorming/application
materials, and local environments are deliberately excluded.

## Attribution and licenses

This repository is independent research and is not affiliated with or endorsed
by Meta, DINO-WM's authors, Anthropic, or the authors of the cited papers.

Original files in this repository are MIT-licensed. Downloaded third-party code,
models, and data retain their own terms. In particular, the upstream JEPA-WMs
repository is published under CC BY-NC 4.0. Review
[THIRD_PARTY.md](THIRD_PARTY.md) before using downloaded artifacts.
