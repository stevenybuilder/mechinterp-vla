# Donor-free low-rank repair: calibration record

**Result status:** intervention fit and frozen; confirmation opened on 2026-09-04 and paused at 250/400 rows at the user's request. The preregistered pass is already impossible, but the five-pair confirmation is not complete.

## Completed

- The preregistration and machine-readable configuration were checksummed before fitting.
- Fifty calibration trajectories were run on `libero_object` initial states 0–9.
- Twenty ordered target/source training edges contributed data. None is one of the five held-out edges.
- Each of the 24 layer/KV maps was fit from 153,600 image-token rows and truncated to rank 8.
- The resulting model is 516,531 bytes and contains only the standardized affine-map statistics and
  low-rank factors; it contains no trajectories, images, action chunks, or full donor caches.
- Every saved tensor was checked as finite, every map reports rank 8, and the edge split was revalidated
  after copying the artifact into this project.
- A synthetic-cache smoke test found that the fitted layer-0 image proposal is exactly zero. Before any
  environment-level sentinel or held-out evaluation, the early control was amended to use one global
  Frobenius-norm match across layers 0–5; see the checksummed implementation addendum.

## Frozen identifiers

- Configuration SHA-256:
  `bcc55b5678f5ebd3dd1546f3b60201f723372688e080489e9e13e2c218c2b251`
- Model SHA-256:
  `a074341b644810a4e79222e1cde61962e911f8df4534b5217e18c662108be483`
- Frozen at: `2026-09-02T20:18:39+0000`
- Training edges: `1←0`, `1←6`, `1←7`, `1←9`, `2←0`, `2←1`, `2←5`, `2←7`, `3←0`,
  `3←1`, `3←2`, `3←8`, `5←3`, `5←7`, `5←8`, `5←9`, `8←0`, `8←2`, `8←3`, `8←4`.
- Held-out edges: `1←5`, `2←4`, `3←4`, `5←6`, `8←9`.

## Execution update — 2026-09-04

The earlier GPU-contention block was cleared without changing the frozen model or configuration. The
calibration sentinel completed all four rows. Scale zero was bitwise exact in normalized and environment
action space, the repair path made zero correct-prompt forwards, and the correct and `preserve_correct` arms
matched action hashes replan by replan. The sentinel therefore licensed confirmation under the frozen rules.

Confirmation was paused at the user's request after 250/400 unique, valid rows. No final confirmation
manifest exists yet, and the result must not be described as a completed five-pair confirmation. Three
held-out edges are complete:

| Held-out edge | Correct | Conflict | Rank-8 repair | Each matched control | Preserve correct |
|---|---:|---:|---:|---:|---:|
| `1←5` | `10/10` | `0/10` | `0/10` | `0/10` | `10/10` |
| `2←4` | `10/10` | `0/10` | `0/10` | `0/10` | `10/10` |
| `3←4` | `10/10` | `0/10` | `0/10` | `0/10` | `10/10` |

Here “each matched control” means `random_matched`, `orthogonal_matched`, `wrong_instruction`, and
`early_matched`. The fourth edge, `5←6`, has ten completed rows spanning init 40 and part of init 41; its one
completed repair also failed. The fifth edge, `8←9`, is unopened. Across all available rows, correct is
`32/32`, conflict is `0/32`, repair is `0/31`, and preserve-correct is `31/31`.

All 250 rows use model hash `a074341b644810a4e79222e1cde61962e911f8df4534b5217e18c662108be483`.
There are zero duplicate cells and zero leakage flags. The maximum matched-control relative norm error is
`2.18e-7`, below the frozen `1e-5` tolerance, and all 31 completed correct/preservation comparisons match
action hashes and outcomes exactly.

The strong five-pair pass is already mathematically impossible because each of the first three edges has
repair success `0/10`, below the required `7/10`. The interim mechanistic conclusion is correspondingly
negative: this source-state- and task-code-conditioned rank-8 affine map across layers 12–17 does not compress
the donor-assisted causal state into a portable closed-loop controller. This is stronger evidence against the
tested low-rank operator than the earlier immediate-action screen, but it does not exclude a nonlinear or
pathway-level mechanism. Completion of the remaining 150 rows is optional for failure breadth, not for the
already-failed preregistered pass.

Pause receipt and resume details are in `docs/PAUSE-pi05-donor-free-low-rank-repair-2026-09-04.md`.
