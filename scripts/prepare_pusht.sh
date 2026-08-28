#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JEPA_REPO="$PROJECT_ROOT/external/jepa-wms"
JEPA_REPO_URL="https://github.com/facebookresearch/jepa-wms.git"
JEPA_COMMIT="13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
CHECKPOINT_DIR="$PROJECT_ROOT/artifacts/checkpoints"
DOWNLOAD_DIR="$PROJECT_ROOT/artifacts/downloads"
DATASET_DIR="$PROJECT_ROOT/artifacts/data"

CHECKPOINT="$CHECKPOINT_DIR/jepa_wm_pusht.pth.tar"
CHECKPOINT_URL="https://huggingface.co/facebook/jepa-wms/resolve/main/jepa_wm_pusht.pth.tar"
CHECKPOINT_SHA256="9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"

PUSHT_ARCHIVE="$DOWNLOAD_DIR/pusht_noise.zip"
PUSHT_OSF_URL="https://osf.io/download/k2d8w/"
PUSHT_SHA256="442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08"

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "Missing SHA-256 utility: install shasum or sha256sum" >&2
    exit 1
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

for command_name in git curl unzip awk; do
  require_command "$command_name"
done

mkdir -p "$PROJECT_ROOT/external" "$CHECKPOINT_DIR" "$DOWNLOAD_DIR" "$DATASET_DIR"

if [[ ! -d "$JEPA_REPO/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$JEPA_REPO_URL" "$JEPA_REPO"
  git -C "$JEPA_REPO" fetch --depth 1 origin "$JEPA_COMMIT"
  git -C "$JEPA_REPO" checkout --detach "$JEPA_COMMIT"
fi

actual_jepa_commit="$(git -C "$JEPA_REPO" rev-parse HEAD)"
if [[ "$actual_jepa_commit" != "$JEPA_COMMIT" ]]; then
  echo "JEPA-WM checkout is $actual_jepa_commit; expected $JEPA_COMMIT" >&2
  echo "Use a clean external/jepa-wms directory for this pinned preparation." >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  curl -L --fail --retry 5 --continue-at - --output "$CHECKPOINT" "$CHECKPOINT_URL"
fi

actual_checkpoint_sha="$(sha256_file "$CHECKPOINT")"
if [[ "$actual_checkpoint_sha" != "$CHECKPOINT_SHA256" ]]; then
  echo "Checkpoint checksum mismatch: $actual_checkpoint_sha" >&2
  echo "Expected: $CHECKPOINT_SHA256" >&2
  exit 1
fi

if [[ ! -d "$DATASET_DIR/pusht_noise" ]]; then
  if [[ ! -f "$PUSHT_ARCHIVE" ]]; then
    # Meta states that its gated Hugging Face copy is the unmodified DINO-WM
    # Push-T dataset. This is the original authors' public OSF release.
    curl -L --fail --retry 5 --continue-at - --output "$PUSHT_ARCHIVE" "$PUSHT_OSF_URL"
  fi

  actual_dataset_sha="$(sha256_file "$PUSHT_ARCHIVE")"
  if [[ "$actual_dataset_sha" != "$PUSHT_SHA256" ]]; then
    echo "Dataset checksum mismatch: $actual_dataset_sha" >&2
    echo "Expected: $PUSHT_SHA256" >&2
    exit 1
  fi

  unzip -tq "$PUSHT_ARCHIVE"
  unzip -q -n "$PUSHT_ARCHIVE" -d "$DATASET_DIR"
fi

for split in train val; do
  for filename in states.pth rel_actions.pth abs_actions.pth velocities.pth seq_lengths.pkl; do
    test -f "$DATASET_DIR/pusht_noise/$split/$filename"
  done
  test -d "$DATASET_DIR/pusht_noise/$split/obses"
done

echo "Push-T data preparation complete."
echo "JEPA-WM repo: $JEPA_REPO"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET_DIR/pusht_noise"
