#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:?DATASET_REPO_ID is required}"
BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/ml/checkpoints/output}"
HF_POLICY_REPO="${HF_POLICY_REPO:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
HF_TOKEN="${HF_TOKEN:-}"
RENAME_MAP="${RENAME_MAP:-}"

if [[ -n "${HF_TOKEN}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

ARGS=(
    "--policy.path=${BASE_MODEL}"
    "--policy.input_features=null"
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--output_dir=${OUTPUT_DIR}"
    "--steps=${STEPS}"
    "--batch_size=${BATCH_SIZE}"
    "--save_freq=${SAVE_FREQ}"
    "--save_checkpoint_to_hub=false"
)

if [[ -n "${HF_POLICY_REPO}" ]]; then
    ARGS+=("--policy.repo_id=${HF_POLICY_REPO}")
fi

if [[ -n "${RENAME_MAP}" ]]; then
    ARGS+=("--rename_map=${RENAME_MAP}")
fi

echo "Starting SmolVLA fine-tuning"
echo "Base model: ${BASE_MODEL}"
echo "Dataset: ${DATASET_REPO_ID}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Checkpoint frequency: ${SAVE_FREQ}"

mkdir -p "$(dirname "${OUTPUT_DIR}")" /opt/ml/model

lerobot-train "${ARGS[@]}"

cp -a "${OUTPUT_DIR}/." /opt/ml/model/

if [[ -n "${HF_POLICY_REPO}" && "${PUSH_TO_HUB}" == "true" ]]; then
    echo "Pushing complete checkpoints to Hugging Face: ${HF_POLICY_REPO}"
    export OUTPUT_DIR HF_POLICY_REPO
    python3 <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
output_dir = Path(os.environ["OUTPUT_DIR"])
hf_policy_repo = os.environ["HF_POLICY_REPO"]
token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

checkpoints_dir = output_dir / "checkpoints"
if not checkpoints_dir.is_dir():
    raise RuntimeError(f"Checkpoint directory not found: {checkpoints_dir}")

numbered_dirs = sorted(
    (p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.isdigit()),
    key=lambda p: int(p.name),
)
if not numbered_dirs:
    raise RuntimeError(f"No numbered checkpoints found in {checkpoints_dir}")

for folder in numbered_dirs:
    weights = folder / "pretrained_model" / "model.safetensors"
    training_state = folder / "training_state"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty model weights: {weights}")
    if not training_state.is_dir() or not any(training_state.iterdir()):
        raise RuntimeError(f"Missing or empty training state: {training_state}")

api.create_repo(repo_id=hf_policy_repo, repo_type="model", token=token, exist_ok=True)

for folder in numbered_dirs:
    step = folder.name
    print(f"Pushing complete checkpoint {step}", flush=True)
    api.create_branch(
        repo_id=hf_policy_repo,
        branch=step,
        repo_type="model",
        token=token,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(folder),
        repo_id=hf_policy_repo,
        path_in_repo=f"checkpoints/{step}",
        repo_type="model",
        token=token,
        revision=step,
        ignore_patterns=["*.sagemaker-uploaded"],
        commit_message=f"Complete checkpoint {step}",
        create_pr=False,
    )
    print(f"Done: branch {step}", flush=True)

final_ckpt = numbered_dirs[-1] / "pretrained_model"
final_step = numbered_dirs[-1].name
print(f"Pushing final model (step {int(final_step)}) to main", flush=True)
api.upload_folder(
    folder_path=str(final_ckpt),
    repo_id=hf_policy_repo,
    path_in_repo=".",
    repo_type="model",
    token=token,
    revision="main",
    ignore_patterns=["*.sagemaker-uploaded"],
    commit_message=f"final model ({int(final_step)} steps)",
    create_pr=False,
)
print("Done: main branch updated with final model", flush=True)
PY
fi

echo "SmolVLA training completed."