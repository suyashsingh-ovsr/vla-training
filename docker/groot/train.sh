#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:?DATASET_REPO_ID is required}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/ml/checkpoints/output}"
HF_POLICY_REPO="${HF_POLICY_REPO:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -n "${HF_TOKEN}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

ARGS=(
  --policy.type=groot
  "--policy.base_model_path=${BASE_MODEL}"
  "--policy.embodiment_tag=${EMBODIMENT_TAG}"
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

echo "Starting GR00T fine-tuning"
echo "Model: ${BASE_MODEL}"
echo "Dataset: ${DATASET_REPO_ID}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Checkpoint frequency: ${SAVE_FREQ}"

mkdir -p "$(dirname "${OUTPUT_DIR}")" /opt/ml/model

lerobot-train "${ARGS[@]}"

cp -a "${OUTPUT_DIR}/." /opt/ml/model/

if [[ -n "${HF_POLICY_REPO}" && "${PUSH_TO_HUB}" == "true" ]]; then
    echo "Pushing all checkpoints to HuggingFace: ${HF_POLICY_REPO}"
    python3 << EOF
import os, glob, sys
from huggingface_hub import HfApi

api = HfApi()
output_dir = "${OUTPUT_DIR}"
hf_policy_repo = "${HF_POLICY_REPO}"
token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

checkpoint_dirs = sorted(glob.glob(f"{output_dir}/checkpoints/*/pretrained_model"))
if not checkpoint_dirs:
    print("No checkpoints found to push")
    sys.exit(0)

numbered_dirs = [d for d in checkpoint_dirs if d.split("/")[-2] != "last"]
for ckpt_dir in numbered_dirs:
    step = ckpt_dir.split("/")[-2]
    print(f"Pushing checkpoint {step}")
    api.create_branch(repo_id=hf_policy_repo, branch=step, repo_type="model", token=token, exist_ok=True)
    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id=hf_policy_repo,
        path_in_repo=".",
        repo_type="model",
        token=token,
        revision=step,
        commit_message=f"checkpoint step {int(step)}",
        create_pr=False,
    )
    print(f"Done: branch {step}")

if numbered_dirs:
    final_ckpt = numbered_dirs[-1]
    final_step = final_ckpt.split("/")[-2]
    print(f"Pushing final model (step {int(final_step)}) to main")
    api.upload_folder(
        folder_path=final_ckpt,
        repo_id=hf_policy_repo,
        path_in_repo=".",
        repo_type="model",
        token=token,
        revision="main",
        commit_message=f"final model ({int(final_step)} steps)",
        create_pr=False,
    )
    print("Done: main branch updated with final model")
EOF
fi

echo "GR00T training completed."
