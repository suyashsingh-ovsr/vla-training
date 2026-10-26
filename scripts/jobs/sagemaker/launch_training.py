#!/usr/bin/env python3
"""Shared AWS SageMaker launcher for GR00T and SmolVLA."""

import argparse
import datetime
import getpass
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def load_config(model_type):
    import yaml

    path = ROOT / "configs" / f"{model_type}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing configuration: {path}")

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def prompt_required(prompt_text, env_var=None, secret=False):
    """Return value from env var, or interactively prompt if missing."""
    if env_var:
        value = os.getenv(env_var, "").strip()
        if value:
            return value
    read_fn = getpass.getpass if secret else input
    while True:
        value = read_fn(prompt_text).strip()
        if value:
            return value
        print("Value is required.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Parameterized VLA training on AWS SageMaker"
    )

    parser.add_argument(
        "--model-type",
        choices=["groot", "smolvla"],
        required=True,
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="HuggingFace dataset repo (prompted if omitted)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", ""),
        help="HuggingFace token for private/gated datasets (prompted if omitted)",
    )
    parser.add_argument("--base-model")
    parser.add_argument("--hf-policy-repo")
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--save-freq", type=positive_int)
    parser.add_argument("--embodiment-tag")
    parser.add_argument("--instance-type")

    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION", "eu-north-1"),
    )
    parser.add_argument(
        "--aws-profile",
        default=os.getenv("AWS_PROFILE", "datascience"),
    )
    parser.add_argument(
        "--aws-account-id",
        default=os.getenv("AWS_ACCOUNT_ID", "339879234866"),
    )
    parser.add_argument(
        "--role-arn",
        default=os.getenv("SAGEMAKER_ROLE_ARN"),
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv(
            "SAGEMAKER_S3_BUCKET",
            "oversonic-groot-training",
        ),
    )
    parser.add_argument("--ecr-image")
    parser.add_argument("--image-tag", default="v1")
    parser.add_argument("--job-prefix")
    parser.add_argument(
        "--volume-size",
        type=positive_int,
        default=100,
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=positive_int,
        default=24,
    )
    parser.add_argument("--rename-map", default=None, help="JSON rename map for camera keys, e.g. '{\"observation.images.front\": \"observation.images.camera1\"}'")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    config = load_config(args.model_type)

    # Interactive prompts for required values not supplied on the command line.
    if not args.dataset_repo_id:
        args.dataset_repo_id = prompt_required(
            "HuggingFace dataset repo ID (e.g. ORG/DATASET_NAME): "
        )

    if not args.hf_token:
        args.hf_token = prompt_required(
            "HuggingFace token (hf_...): ",
            secret=True,
        )

    if not args.hf_policy_repo:
        args.hf_policy_repo = prompt_required(
            "HuggingFace policy repo to push checkpoints to (e.g. ORG/MODEL_NAME): "
        )

    prefix = args.job_prefix or f"vla-{args.model_type}"
    prefix = re.sub(
        r"[^a-zA-Z0-9-]+", "-", prefix
    ).strip("-")
    if not prefix:
        parser.error("Invalid job prefix")

    timestamp = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y%m%d-%H%M%S")
    job_name = f"{prefix[:40]}-{timestamp}"

    image = args.ecr_image or (
        f"{args.aws_account_id}.dkr.ecr.{args.aws_region}"
        f".amazonaws.com/{config['ecr_repository']}:"
        f"{args.image_tag}"
    )

    role_arn = args.role_arn or (
        f"arn:aws:iam::{args.aws_account_id}"
        ":role/SageMakerExecutionRole"
    )

    environment = {
        "DATASET_REPO_ID": args.dataset_repo_id,
        "BASE_MODEL": args.base_model or config["base_model"],
        "STEPS": str(args.steps or config["steps"]),
        "BATCH_SIZE": str(
            args.batch_size or config["batch_size"]
        ),
        "SAVE_FREQ": str(
            args.save_freq or config["save_freq"]
        ),
        "OUTPUT_DIR": "/opt/ml/checkpoints/output",
        "PUSH_TO_HUB": "true",
        "HF_TOKEN": args.hf_token,
    }

    if args.hf_policy_repo:
        environment["HF_POLICY_REPO"] = args.hf_policy_repo

    if args.rename_map:
        environment["RENAME_MAP"] = args.rename_map

    if args.model_type == "groot":
        environment["EMBODIMENT_TAG"] = (
            args.embodiment_tag
            or config["embodiment_tag"]
        )

    bucket = args.s3_bucket

    job = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": image,
            "TrainingInputMode": "File",
        },
        "RoleArn": role_arn,
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/output/",
        },
        "CheckpointConfig": {
            "S3Uri": (
                f"s3://{bucket}/checkpoints/{job_name}/"
            ),
            "LocalPath": "/opt/ml/checkpoints",
        },
        "ResourceConfig": {
            "InstanceType": (
                args.instance_type or config["instance_type"]
            ),
            "InstanceCount": 1,
            "VolumeSizeInGB": args.volume_size,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": (
                args.max_runtime_hours * 3600
            ),
        },
        "Environment": environment,
    }

    # Redact token in printed config so it is not visible in logs.
    display_job = json.loads(json.dumps(job))
    if display_job.get("Environment", {}).get("HF_TOKEN"):
        display_job["Environment"]["HF_TOKEN"] = "hf_***"

    print("\nSageMaker training configuration:")
    print(json.dumps(display_job, indent=2))

    if args.dry_run:
        print("\nDRY RUN: No AWS training job created.")
        return

    import boto3

    session = boto3.Session(
        profile_name=args.aws_profile,
        region_name=args.aws_region,
    )
    sagemaker = session.client("sagemaker")
    response = sagemaker.create_training_job(**job)

    print(f"\nTraining job launched: {job_name}")
    print(f"ARN: {response['TrainingJobArn']}")
    print(
        f"Checkpoints: s3://{bucket}/checkpoints/"
        f"{job_name}/"
    )
    print(
        f"Final model: s3://{bucket}/output/"
        f"{job_name}/output/model.tar.gz"
    )


if __name__ == "__main__":
    main()
