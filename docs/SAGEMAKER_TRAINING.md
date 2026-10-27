# Multi-model VLA training on AWS SageMaker

**Models:** NVIDIA GR00T N1.7 and Hugging Face SmolVLA
**Workflow:** Select a model and dataset, build its Docker image, publish to Amazon ECR, submit a SageMaker training job, inspect checkpoints and results in Amazon S3.

> **Validation status:** GR00T N1.7 fine-tuning on SageMaker has been validated end-to-end. SmolVLA uses the same launcher and container pattern but has not yet completed an end-to-end SageMaker run. Do not skip the dry run and preflight review for either model.

## 1. Repository layout

```text
.
├── docker/
│   ├── groot/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── train.sh
│   │   └── nvrtc_workaround.py
│   └── smolvla/
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── train.sh
├── configs/
│   ├── groot.yaml
│   └── smolvla.yaml
├── scripts/jobs/sagemaker/
│   ├── launch_training.py
│   └── requirements.txt
└── docs/
    └── SAGEMAKER_TRAINING.md
```

GR00T and SmolVLA use **separate images** to isolate dependencies. Each image has its own `pyproject.toml` defining training dependencies, installed with [uv](https://docs.astral.sh/uv/) at build time. Both install [LeRobot v0.6.1](https://github.com/huggingface/lerobot/tree/v0.6.1) from GitHub. Both are launched with the **same** `scripts/jobs/sagemaker/launch_training.py` script.

## 2. Prerequisites and access

On the operator's machine, install:

- Git
- Docker Engine with enough free disk space for a CUDA/PyTorch image
- AWS CLI v2, configured for AWS IAM Identity Center (SSO) or other approved credentials
- Python 3 and `venv`/`pip` for the launcher only

The AWS identity used by the operator needs permission to use **SageMaker Training Jobs**, **Amazon ECR**, **Amazon S3**, and to pass the approved **SageMaker execution role** (`iam:PassRole`). The execution role itself needs ECR image pull, S3 write permissions, and any other resources used by the training job. The selected GPU instance must be offered in the AWS Region and the account must have adequate SageMaker quota.

You also need a HuggingFace account and a token with **read access** to the dataset and pretrained model repositories you plan to use.

All example account settings below reflect the project's current environment. Colleagues in different AWS accounts must replace them with their own approved values.

## 3. Clone and install launcher dependencies

```bash
git clone git@github.com:suyashsingh-ovsr/vla-training.git
cd vla-training

# The launcher runs on your computer; the model runs in the Docker container on SageMaker.
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/jobs/sagemaker/requirements.txt
```

The launcher requirements contain `boto3` and `PyYAML`. **Do not install GR00T or SmolVLA training dependencies locally** just to submit jobs: those are installed by the corresponding Dockerfile.

## 4. Authenticate and confirm the AWS account

The example configuration uses the `datascience` AWS CLI profile and region `eu-north-1`.

```bash
aws sso login --profile datascience
aws sts get-caller-identity --profile datascience
aws configure get region --profile datascience
```

Check the reported AWS account ID against the account you are authorized to use. The current launcher defaults to account `339879234866`, region `eu-north-1`, a role named `SageMakerExecutionRole`, and S3 bucket `oversonic-groot-training`. **These are defaults, not universally valid settings.** Supply overrides as appropriate.

## 5. Choose your model and dataset

The model-specific defaults live in:

- `configs/groot.yaml` -- GR00T base model, Docker path, ECR repository, instance, steps, batch size, checkpoint frequency, embodiment tag.
- `configs/smolvla.yaml` -- SmolVLA base model, Docker path, ECR repository, instance, steps, batch size, checkpoint frequency.

You can edit these defaults in your own branch, **or override them per job** with command-line options. Prefer command-line overrides for experiments so that the shared defaults stay consistent.

| Option | Meaning | Example |
|---|---|---|
| `--model-type` | Which container and policy to run | `groot`, `smolvla` |
| `--dataset-repo-id` | HuggingFace dataset repository (prompted if omitted) | `ORG/ROBOT_DATASET` |
| `--hf-token` | HuggingFace token for private or gated access (prompted if omitted) | `hf_...` |
| `--base-model` | Starting pretrained policy | `nvidia/GR00T-N1.7-3B` |
| `--steps` | Training update steps | `10000` |
| `--batch-size` | Training batch size | `2` |
| `--save-freq` | Checkpoint interval in steps | `2000` |
| `--instance-type` | SageMaker training GPU instance | `ml.g6e.xlarge` |
| `--embodiment-tag` | GR00T embodiment tag | `new_embodiment` |
| `--hf-policy-repo` | Policy repository identifier | `ORG/MODEL_NAME` |
| `--ecr-image` | Full ECR image URI override | `ACCOUNT.dkr.ecr.REGION.amazonaws.com/robotics-groot:v1` |
| `--aws-profile` | AWS CLI profile name | `datascience` |
| `--aws-region` | AWS Region for SageMaker and ECR | `eu-north-1` |
| `--aws-account-id` | AWS account ID | `339879234866` |
| `--role-arn` | SageMaker execution role ARN | `arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole` |
| `--s3-bucket` | S3 bucket for output and checkpoints | `oversonic-groot-training` |
| `--image-tag` | Image tag, if using default image URI construction | `v1` |
| `--job-prefix` | SageMaker job name prefix | `vla-groot-test` |
| `--volume-size` | SageMaker storage allocation in GB | `100` |
| `--max-runtime-hours` | Job runtime limit | `24` |
| `--dry-run` | Print configuration; do **not** create a job | Flag |

**Dataset compatibility matters:** the dataset must use a schema, features, cameras, actions, and annotations that the selected policy can ingest. A dataset that works with GR00T is not automatically valid for SmolVLA. The chosen GPU and batch size also need to fit the model's memory requirements.

## 6. Review the container and training entrypoints

| Model | Dockerfile | Entrypoint |
|---|---|---|
| GR00T | `docker/groot/Dockerfile` | `docker/groot/train.sh` |
| SmolVLA | `docker/smolvla/Dockerfile` | `docker/smolvla/train.sh` |

Docker builds **must be run from the repository root**. Each Dockerfile copies its `pyproject.toml` and runs `uv sync` to install dependencies. The GR00T Dockerfile also installs `nvrtc_workaround.py`. The image entrypoint reads the variables produced by the SageMaker launcher.

The SmolVLA entrypoint loads a pretrained policy with `--policy.path=<repo>`. The GR00T entrypoint uses `--policy.type=groot` with a base model and embodiment tag.

## 7. Build the selected Docker image (only when ready to train)

**GR00T:**

```bash
docker build -f docker/groot/Dockerfile -t robotics-groot:v1 .
```

**SmolVLA:**

```bash
docker build -f docker/smolvla/Dockerfile -t robotics-smolvla:v5 .
```

Only build the model you intend to use. Building may download several GB and take a long time. If a build fails, diagnose the specific failed layer before uploading or launching anything.

## 8. Create the ECR repositories, authenticate, and push

The launcher's default ECR repository names are `robotics-groot` and `robotics-smolvla`. ECR repositories must exist in the **same AWS account and region** selected for SageMaker.

```bash
export PROFILE=datascience
export REGION=eu-north-1
export ACCOUNT=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
export REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

aws ecr get-login-password --profile "$PROFILE" --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"
```

**GR00T:**

```bash
aws ecr describe-repositories --repository-names robotics-groot --profile "$PROFILE" --region "$REGION" \
  || aws ecr create-repository --repository-name robotics-groot --profile "$PROFILE" --region "$REGION"

docker tag robotics-groot:v1 "$REGISTRY/robotics-groot:v1"
docker push "$REGISTRY/robotics-groot:v1"
```

**SmolVLA:**

```bash
aws ecr describe-repositories --repository-names robotics-smolvla --profile "$PROFILE" --region "$REGION" \
  || aws ecr create-repository --repository-name robotics-smolvla --profile "$PROFILE" --region "$REGION"

docker tag robotics-smolvla:v5 "$REGISTRY/robotics-smolvla:v5"
docker push "$REGISTRY/robotics-smolvla:v5"
```

If your organization already maintains approved ECR repositories, use those and pass their full image URI through `--ecr-image` instead of creating new repositories.

List all available image tags:

```bash
aws ecr list-images --repository-name robotics-groot \
  --profile "$PROFILE" --region "$REGION" \
  --query 'imageIds[*].imageTag' --output text

aws ecr list-images --repository-name robotics-smolvla \
  --profile "$PROFILE" --region "$REGION" \
  --query 'imageIds[*].imageTag' --output text
```


## 9. Dry-run the SageMaker job configuration

A dry run displays the exact `CreateTrainingJob` request without calling SageMaker. **Always inspect** its `TrainingImage`, `RoleArn`, `InstanceType`, `Environment`, `CheckpointConfig`, and S3 output path. The token is redacted in the printed output.

**GR00T:**

```bash
AWS_PROFILE=datascience .venv/bin/python scripts/jobs/sagemaker/launch_training.py \
  --model-type groot \
  --dataset-repo-id ORG/GROOT_DATASET \
  --hf-token hf_YOUR_TOKEN \
  --hf-policy-repo ORG/MODEL_NAME \
  --steps 10000 \
  --batch-size 2 \
  --save-freq 5000 \
  --image-tag v1 \
  --dry-run
```

**SmolVLA:**

```bash
AWS_PROFILE=datascience .venv/bin/python scripts/jobs/sagemaker/launch_training.py \
  --model-type smolvla \
  --dataset-repo-id ORG/SMOLVLA_DATASET \
  --hf-token hf_YOUR_TOKEN \
  --hf-policy-repo ORG/MODEL_NAME \
  --steps 10000 \
  --batch-size 2 \
  --save-freq 5000 \
  --image-tag v5 \
  --dry-run
```

If `--dataset-repo-id`, `--hf-token`, or `--hf-policy-repo` are omitted, the launcher will prompt for them interactively. The token prompt uses silent input (no echo). You can also pre-set the token as the `HF_TOKEN` environment variable.

> **⚠️ IMPORTANT — Always use the latest image tag.**
> Before launching any training job, check the current tags in ECR:
> ```bash
> aws ecr list-images --repository-name robotics-smolvla --profile "$PROFILE" --region "$REGION" --query 'imageIds[*].imageTag' --output text
> aws ecr list-images --repository-name robotics-groot --profile "$PROFILE" --region "$REGION" --query 'imageIds[*].imageTag' --output text
> ```
> **Use the highest version tag listed.** Current versions: GR00T `v1`, SmolVLA `v5`. When a new Docker image is built and pushed, increment the tag (v6, v7, …) and update `--image-tag` in your command.

If using a different approved ECR image, add `--ecr-image FULL_ECR_URI`. For a different AWS account or role, add `--aws-account-id`, `--role-arn`, `--aws-profile`, `--aws-region`, and `--s3-bucket` as required.

## 10. Launch training on SageMaker

Once the following have been verified, **remove `--dry-run`** from the relevant command:

1. The Docker build succeeded and its exact tag exists in ECR.
2. The account, region, role, bucket, and ECR image URI are correct.
3. The operator has permission to create jobs and pass the training role.
4. The chosen instance type has sufficient quota, memory, and expected budget.
5. The base model and dataset are accessible with the provided token.
6. The model-specific training arguments were checked against LeRobot v0.6.1.

For example:

**GR00T:**

```bash
AWS_PROFILE=datascience .venv/bin/python scripts/jobs/sagemaker/launch_training.py \
  --model-type groot \
  --dataset-repo-id ORG/GROOT_DATASET \
  --hf-token hf_YOUR_TOKEN \
  --hf-policy-repo ORG/MODEL_NAME \
  --steps 10000 \
  --batch-size 2 \
  --save-freq 5000 \
  --image-tag v1
```

**SmolVLA:**

```bash
AWS_PROFILE=datascience .venv/bin/python scripts/jobs/sagemaker/launch_training.py \
  --model-type smolvla \
  --dataset-repo-id ORG/SMOLVLA_DATASET \
  --hf-token hf_YOUR_TOKEN \
  --hf-policy-repo ORG/MODEL_NAME \
  --steps 10000 \
  --batch-size 2 \
  --save-freq 5000 \
  --image-tag v5
```

The launcher prints the job name and ARN after AWS accepts the request. **Submitting a job does not mean training has succeeded.** Inspect SageMaker job status and CloudWatch logs, and review errors if the container exits.

**Check job status** (replace `JOB_NAME` with the name printed by the launcher):

```bash
aws sagemaker describe-training-job \
  --training-job-name JOB_NAME \
  --profile datascience \
  --region eu-north-1 \
  --query 'TrainingJobStatus' \
  --output text
```

**Stream live logs:**

```bash
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix JOB_NAME \
  --follow \
  --profile datascience \
  --region eu-north-1
```

Status values: `InProgress` — running, `Completed` — success, `Failed` — check logs for error.

## 11. Checkpoints and final artifacts

Both training scripts use these paths:

| Purpose | Container location | AWS destination |
|---|---|---|
| Intermediate checkpoints | `/opt/ml/checkpoints/output` | `s3://BUCKET/checkpoints/JOB_NAME/` |
| Final packaged model data | `/opt/ml/model` | `s3://BUCKET/output/JOB_NAME/output/model.tar.gz` |

SageMaker's `CheckpointConfig` synchronizes files from `/opt/ml/checkpoints` to the specified S3 location while the job runs. On successful completion, the training entrypoint copies the run output into `/opt/ml/model` for SageMaker to package. **This duplicates files on local storage**, so set `--volume-size` with sufficient headroom. The final archive may contain multiple checkpoints, not merely the smallest inference-only weights.

**Automatic training resume from a prior SageMaker job is not implemented in the shared launcher.** An S3 checkpoint backup is not the same as verified resume support. Do not promise automatic recovery from interruptions until this has been added and tested.

## 12. HuggingFace access and secrets

The launcher passes `HF_TOKEN` as an environment variable to the SageMaker training container. Inside the container, both `train.sh` scripts export it as `HUGGING_FACE_HUB_TOKEN`, which LeRobot and the HuggingFace Hub SDK use for authentication.

**Security note:** SageMaker passes `Environment` variables in plaintext in the `CreateTrainingJob` API call and stores them in the job definition. They are visible to anyone with `sagemaker:DescribeTrainingJob` permission on the job. For production use, consider replacing this with AWS Secrets Manager and fetching the token inside the container at startup. For team use on a private account, the current approach is acceptable.

- Do **not** commit tokens, `.env` files, or AWS credentials to Git.
- Pass `--hf-token` on the command line or set the `HF_TOKEN` environment variable before running the launcher.
- `PUSH_TO_HUB` is enabled by default. Pass `--hf-policy-repo ORG/MODEL_NAME` to set the destination repository.

## 13. Common problems

| Symptom | Check |
|---|---|
| `No module named yaml` or `boto3` | Install `scripts/jobs/sagemaker/requirements.txt` inside the launcher virtualenv |
| ECR image not found | Verify account, region, repository, and tag; confirm the image was pushed |
| `AccessDenied` or role errors | AWS SSO session, SageMaker permissions, ECR/S3 role policies, `iam:PassRole` |
| SageMaker instance unavailable | Region availability and SageMaker training instance quota |
| HuggingFace 401/403 | Token is missing, expired, or lacks read access to the dataset or model repo |
| LeRobot argument or configuration error | Check LeRobot v0.6.1 API and model-specific `train.sh` |
| CUDA OOM | Reduce batch size, choose a larger GPU, or adjust model-specific settings |
| Missing checkpoints | Check actual model output path, checkpoint creation, and SageMaker sync logs |
| No final `model.tar.gz` | Inspect job status, errors, runtime, and `/opt/ml/model` contents |
| Job costs unexpectedly high | Confirm instance type, runtime limit, and stop failed or unneeded jobs |

## 14. Development and handoff standards

- Keep model-specific dependencies in each image's `pyproject.toml`.
- Keep AWS launching logic shared in `launch_training.py`.
- Do not commit secrets, training datasets, videos, virtual environments, or checkpoints.
- Keep source and dependency versions pinned or reviewed when updating images.
- Review changes on a feature branch and merge through your team's normal process.
- For new models, add a Dockerfile, entrypoint, model config, and an explicit compatibility test.
- Before declaring the workflow production-ready, perform both image builds, a compatible-data smoke test, and at least one end-to-end SageMaker run for each model.

**Security and ownership:** Access to this repository, company source code, dataset repositories, and AWS resources must be authorized by the organization, even when the GitHub repository is private.
