# VLA Training on AWS SageMaker

Fine-tune Vision-Language-Action (VLA) models on AWS SageMaker.

## Supported Models

- **NVIDIA GR00T N1.7** — robotics foundation model
- **SmolVLA** — lightweight VLA from Hugging Face

## Quick Start

1. Fill in your hyperparameters in `configs/groot.yaml` or `configs/smolvla.yaml`
2. Build the Docker image for your chosen model
3. Push the image to Amazon ECR
4. Run the training launcher

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/jobs/sagemaker/requirements.txt

AWS_PROFILE=datascience .venv/bin/python scripts/jobs/sagemaker/launch_training.py \
  --model-type groot \
  --steps YOUR_STEPS \
  --batch-size YOUR_BATCH_SIZE \
  --save-freq YOUR_SAVE_FREQ
```

See [docs/SAGEMAKER_TRAINING.md](docs/SAGEMAKER_TRAINING.md) for full setup, Docker build, ECR push, and training instructions.
