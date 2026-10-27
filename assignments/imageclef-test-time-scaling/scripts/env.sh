# Shared settings, sourced by every instance-side script.
# Paths assume the SageMaker Studio JupyterLab space layout but can be
# overridden for any other Linux + NVIDIA GPU host.
export WORK=${WORK:-/home/sagemaker-user/work}
export REPO=${REPO:-$WORK/tts-small-vlm}            # official repo, never edited
export REPL=${REPL:-$WORK/replication}              # this package
export VENV=${VENV:-$WORK/venv}
export HF_HOME=${HF_HOME:-$WORK/hf-cache}           # model/dataset cache (NOT part of the package)
export REPO_URL=https://github.com/lesterpjy/tts-small-vlm
export REPO_COMMIT=2247c130075889fcdcac31e0eb83dba40c67afbd
export MODEL_ID=Qwen/Qwen3.5-4B
export MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export DATASET_ID=MBZUAI/EXAMS-V
export DATASET_REVISION=7594b37a10e87fcfbc4def0fa61809ad7114b7f2
export SUBSET_SIZE=${SUBSET_SIZE:-200}
export SMOKE_SIZE=${SMOKE_SIZE:-10}
# Observability backends of the official repo are optional; disable them.
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Environment deviation (documented in README): vLLM 0.32's FlashInfer top-p
# sampler JIT-compiles CUDA sources at first use; the Studio image's nvcc 12.9
# cannot build them against the cu130 wheels. Use vLLM's PyTorch-native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Price inputs for the cost estimate (us-east-1, AWS Pricing API, 2026-10-25).
export PRICE_INSTANCE_HOURLY_USD=2.80      # Studio JupyterLab ml.g6e.2xlarge
export PRICE_EBS_GB_MONTH_USD=0.112        # Studio gp3 ML storage
export EBS_GB=200
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
true
