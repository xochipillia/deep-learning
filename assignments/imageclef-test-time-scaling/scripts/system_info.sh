#!/usr/bin/env bash
# Print the system/software provenance block written to environment/system-info.txt.
source "$(dirname "$0")/env.sh"
echo "# System information (collected $(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo "## OS";            (cat /etc/os-release | grep -E '^(PRETTY_NAME|VERSION_ID)='; uname -srm)
echo "## Host";          echo "instance_type=${INSTANCE_TYPE:-ml.g6e.2xlarge (SageMaker Studio JupyterLab)}"
python3 - <<'EOF' 2>/dev/null || true
import json, pathlib
p = pathlib.Path("/opt/ml/metadata/resource-metadata.json")
if p.exists():
    m = json.loads(p.read_text())
    print("sagemaker_app_type=" + str(m.get("AppType")), "sagemaker_resource=" + str(m.get("ResourceName")))
EOF
echo "## CPU/RAM";       nproc; free -g | head -2
echo "## GPU";           nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi | grep -E "CUDA Version" | sed 's/|//g' | xargs
echo "## Python";        python --version; which python
echo "## Libraries"
python - <<'EOF'
import importlib
for name in ["torch", "vllm", "transformers", "datasets", "huggingface_hub", "PIL", "numpy"]:
    try:
        m = importlib.import_module(name); print(f"{name}={getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{name}=NOT_IMPORTABLE ({type(e).__name__})")
import torch
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device={torch.cuda.get_device_name(0)}")
    print(f"torch.cuda.total_memory_GiB={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}")
print(f"cudnn={torch.backends.cudnn.version()}")
EOF
echo "## nvcc";          (nvcc --version 2>/dev/null | tail -2) || echo "nvcc not on PATH (CUDA runtime comes from PyTorch/vLLM wheels)"
echo "## Repository";    echo "url=$REPO_URL"; echo "commit=$(git -C "$REPO" rev-parse HEAD)"
echo "## Model";         echo "id=$MODEL_ID"; echo "revision=$MODEL_REVISION"
ls "$HF_HOME/hub/models--${MODEL_ID//\//--}/snapshots/" 2>/dev/null | sed 's/^/cached_snapshot=/'
echo "## Dataset";       echo "id=$DATASET_ID"; echo "revision=$DATASET_REVISION"; echo "split=validation"
ls "$HF_HOME/hub/datasets--${DATASET_ID//\//--}/snapshots/" 2>/dev/null | sed 's/^/cached_snapshot=/'
