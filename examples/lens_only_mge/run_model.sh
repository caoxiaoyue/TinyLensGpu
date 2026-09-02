#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly CONDA_ENV="tinylens_gpu"
readonly MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-500}"
readonly MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"

# Select the idle GPU with the least allocated memory. nvidia-smi reports one
# line per physical GPU as: index, memory.used (MiB), utilization.gpu (%).
gpu_id="$({
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
} | awk -F',' \
    -v max_mem="${MAX_MEMORY_USED_MB}" \
    -v max_util="${MAX_GPU_UTILIZATION}" '
        {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
        }
        $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ &&
        $2 <= max_mem && $3 <= max_util {
            print $1, $2, $3
        }
    ' | sort -k2,2n -k3,3n | awk 'NR == 1 { print $1 }')"

if [[ -z "${gpu_id}" ]]; then
    echo "Error: no idle GPU found." >&2
    exit 1
fi

conda_base="$(conda info --base)"
# shellcheck source=/dev/null
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using physical GPU ${gpu_id} in conda environment '${CONDA_ENV}'."
cd "${SCRIPT_DIR}"
exec python -u run_model.py "$@"
