#!/bin/bash
# One-time setup, step 2 of 2: download models (~29 GB) and datasets (~1 GB) into $PA_SCRATCH
# so that GPU jobs can run offline.
#
# WHERE TO RUN THIS - check your cluster's rules first:
#   * Many clusters allow downloads on the login node; some ask you to use a dedicated
#     data-transfer node, or to keep heavy work off login nodes. Read the usage policy or
#     ask the admins before running this on the login node.
#   * The download is network/disk I/O, not computation, and uses one process.
#   * Run it inside tmux/screen so a dropped SSH connection does not stop it; it resumes
#     where it stopped if you run it again.
#   * Start with --small (OPT-125m + OPT-1.3B, ~3 GB) and add OPT-13B later if needed.
#
#   bash slurm/download_data.sh --small     # models for smoke tests + datasets
#   bash slurm/download_data.sh             # everything, including OPT-13B (~26 GB)
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/slurm/cluster.env"
module load $PA_MODULES
source "$PA_CONDA_SH"
conda activate "$PA_CONDA_ENV"

MODELS="facebook/opt-125m facebook/opt-1.3b"
[ "${1:-}" != "--small" ] && MODELS="$MODELS facebook/opt-13b"

df -h "$PA_SCRATCH" | tail -1
# Python API instead of the CLI: the command was renamed (huggingface-cli -> hf) in newer
# huggingface_hub versions, while snapshot_download works with all of them.
MODELS="$MODELS" python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download

for m in os.environ["MODELS"].split():
    print(f"Downloading {m} ...", flush=True)
    snapshot_download(m, ignore_patterns=["*.msgpack", "*.h5", "*tf_model*", "*flax*"])

print("Downloading datasets (ShareGPT, Alpaca, WMT16 de-en) ...", flush=True)
hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered",
                "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")
snapshot_download("tatsu-lab/alpaca", repo_type="dataset")
snapshot_download("wmt/wmt16", repo_type="dataset", allow_patterns=["de-en/*"])
PY

du -sh "$HF_HOME"
echo "Downloads done. Next: sbatch slurm/check_gpu.sbatch"
