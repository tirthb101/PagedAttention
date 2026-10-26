#!/bin/bash
#SBATCH --job-name=pa-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# Environment sanity check. From the repo root:  sbatch slurm/check_gpu.sbatch

source slurm/common.sh

python - <<'EOF'
import torch, time
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "No GPU visible - check --gres"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
try:
    import triton; print("triton", triton.__version__)
except ImportError:
    print("triton NOT installed")

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16).cuda()
ids = tok("Four score and seven years ago our", return_tensors="pt").input_ids.cuda()
t = time.time(); out = model.generate(ids, max_new_tokens=20, do_sample=False)
print("generated in %.2fs:" % (time.time() - t), tok.decode(out[0]))
EOF

free -g | head -2
# Does this compute node have internet? (informational)
if timeout 5 curl -sI https://huggingface.co >/dev/null 2>&1; then
  echo "compute node HAS internet"
else
  echo "compute node has NO internet - jobs run offline (expected)"
fi
