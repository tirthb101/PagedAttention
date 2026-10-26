# Running on the college SLURM cluster

We have no local GPU. All GPU work runs as batch jobs (`sbatch`) on the college cluster. Everything that does not need a GPU (block manager, scheduler, simulator, most tests) runs on a laptop CPU, so the cluster is only used for model runs.

---

## 0. Google Colab (T4)

Colab gives us a **Tesla T4 (16 GB)**, not an A100. Use it for interactive correctness checks and Triton kernel debugging with OPT-125m / OPT-1.3B (`colab/run_prototype.ipynb`). OPT-13B does not fit; all reported measurements come from the cluster.

## 1. What we know about the cluster

From the department's *Singularity HPC Cluster User Guide* (cs.iitgn.ac.in/hpc):

- Log in to the **master node** with `ssh` (Linux/WSL) or MobaXterm (Windows); campus network/VPN needed. The master node's address is in the guide.
- Normal shell use on the master node is fine (`ls`, `mkdir`, `vim`, `git`, `sbatch`, `squeue`, `scancel`, `sacct`, `sinfo`). **Computation goes through `sbatch`.**
- Conda environments are created on the master node: `module load anaconda3-2024`, `source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh`, `conda create ...` -- exactly what `slurm/setup_env.sh` does.
- Files from the laptop: `scp <file> <user>@<master>:<dir>` (fallback for models/data if the cluster has no internet).
- The cluster also runs **GitLab** and **JupyterHub** (same login as SSH). GitLab is a fallback if GitHub is unreachable; don't run GPU work in JupyterHub on the master node.
- `squeue` shows QOS `32c2g5dw` (likely 32 cores / 2 GPUs / 5 days) and node `node1`.
- **Checked 26 Oct:** GPU nodes `node1`, `node2` have **NVIDIA L40S 48 GB**, driver 570.86.10 (**CUDA ≤ 12.8**). PyTorch must be the `cu128` build (`setup_env.sh` does this); the default `cu130` wheel cannot see the GPU.


From a known-working job script (`run_large_dataset.sh`) and the cluster limits:

| Item | Value |
| --- | --- |
| Partition | `gpu` |
| GPU request | `--gres=gpu:1` (max **2** GPUs per job) |
| CPUs | `--cpus-per-task` up to **16** |
| Job time | up to **5 days** per job (`--time=5-00:00:00`) |
| Python environment | conda: `module load anaconda3-2024`, then `source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh`, `conda activate <env>` |
| Working directory | jobs `cd $SLURM_SUBMIT_DIR`; submit from the repo root |

Still to check (the first `check_gpu.sbatch` run prints most of these):

| Question | How to check | Answer |
| --- | --- | --- |
| GPU model and memory (L40S 48 GB?) | `check_gpu.sbatch` log | |
| Are both GPUs on a node the same type? | `sbatch --gres=gpu:2 slurm/check_gpu.sbatch` | |
| Do compute nodes have internet? | `check_gpu.sbatch` log | |
| Default RAM per job (OPT-13B needs ~30 GB host RAM to load) | `check_gpu.sbatch` prints `free -g`; `scontrol show partition gpu` | |
| Home quota (models are ~30 GB) | `quota -s` / `df -h $HOME` | |
| Max jobs running at once per user | `sacctmgr show qos format=name,maxjobspu,maxtrespu` | |

## 2. Where things live

```
laptop (CPU)                       cluster
────────────                       ───────
edit code, run CPU tests  ──git──► ~/PagedAttention          (code, small)
                                   conda env "pagedattn"
                                   $PA_SCRATCH/hf_cache      (models ~30 GB, datasets)
                                   $PA_SCRATCH/results       (CSVs)
results/ + plots  ◄──scp/git───────┘
```

- **Code** moves by git (push from laptop, `git pull` on the cluster). This also gives the instructor the commit history.
- **Models and datasets** go under `$PA_SCRATCH` (default `~/pa_data`; point it at a scratch/project dir if the home quota is small). `HF_HOME` is set there in every job.
- **Results** are small CSVs; copy them back with `scp` or commit them.

## 3. One-time setup (on the login node, which has internet)

```bash
git clone <your-private-repo> ~/PagedAttention && cd ~/PagedAttention
cp slurm/cluster.env.example slurm/cluster.env   # check PA_SCRATCH
sbatch slurm/setup_env.sbatch                    # step 1 (job): conda env + pip packages
#   ...wait until it finishes (squeue -u $USER), check logs/pa-setup-<id>.out ends with "Environment ready"
sbatch --export=ALL,SMALL=1 slurm/download_data.sbatch   # step 2 (job): small models + datasets (~3 GB)
#   later, for OPT-13B: sbatch slurm/download_data.sbatch
sbatch slurm/check_gpu.sbatch                    # GPU visible, torch.cuda works, OPT-125m generates
cat logs/pa-check-<jobid>.out
```

Always submit from the repo root: job scripts use relative paths (`slurm/common.sh`, `logs/`). Models and datasets are downloaded up front because compute nodes may have no internet; jobs run with `HF_HUB_OFFLINE=1`.

**If your cluster only allows work through `sbatch`** (no scripts on the login node), use the two `.sbatch` jobs above; they run `setup_env.sh` / `download_data.sh` on a compute node. They need internet on that node and say so clearly in the log if there is none. If the login node may run scripts, `bash slurm/setup_env.sh` and `bash slurm/download_data.sh --small` do the same directly.

**Cluster rules for the download step.** `download_data.sh` moves ~3 GB (`--small`) or ~30 GB (full, with OPT-13B). Many clusters allow this on the login node, but some require a data-transfer node or ask that heavy work stay off login nodes. Check the usage policy or ask the admins before running it there. Run it inside `tmux`/`screen`; rerunning resumes an interrupted download. Never run benchmarks or GPU work on the login node -- only through `sbatch`.

## 4. Job scripts

All in `slurm/`, all follow the same pattern as the known-working script (partition `gpu`, conda via module, `cd $SLURM_SUBMIT_DIR`, unbuffered Python).

| Script | Purpose | GPUs / CPUs / time |
| --- | --- | --- |
| `check_gpu.sbatch` | Environment sanity check | 1 / 4 / 15 min |
| `test_gpu.sbatch` | `pytest -m gpu` (Triton kernel, e2e equivalence on GPU) | 1 / 4 / 30 min |
| `serve_sweep.sbatch` | One system × one trace × all request rates (E1–E5, E7–E9) | 1 / 8 / 12 h default (override with `--time`) |
| (no script needed) `python benchmarks/kernel_bench.py` inside any GPU job | Kernel microbenchmark (E6) | 1 / 4 / 30 min |

**Submitting a full experiment** = a job **array**, one array task per system:

```bash
# E1 on ShareGPT: 4 systems (paged, orca-oracle, orca-pow2, orca-max) → indices 0..3
sbatch --export=ALL,EXP=e1,MODEL=facebook/opt-13b,TRACE=sharegpt slurm/serve_sweep.sbatch   # array 0-3, max 2 at once (set in the script)
squeue -u $USER          # watch
sacct -j <jobid>         # after it ends: state, elapsed time
```

Each task loads the model **once** and loops over all request rates, appending one CSV row per rate. If a job hits its time limit, the rates already finished are kept.

**Using the second GPU.** Our experiments are single-GPU (like the paper's OPT-13B setup), so the second GPU is best used by running **two 1-GPU array tasks at the same time**, which halves wall-clock time. A 2-GPU job (`--gres=gpu:2`) is only needed for the optional §4.6 tensor-parallel stretch goal (see EXTENSIONS.md, X6).

## 5. Habits that save GPU time

- **Debug on CPU first.** Never submit a GPU job to find a Python bug. `pytest` on the laptop must pass before any `sbatch`.
- **Smoke-test with OPT-125m and one rate** before a 13B sweep (example at the top of `serve_sweep.sbatch`).
- **Ask for realistic time limits** (`--time` a bit above what you need) even though 5 days is allowed. Shorter jobs usually start sooner, and a buggy job dies sooner.
- **Don't merge everything into one multi-day job.** One bug or node failure would lose all of it; one array task per system keeps failures small.
- **Submit big sweeps in the evening**, check results in the morning.
- **Logs:** every job writes `logs/%x-%j.out`, starting with host, GPU name, git commit hash and any other processes on the GPU.
- **Fixed seeds** for traces so reruns are comparable.
- **Don't request more than you use** (`--cpus-per-task=8` is plenty; 16 is the max).

## 6. Cluster-specific risks

| Risk | Mitigation |
| --- | --- |
| Queue waits near the deadline (other students' projects) | Submit the OPT-13B headline runs by 14 Nov; keep OPT-1.3B results from week 3 as fallback |
| A 1-hour-trace sweep runs for many hours | Fine within the 5-day limit; CSV rows are written per rate, so a crash loses at most one rate |
| Another job sharing the GPU | Logs print `nvidia-smi` compute apps; discard runs where something else was running |
| OOM-kill while loading OPT-13B on the host | Add `#SBATCH --mem=64G` |
| Home quota full | Move `PA_SCRATCH` to a scratch/project directory |
| Triton compile cache on a slow shared filesystem | `TRITON_CACHE_DIR` is set under `PA_SCRATCH` |
