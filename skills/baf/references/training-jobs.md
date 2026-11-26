# Training jobs on BAF

Pattern taken from the group's guide (`docs/baf/` in Polygl0t/llm-foundry).
Start from the current `job.jdl`, `train_ddp.sh` and `create_venv.sh` in the
user's checkout; the sketches here explain the structure and what must not
change.

## 1. Environment as a tarball (one-time, as a short batch job)

A Python environment is tens of thousands of files and `$BUDDY` allows
100,000. So the environment is built on `/jwd`, packed into one file, and
unpacked by every job. The agent has no terminal for an interactive job, so
the build runs as a **batch job** (user's decision, 2026-11-26); within the
GPU budget it needs no approval.

The group's `create_venv.sh` (in `docs/baf/` of the user's `llm-foundry`
checkout on `$BUDDY`) already does the whole build on `/jwd` and writes
`$BUDDY/.venv.tar.gz`; it only needs `$BUDDY` set. Wrapper and submit file,
both in home:

```bash
# ~/jobs/build_env.sh
#!/bin/bash
set -euo pipefail
source /etc/profile >/dev/null 2>&1          # sets $BUDDY inside the job
cd /jwd
bash "$BUDDY/llm-foundry/docs/baf/create_venv.sh"   # adjust to the checkout's path
```

```text
# ~/jobs/build_env.jdl
executable            = build_env.sh
output                = build_env-$(ClusterId).out
error                 = build_env-$(ClusterId).err
log                   = build_env-$(ClusterId).log
request_gpus          = 1
request_cpus          = 8
request_memory        = 32 GB
request_disk          = 60 GB
requirements          = (CUDADeviceName == "NVIDIA H200")
+ContainerOS          = "Rocky9"
+CephFS_IO            = "medium"
+MaxRuntimeHours      = 2
+ClaudeAgent          = True
JobBatchName          = build-env
getenv                = BUDDY
transfer_output_files = ""
queue 1
```

- Build on the GPU type you will train on: packages such as flash-attn
  compile against the GPU and CUDA they see. Check that machine's health
  first (`NODE_IS_HEALTHY`); an H200 build cannot run while `gpu003` is down.
- More CPUs than the interactive recipe (`request_cpus = 1`) shorten the
  kernel compiles; 8 is a reasonable middle.
- The build log is the job's `.out`/`.err` in home. Read it: the script
  prints installed versions and an import check at the end, and some
  packages fail "non-fatally" with only a warning.
- Rebuild only when dependencies change. Keep one tarball; each one counts
  against the 500 GB, and an old one should be removed once the new one
  works.

## 2. Submit file

Kept in home, submitted from home.

```text
executable            = /home/<uni-id>/scripts/train_ddp.sh
arguments             = $(ClusterId)

request_gpus          = 2
request_cpus          = 48
request_memory        = 200 GB
request_disk          = 100 GB
requirements          = (CUDADeviceName == "NVIDIA H200")

+ContainerOS          = "Rocky9"
+CephFS_IO            = "high"
+MaxRuntimeHours      = 24
+ClaudeAgent          = True
JobBatchName          = <run name; reuse it for the short test run>

output                = /dev/null
error                 = /dev/null
log                   = $(ClusterId).log

getenv                = BUDDY
transfer_output_files = ""
queue 1
```

- stdout/stderr go to `/dev/null` because the wrapper writes its own logs to
  `$BUDDY/logs/` where they can be read while the job runs.
- `getenv = BUDDY` passes only that variable, and only if it is set where
  `condor_submit` runs: submit from a login shell (`bash -lc`). Add others by name
  (`getenv = BUDDY, HF_TOKEN, WANDB_API_KEY`), never the whole environment.
- `request_disk` covers `/jwd`: the unpacked venv, the code copy and any
  node-local cache.
- Requesting all 8 H200s means waiting for the whole machine to drain. Check
  `condor_status` and say what is free before proposing a GPU count.

## 3. Wrapper script

What the wrapper does, in order:

1. `cd /jwd`; unpack `$BUDDY/.venv.tar.gz`; copy the code from `$BUDDY` to
   `/jwd` (so concurrent jobs do not all read source files from CephFS and do
   not see each other's edits).
2. Load modules, activate the venv.
3. Point caches at `/jwd` (`HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`,
   `WANDB_DIR`, `TRITON_CACHE_DIR`). Use `$BUDDY` for a cache only when a
   multi-job run needs to reuse it, and mind the file count.
4. Launch, with logs appended to `$BUDDY/logs/out-<ClusterId>` and
   `err-<ClusterId>`.
5. Exit with the training process's exit code.

Single-node DDP launch:

```bash
NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
export MASTER_ADDR=localhost
export MASTER_PORT=$(( 49152 + (CLUSTER_ID % 16384) ))   # avoids clashes between jobs on one node

torchrun --nnodes=1 --nproc_per_node="$NGPU" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  /jwd/<repo>/distributed/train_ddp.py --specs "$BUDDY/specifications.yaml" \
  1>>"$out" 2>>"$err"
```

Deriving `NGPU` from `CUDA_VISIBLE_DEVICES` keeps `--nproc_per_node` in step
with `request_gpus`. (The group template hard-codes the count; if you keep it
hard-coded, change both together.) **Read** `CUDA_VISIBLE_DEVICES`, never
assign it.

Check GPU use in the first minutes (the conduct rule): have the wrapper log
`nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv`
once after start-up. `nvidia-smi` shows **every** GPU on the node, so read only
the rows whose UUID starts with one of your `CUDA_VISIBLE_DEVICES` entries.

Checkpoints go to `$BUDDY`, few and large. With a 7-day cap and possible node
drains, the job must resume from the latest checkpoint when resubmitted.

## 4. Multi-GPU jobs and vLLM: expand the GPU IDs first

HTCondor on BAF exports `CUDA_VISIBLE_DEVICES` as short UUID prefixes
(`GPU-af7b61d8`). PyTorch accepts them, but vLLM (and anything resolving GPUs
through NVML) fails with `NVMLError_NotFound`, also in the subprocesses it
spawns, so the fix must be in the environment before anything starts.

**Rule (user, 2026-11-26): every job with more than one GPU, and every vLLM
job, runs BAF's own helper first**, before any `torchrun`/`vllm`/Python:

```bash
source /etc/profile                   # also sets $BUDDY
if type expand_cuda_visible_devices >/dev/null 2>&1; then
  echo "CUDA_VISIBLE_DEVICES before: $CUDA_VISIBLE_DEVICES"
  expand_cuda_visible_devices         # short IDs -> full UUIDs, same GPUs
  echo "CUDA_VISIBLE_DEVICES after:  $CUDA_VISIBLE_DEVICES"
  # the helper silently keeps an ID it cannot resolve: insist on full UUIDs
  [[ $CUDA_VISIBLE_DEVICES =~ ^GPU-[0-9a-f-]{36}(,GPU-[0-9a-f-]{36})*$ ]] \
    || { echo "GPU IDs not fully expanded" >&2; exit 1; }
else
  echo "expand_cuda_visible_devices missing; use the fallback below" >&2
  exit 1
fi
NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
```

- This is the sanctioned way to change `CUDA_VISIBLE_DEVICES`: it names the
  same GPUs in full. Never write the variable yourself.
- Take `--nproc_per_node` / `--tensor-parallel-size` from `NGPU`.
- Tested 2026-11-26 in a 2-GPU job on `gpu001` (Rocky9, job 213779): the
  function exists after `source /etc/profile`, turned
  `GPU-d37cc514,GPU-03ed5210` into the two full UUIDs, CUDA then saw exactly
  2 devices, and NVML's lookup by UUID (the call vLLM makes) found both.
  It matches each short ID as a prefix against `nvidia-smi`'s UUID list and
  **leaves an ID unchanged if nothing matches**, hence the check above.
  vLLM itself has not been started on BAF yet.

Fallback if the function is missing: the group guide's block
(`docs/baf/README.md`, "Non-integer GPU IDs break vLLM") maps each short ID to
the **physical** index of the same GPU via
`nvidia-smi --query-gpu=index,uuid`, matched as a prefix (never `0..N-1`:
`nvidia-smi` in a job sees every GPU on the node, including other users').
If you use it, also `export CUDA_DEVICE_ORDER=PCI_BUS_ID` so CUDA's numbering
matches `nvidia-smi`'s; the group block does not set it.

vLLM specifics: model weights may be read from `$BUDDY` (few large files);
caches go to `/jwd`. Smoke-test with `+MaxRuntimeHours = 1` and a small model.

## 5. After submitting

- Note ClusterId and the submit host (the fixed `baf` desktop), and check `klist` first (see SKILL.md, Access).
- First check after a minute or two: `tail` the log on `$BUDDY`; confirm the
  venv unpacked, the expected number of processes started and loss is
  printing. Then check at long intervals, or `condor_wait <ClusterId>.log`.
- Idle for long: `condor_q -better-analyze <id>`; usually the GPU/CPU/memory
  combination is not free. Lower the request with `condor_qedit` rather than
  stacking new submissions.
- OOM: lower batch sizes or raise `request_memory`.

## 6. Things to tell the user when you see them

- A `.jdl` or wrapper that assigns `CUDA_VISIBLE_DEVICES` or a thread-count
  variable.
- An unpacked environment, cache or dataset of many small files on `$BUDDY`.
- An interactive job left open.
- A CPU-only step submitted with `request_gpus > 0`.
