---
name: htcondor
description: How to use the HTCondor scheduler well - writing submit description files (.jdl/.sub), interactive jobs, queueing many jobs, GPU requests, file transfer, waiting for and inspecting jobs, diagnosing idle and held jobs. Use for any task that submits, monitors, debugs or removes jobs on an HTCondor pool (condor_submit, condor_q, condor_status, condor_rm). Cluster-agnostic; pair it with hpc-agent-conduct and the cluster skill (baf).
---

# HTCondor

Generic mechanics only. Site attributes, storage and limits come from the
cluster skill and the live pool. Follow `hpc-agent-conduct` for what needs
approval and how often to query.

## How it differs from Slurm

- No partitions. A job states what it needs (`request_*`, `requirements`) and
  is matched to any machine slot that fits.
- Jobs have a **ClusterId.ProcId** (`1234.0`). One submit creates one cluster
  with one or more procs.
- A failing job is not always removed: it may go on **hold** and wait for you.
- The **user log** (`log = ...`) records every state change of the job. Read it
  instead of asking the scheduler.
- There is usually no shared environment: the job starts in a scratch
  directory, often inside a container, and sees only what the site mounts.

## Discover, do not guess

```bash
condor_status -compact                                  # machines and free resources
condor_status -compact -constraint 'GPUs > 0' \
  -af:h Machine State TotalGPUs GPUs TotalCpus Cpus TotalMemory Memory CUDADeviceName
condor_status -avail -af:h Name Cpus Memory             # idle slots
condor_userprio                                         # your priority (lower is better)
```

## Submit description file

```text
executable     = /path/to/run.sh
arguments      = $(ClusterId) $(ProcId)

request_cpus   = 8
request_memory = 32 GB
request_disk   = 20 GB
request_gpus   = 1
requirements   = (CUDADeviceName == "<name from condor_status>")

output         = logs/$(ClusterId).$(ProcId).out
error          = logs/$(ClusterId).$(ProcId).err
log            = logs/$(ClusterId).log

getenv         = False
queue 1
```

- Units: bare numbers are MiB for `request_memory` and KiB for `request_disk`.
  Write the unit explicitly.
- `output`/`error` are staged in the job's scratch directory and copied back
  when the job ends; the directories must exist at submit time. For live
  progress on a long job, have the script write to shared storage itself.
- Do not pass the whole submit environment (`getenv = True`). List the
  variables you need: `getenv = VAR1, VAR2` or `environment = "A=1 B=2"`.
- The executable is copied and renamed (`condor_exec.exe`), so `$0` and
  paths relative to the script do not work. Use absolute paths.
- Site-specific job attributes are written with a leading plus:
  `+SomeSiteAttribute = "value"`.

## Interactive job

```bash
condor_submit -interactive job.jdl
condor_submit -interactive -append 'request_cpus = 4' -append 'request_memory = 16 GB'
```

Exit as soon as you are done.

## Many jobs: one cluster, throttled

```text
arguments       = $(Item)
max_materialize = 20          # at most 20 procs idle or running at once
queue Item from items.txt     # or: queue 100   (uses $(ProcId) 0..99)
```

Never call `condor_submit` in a loop. Batch very short tasks inside one job.
For multi-stage pipelines use DAGMan (`condor_submit_dag`) with `MAXJOBS`.

## Waiting and inspecting

Prefer the user log and one blocking wait over repeated `condor_q`.

```bash
condor_wait logs/<ClusterId>.log            # blocks until the cluster finishes
condor_wait -wait 3600 logs/<ClusterId>.log # ... or one hour at most
condor_q                                    # your jobs on this schedd
condor_q -nobatch                           # one line per proc
condor_q <id> -af:h JobStatus HoldReason RemoteHost RequestMemory ResidentSetSize_RAW
condor_tail <id>                            # last lines of stdout of a running job
condor_tail -stderr <id>
condor_ssh_to_job <id>                      # shell next to the running job (if the site allows)
condor_history <id> -limit 1 -af:h ExitCode RemoteWallClockTime ResidentSetSize_RAW RequestMemory
```

JobStatus: 1 idle, 2 running, 3 removed, 4 completed, 5 held.

A pool can have **several schedds** (one per submit node). `condor_q` only
shows the schedd of the machine you are on. If a job seems to have vanished,
check which host you are on before concluding anything; query a specific one
with `condor_q -name <schedd>`. Avoid `condor_q -global` and `-all` in routine
checks - they query every schedd for every user.

Keep `condor_history` bounded: always give an id or `-limit N`. Unbounded
history scans are expensive.

## Idle and held jobs

```bash
condor_q -better-analyze <id>     # why an idle job does not match
condor_q -hold <id>               # hold reason
```

- **Idle**: usually the request matches no slot (too many GPUs/CPUs/memory, a
  `requirements` typo, or everything is busy). Compare with `condor_status`.
  Reduce the request with `condor_qedit <id> RequestMemory <MiB>` instead of
  resubmitting.
- **Held**: read the reason first. Common ones: memory or disk exceeded the
  request, a transfer file is missing, output directory missing, credential
  expired. Fix the cause, then `condor_release <id>` (ask first). Releasing
  without fixing just holds it again.

## Removing (ask first, unless it is your own tagged job - see hpc-agent-conduct)

```bash
condor_rm <ClusterId>             # whole cluster
condor_rm <ClusterId>.<ProcId>    # one proc
```

Never run `condor_rm -all` or `condor_rm $USER` without the user confirming
that every job should go.

## GPU jobs

- HTCondor sets `CUDA_VISIBLE_DEVICES` for the job, often as GPU UUIDs rather
  than indices. Never override it. `nvidia-smi` may show GPUs that are not
  yours.
- Derive process counts from the job, not from the machine: use the number of
  entries in `CUDA_VISIBLE_DEVICES`, and the CPU count from the job ad
  (`grep -i '^Cpus ' "$_CONDOR_MACHINE_AD"`) rather than `nproc`.
- Thread-count variables (`OMP_NUM_THREADS` and friends) are set to match
  `request_cpus`. Leave them alone.

## File transfer vs shared storage

- With a shared filesystem, read and write there directly and set
  `transfer_output_files = ""` if nothing should be copied back.
- Otherwise list inputs in `transfer_input_files`; everything new in the top
  level of the scratch directory is transferred back at exit. Put temporary
  files in a subdirectory so they are not.
- Useful job-side variables: `$_CONDOR_SCRATCH_DIR` (job working directory),
  `$_CONDOR_JOB_AD`, `$_CONDOR_MACHINE_AD`.

## Resilience

- Set an explicit runtime limit where the site provides one, with a margin.
- Checkpoint to shared storage and resume at start; a job can be evicted or
  its node drained.
- Return the real exit code from the wrapper script so HTCondor and DAGMan see
  failures.
