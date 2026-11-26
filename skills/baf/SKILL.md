---
name: baf
description: Site knowledge for BAF, the Bonn Analysis Facility at the Physikalisches Institut of Uni Bonn (HTCondor, containerised jobs, CephFS, H200/H100 GPUs; login via desktop.physik.uni-bonn.de). Use whenever work touches BAF - connecting, $BUDDY and CephFS quotas, writing .jdl files with ContainerOS/CephFS_IO/MaxRuntimeHours, the venv-tarball pattern, single-node multi-GPU training, dataset downloads, or anything run over `ssh baf`. Load together with hpc-agent-conduct and htcondor.
---

# BAF (Bonn Analysis Facility)

Sources, read 2026-11-25:

- The group's guide and templates: `docs/baf/` in
  https://github.com/Polygl0t/llm-foundry (README, `job.jdl`, `train_ddp.sh`,
  `create_venv.sh`, `download.sh`). This is the reference for the group's
  workflows; **read the current version in the user's checkout before writing
  job files**, it is updated more often than this skill.
- Official documentation (PHY-IT Confluence; readable without login from the
  university network, so `curl` from the workstation works):
  [BAF](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814258/BAF),
  [HTCondor on BAF](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814633/HTCondor+on+BAF),
  [GPU jobs](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/12289903/HTCondor+GPU+jobs),
  [Helpful commands](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814637/Helpful+HTCondor+commands),
  [JDL tips](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/27297351/HTCondor+JDL+Tips+Tricks),
  [CephFS](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814260/CephFS),
  [Conda and VirtualEnvs](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814595/Conda+and+VirtualEnvs+on+BAF),
  [Prolonging Kerberos TGTs](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/17011120/Prolonging+Kerberos+TGTs).
  Compared with this skill on 2026-11-26. Where they disagree, the official
  pages win.

Hardware counts and limits change. Confirm with `condor_status` before relying
on any number below. Facts marked "(live 2026-11-26)" were checked on the
pool from one submit desktop.

## Mental model

Three places, each visible from different sides:

| Place | Path | Visible on the desktop | Visible in job | Persistent |
|---|---|---|---|---|
| Home | `$HOME` (`/home/<uni-id>/physik` on the desktops) | yes | **no** (test job 2026-11-26) | yes, backed up |
| CephFS ("BUDDY") | `/cephfs/user/<uni-id>` = `$BUDDY` | yes | yes | yes |
| Job working dir | `/jwd` | no | yes | **wiped at job end** |

Every job runs in a container. Consequences:

- Anything the job needs (code, env tarball, configs, data) must be on
  `$BUDDY`. Symlinks from `$BUDDY` into home do not work.
- **Submit from home, never from `/cephfs`.** Keep `.jdl` files and the
  executable wrapper script in home; `condor_submit ~/scripts/job.jdl`.
- Fast, throwaway work happens in `/jwd`; anything worth keeping is written to
  `$BUDDY`.
- **`$BUDDY` is not set in a batch job by default** (test job 2026-11-26).
  Put `getenv = BUDDY` in the submit file **and submit from a login shell**
  (`ssh baf 'bash -lc "cd ~/... && condor_submit job.jdl"'`): a plain
  `ssh baf 'condor_submit ...'` has no `$BUDDY` to pass on. Alternatively
  `source /etc/profile` in the wrapper script. Without it, `$BUDDY/...` paths
  silently become `/...`.

## Access

BAF has no dedicated login hosts: the institute's Ubuntu desktops are the
submit machines, and each runs its own schedd (60 schedds, live 2026-11-26).
`desktop.physik.uni-bonn.de` is load-balanced across them (`exp198`,
`theo198`, ...), so two logins can land on different machines. Access is
granted by PHY-IT (it-support@physik.uni-bonn.de) on request.

**The agent always uses one fixed desktop per user**: the `baf` alias in that
user's `~/.ssh/config` points at a single host, never at the load balancer.
Which desktop is the user's choice and differs between users; this skill
names none, so that a group does not pile onto one machine. Two reasons for
fixing it:

1. Jobs need a **Kerberos ticket on the submit machine** for as long as they
   are queued or running: HTCondor uses it to write logs and output to home and
   `$BUDDY`. Without one, jobs go on HOLD, are retried every 30 minutes, block
   resources and worsen the user's priority.
2. Jobs live in the schedd of the machine they were submitted from. Plain
   `condor_q`/`condor_rm` only see that machine's jobs.

A key login does not create a renewable ticket. The user creates it with a
**password** login to the fixed machine and keeps a `screen` session there, so
SSSD renews it. Tickets last 24 h, are renewed automatically, and expire for
good 7 days after the password login ("renew until"). Only the user's password
can create a new one; never ask for it, and never create a keytab.

### First use: choosing the submit desktop

Before the first BAF command for a user, check where `baf` points:

```bash
ssh -G baf | grep -i '^hostname '
```

If it prints one specific desktop (`<name>.physik.uni-bonn.de`), use it. If
the alias is missing, or it prints `desktop.physik.uni-bonn.de` (the load
balancer), set it up with the user - do not copy a host from this skill,
from examples or from another user's configuration:

1. Ask whether they have **their own institute desktop**; that is the best
   submit machine, since their ticket is renewed there whenever they unlock
   the screen.
2. If they have none or no preference, take the machine the load balancer
   assigns them: `ssh <uni-id>@desktop.physik.uni-bonn.de hostname -f` once,
   and use that name. Different users land on different machines.
3. With their agreement, write the alias (and the shared-connection settings
   from `hpc-agent-conduct`) into their `~/.ssh/config`:

   ```text
   Host baf
     HostName <that desktop>.physik.uni-bonn.de
     User <uni-id>
   ```

4. The user then logs in there once **with their password**
   (`ssh -o PreferredAuthentications=password baf`) and starts a `screen`
   session, as described above. Until that is done, do not submit.

If the fixed desktop is later rebooted, retired or unreachable, tell the
user and repeat this; jobs already submitted from it stay in its schedd.

Before every submission, check once:

```bash
ssh baf 'klist; screen -ls'
```

- `Ticket cache:` must be `KEYRING:persistent:...`. A path under `/tmp` is a
  forwarded ticket that dies at logout: stop and ask the user to renew.
- The `renew until` date must lie after the job's `MaxRuntimeHours` from now
  (plus queueing time). Otherwise ask the user to renew first:
  `ssh -o PreferredAuthentications=password baf`, and check their `screen`
  session still exists.
- Do not run `kinit` or handle the ticket yourself; do not print ticket
  contents beyond `klist`.

Non-interactive ssh does not set `$BUDDY`; use a login shell:
`ssh baf 'bash -lc "..."'` (live 2026-11-26).

## Storage limits

`$BUDDY` has a quota of **100,000 files and 500 GB**. The file-count limit is
the one that bites: a single unpacked Python environment can use most of it.

- Never unpack a venv, a conda env or a `node_modules` tree on `$BUDDY`.
- Keep datasets as a few large files (Parquet, tar shards).
- Hugging Face caches create many small files and symlinks; point caches at
  `/jwd` unless reuse across jobs is worth the file count.
- Check usage without walking the tree. Never `du` or `find | wc -l`.
  - **On a desktop** (what `ssh baf` reaches), `/cephfs` is an NFS mount and
    the `ceph.*` xattrs are not available ("Operation not supported", live
    2026-11-26). Bytes are still visible: CephFS reports a directory's size as
    its recursive total, so `ls -ld "$BUDDY"` (or `ls -lh` of a subdirectory)
    gives bytes used. The file count is not visible from a desktop.
  - **Inside a job** (CephFS mounted natively), `bquota` prints files and
    bytes against the quota (works once `$BUDDY` is set; tested 2026-11-26). Same data by hand:
    `getfattr -n ceph.dir.rentries --only-values "$BUDDY"` (files + dirs) and
    `ceph.dir.rbytes` (bytes); `getfattr -d -m ceph "$BUDDY"` shows all.
  - So a file-count check costs a job. Do it inside the next job you run
    anyway (add `bquota` to the wrapper script) rather than starting one only
    for that. Recursive totals lag behind by a while due to caching.

Home is a shared filesystem with heavy use: do not download or write large
data there.

## Jobs

HTCondor, `vanilla` universe only (no MPI/parallel universe, so
**multi-GPU means single node**). BAF-specific job attributes, required in
every submit file:

| Attribute | Values | Meaning |
|---|---|---|
| `+ContainerOS` | `"Rocky9"` (recommended), `"Rocky8"`, `"Debian13"`, `"Debian12"` | OS image of the job container (official list; `Debian11` is no longer offered) |
| `+CephFS_IO` | `"high"`, `"medium"`, `"low"`, `"none"` | throughput class for `/cephfs`; `high` uses InfiniBand, `medium` Ethernet, `low` only status and small text files, `none` means the job does not touch CephFS |
| `+MaxRuntimeHours` | integer | hard wall-clock limit per job, counted from start; max 168 for batch, 24 for interactive |

Ask for the lowest `CephFS_IO` class that fits and a realistic
`MaxRuntimeHours`: shorter jobs can backfill onto draining nodes. These three
attributes cannot be changed with `condor_qedit`, even for idle jobs; fixing
them means resubmitting (or mailing PHY-IT with job IDs and the submit host).
Slots named `sloti...` are reserved for interactive jobs.

GPU machines (live 2026-11-26):

| Machine | `CUDADeviceName` | GPUs | CPUs | RAM | VRAM per GPU |
|---|---|---|---|---|---|
| `gpu003` | `"NVIDIA H200"` | 8 | 192 | 2.0 TB | 141 GB |
| `gpu002` | `"NVIDIA H100 PCIe"` | 2 | 128 | 2.3 TB | 80 GB |
| `gpu001` | `"NVIDIA GeForce GTX 1080 Ti"` | 4 | 32 | 126 GB | 11 GB |

`gpu001` is old (no bf16, 11 GB); always pin `CUDADeviceName` in
`requirements` so a job does not land there by accident.

There are three GPU machines against about a hundred CPU-only ones.

**Free is not the same as available.** On 2026-11-26 `gpu003` showed all
8 H200s free while `NODE_IS_HEALTHY` was `false`
(`MARKED_UNHEALTHY_BY_PUPPET:MEM_ISSUE`): it accepted no jobs, and GPU jobs
pinned to it sat idle with "no match found". Before proposing or submitting
a GPU job, check health, `StartJobs` and `BackfillableMaxRuntimeHours` (a
draining node only takes jobs shorter than that). If the only matching
machine is unhealthy, tell the user instead of queueing; an idle GPU job
still counts against the budget. `condor_q -better-analyze <id>` explains
an idle job.
**CPU-only work (tokenisation, preprocessing, filtering) goes to non-GPU
machines with `request_gpus = 0`.** Scale CPUs with GPUs (about 24 per H200).

```bash
# free GPU resources - "free" means nothing if the machine refuses jobs:
# check NODE_IS_HEALTHY and StartJobs too
condor_status -compact -constraint 'TotalGPUs > 0' \
  -af:h Machine CUDADeviceName TotalGPUs GPUs Cpus NODE_IS_HEALTHY StartJobs NODE_HEALTH BackfillableMaxRuntimeHours
# free CPU-only resources
condor_status -compact -constraint 'isUndefined(Gpus) || Gpus == 0' \
  -af:h Machine State TotalCpus Cpus TotalMemory Memory TotalDisk Disk
```

### GPU budget: two

**Within the budget, submit without asking** (the user's standing rule for
BAF, 2026-11-26). That covers CPU-only jobs and GPU jobs up to 2 GPUs in total,
as long as the other limits hold (6 submissions an hour, 50 queued, capped
arrays). Still tell the user what you submitted: ClusterId, GPUs/CPUs/memory,
`MaxRuntimeHours`, where the logs go. The guard hook lets such submissions
through and asks only when a limit would be crossed. Put `+ClaudeAgent = True`
in every submit file; your own tagged jobs can then be removed by ID without
asking. Holding, releasing and editing jobs, and removing anything else,
still need approval.

**Jobs above 48 hours need a test run first** (user, 2026-11-26), like the
devel check on Marvin. Before submitting anything with `+MaxRuntimeHours`
above 48, run a reduced version of the same job - smaller problem, same GPU
type (H200 for an H200 job), `+MaxRuntimeHours` of 8 or less, **the same
`JobBatchName`** - and confirm from its output that it starts, uses its GPUs,
writes output and a checkpoint, and resumes from that checkpoint. Only then
submit the long job. The guard hook checks that such a test run completed
with exit code 0 and asks the user otherwise; whether the test really
exercised checkpoint and resume is yours to verify. Every submit file
therefore sets a `JobBatchName`.

GPUs are the scarcest thing on BAF and a job may hold them for up to a week.
The group's rule: **holding up to 2 GPUs in total is fine without asking, even
for long jobs. Anything that would take the total above 2 needs the user's
explicit approval, every time** - no standing permission, no "it is only for a
short test". **A GPU job counts from the moment it is submitted until it has
left the queue** (completed or removed): running, idle and interactive jobs
count, and so do **held** jobs, because BAF automatically releases held jobs
every 30 minutes (to ride out expired tickets and network glitches), so a held
GPU job will claim its GPUs again without anyone acting.

**Never stage jobs around the budget.** Do not submit GPU jobs on hold
(`hold = True`), hold running jobs to make room, release held jobs, or raise
`RequestGPUs` with `condor_qedit` to get more than 2 GPUs over time without
asking. Holding, releasing and editing jobs always need the user's
approval, and the guard hook enforces this. A job that should not run any
more is removed (`condor_rm <id>`; no approval needed for your own tagged
jobs), not parked.

When BAF holds a job on its own, the cause is usually: an expired Kerberos
ticket (the most common; see Access), a missing input file or log directory,
the job exceeding its memory or disk request, or `on_exit_hold` in the submit
file. Read `condor_q -hold <id>` and tell the user; after too many retries BAF
removes the job. `condor_hold -subcode 42 <id>` holds a job without automatic
release, but it still counts against the budget until it is removed.

Before every GPU submission, count what is already held across all schedds:

```bash
condor_q -global "$USER" -af JobStatus RequestGPUs    # status 1 = idle, 2 = running
```

(Live 2026-11-26: prints nothing when you have no jobs, `undefined` as
`RequestGPUs` for CPU-only jobs, and answers in under a second.)

Add the new job's `request_gpus` times the number of procs it queues. If the
sum is above 2, tell the user both numbers and wait for a yes. CPU-only jobs
(`request_gpus = 0`) are outside this budget.

Interactive jobs need a terminal, so they are for the user, not the agent.
The agent builds environments as a short batch job
(`references/training-jobs.md`, section 1). For the user's own use:

```bash
condor_submit -interactive \
  -append '+ContainerOS = "Rocky9"' -append '+CephFS_IO = "medium"' \
  -append '+MaxRuntimeHours = 2' -append 'request_gpus = 1' \
  -append 'requirements = (CUDADeviceName == "NVIDIA H200")' \
  -append 'request_cpus = 24' -append 'request_memory = 32000 MB'
```

For batch jobs, multi-GPU training and the environment tarball, read
`references/training-jobs.md`.

## Rules that are specific to BAF

- **Never set `CUDA_VISIBLE_DEVICES` yourself.** GPUs are not isolated by
  cgroups: `nvidia-smi` in a job shows every GPU on the node, and a manual
  setting can put your job on someone else's GPU. HTCondor exports short GPU
  UUIDs (`GPU-af7b61d8`), which PyTorch and `torchrun` accept. vLLM does not.
  Every multi-GPU or vLLM job runs `source /etc/profile;
  expand_cuda_visible_devices` first (`references/training-jobs.md`,
  section 4) - the only allowed rewrite.
- **Do not override thread-count variables** (`OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, ...). HTCondor sets them to the reserved cores (a
  1-CPU job saw `OMP_NUM_THREADS=1`). The job is **not** pinned to its cores:
  CPU affinity and `nproc --all` show the whole machine (32 cores on `wn099`,
  2026-11-26), and plain `nproc` only looks right because it honours
  `OMP_NUM_THREADS`. Size thread pools from `request_cpus`, never from the
  core count.
- **Do not list CephFS paths in `transfer_input_files` or
  `transfer_output_files`.** Everything left at the top level of `/jwd` is
  transferred back to the submit directory; set `transfer_output_files = ""`
  and put temporary files in a subdirectory.
- **Large downloads run on a login node, not in a job**, writing straight to
  `$BUDDY` (the group guide names `desktop12.physik.uni-bonn.de`), inside
  `screen`. Jobs share a gateway that limits outbound connections. This is the
  one sanctioned heavy task on a login node - it is network-bound, not
  CPU-bound; do not extend the exception to anything else. Ask the user before
  starting one, and never put the token on the command line.
- **Every job depends on the Kerberos ticket** on the submit machine (see
  Access). Check `klist` before submitting; never submit a job that cannot
  finish before `renew until`.
- **Software**: PHY-IT prefers central installations (their CVMFS, LCGENV) and
  will install missing packages on request. For a venv, use their central
  Miniforge (`module load miniforge/<version>`, from CVMFS), never `conda init`
  or `mamba init`, never Anaconda/Miniconda or the `defaults` channel. Build the
  env in `/jwd` inside a job and persist it as one tarball on `$BUDDY` (see
  `references/training-jobs.md`). Desktop and container OS must match if you
  build on the desktop.
- **Per-OS shell setup** goes in `$BUDDY/.bashrc_Rocky9` (etc.), sourced at the
  start of interactive jobs and of batch jobs that `source /etc/profile`.
  A broken one stops jobs from starting; fix it from the desktop.
- **Python output**: run `python -u` (or flush) so `condor_tail` shows
  progress; redirected output is otherwise buffered.
- **HTCondor Python bindings**: use the ones installed on the desktops, not
  the PyPI package (Kerberos authentication problems).
- `git clone` directly into `$BUDDY` fails. Clone in home, copy to `$BUDDY`,
  then run `git config fetch.unpackLimit 10000` in that copy so later
  fetch/pull works. Home is not visible in jobs, so the copy on `$BUDDY` is
  the one jobs use: keep the two in sync deliberately.
- The executable is renamed to `condor_exec.exe`; do not rely on `$0`.

## "My job disappeared"

Each desktop has its own schedd. With the fixed `baf` host this should not
happen, but a job submitted by the user from another desktop is only visible
there. Before concluding a job is gone, look at its log, then ask all schedds
at once (run once, not in a loop):

```bash
condor_q -global "$USER"
```

or, to see which schedd holds what:

```bash
for sd in $(condor_status -schedd -af Name); do condor_q "$USER" -name "$sd"; done
```

`condor_rm`, `condor_release` and `condor_qedit` also need `-name <schedd>`
when you are on a different login node. Record the schedd (the submit host)
together with the ClusterId whenever you submit.
