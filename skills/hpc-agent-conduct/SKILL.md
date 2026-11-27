---
name: hpc-agent-conduct
description: Rules of conduct for coding agents on shared HPC clusters. Load this before running ANY command on or against a cluster (directly on a login node, inside a job, or over ssh) - including file searches, environment setup, job submission, job monitoring and cleanup. Covers login-node limits, parallel-filesystem I/O (Lustre, CephFS), file-count and quota hygiene, scheduler polling, and what needs the user's approval. Applies to Marvin, Bender and BAF at Uni Bonn and to any other Slurm or HTCondor cluster.
---

# Conduct for agents on shared HPC clusters

A cluster is shared by hundreds of people and has few technical guard rails.
Habits that are harmless on a laptop (search everything, install freely, poll
in a loop, start a test run right here) are the ones that take down login nodes
and metadata servers. The user is accountable for everything you do under their
account, and admins kill offending processes without notice.

Layering: this skill holds the rules that apply everywhere. Load the scheduler
skill (`slurm` or `htcondor`) for mechanics and the cluster skill (`marvin`,
`bender`, `baf`) for paths, limits and local policy. Where they disagree, the
cluster skill wins, and the site's own documentation wins over all of them.

## 0. Know where you are before you act

Run this once per session and keep the answer in mind:

```bash
hostname; echo "slurm=${SLURM_JOB_ID:-none} condor=${_CONDOR_SCRATCH_DIR:-none}"
```

**The agent runs on the user's workstation and reaches the cluster only through
`ssh host 'command'`.** It is never installed or started on a cluster, not on a
login node and not inside a job. That keeps the agent's own process, session
files and file indexing off the shared system. If you find yourself running
directly on a cluster host, stop and tell the user.

- Every `ssh host 'cmd'` runs on a **login node**: edit, git, small builds,
  submit, inspect. Nothing else.
- Code that runs **inside a job** owns the cores, memory and GPUs the job was
  given, and nothing more.
- Edit code locally and sync it (`git push`/`pull`, or `rsync` of the project
  directory only). Do not mount cluster filesystems on the workstation
  (`sshfs`, NFS): local search and indexing would then walk the cluster's
  filesystem.
- Interactive jobs need a terminal, which you do not have. When a step needs
  one (building an environment on a GPU node, for example), prepare the script
  and ask the user to run it, or wrap it in a short batch job.

For ssh-driven work, reuse one connection instead of opening a new login per
command. Ask the user to add this to `~/.ssh/config` if it is missing:

```text
Host marvin bender baf
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
```

## 1. Login nodes are for editing and submitting

Never start compute on a login node: no training or inference, no test suites
that load models or data, no `torchrun`/`mpirun`, no Jupyter or model servers,
no large parallel builds (`make -j` beyond a few cores), no data preprocessing.
"The queue is slow" is not a reason. If something needs more than about a
minute of one core, it goes into a job - use the shortest debug queue or an
interactive job.

Leave nothing behind. Do not start background servers, watchers or daemons on a
login node. If you started a `tmux`/`screen` session or background process,
stop it when the task ends, and say so if you could not.

## 2. Treat the shared filesystem as a slow, shared database

Every `stat`, `open` and directory listing is a request to a metadata server
that all users share.

- **Never walk a shared tree.** No `find`, `du`, `tree`, `ls -R`, `grep -r`,
  `rg`, `chmod -R` or `chown -R` on a filesystem root, a group directory, or
  another user's directory. The same applies to your built-in search tools:
  always give Glob and Grep an explicit path inside the current project, never
  a workspace root or `$HOME`.
- **Bound every search**: exact subdirectory, `-maxdepth`, a file-type filter.
  If you do not know where something is, ask the user instead of scanning.
- **Sizes and counts come from the filesystem's own accounting**, not from
  `du`/`find | wc -l`: `lfs quota` on Lustre, `getfattr -n ceph.dir.rentries`
  and `ceph.dir.rbytes` on CephFS, `quota -s` where available.
- Use plain `ls`; `ls -l` on a large Lustre directory is expensive.
- Do not point file watchers, IDE indexers or language servers at data or
  workspace directories.

## 3. Keep file counts low

Small files are the scarce resource, more than bytes. Some sites store small
files on the metadata servers themselves, others enforce a hard file-count
quota.

- **Environments**: a Python venv or conda env is 20k-100k files. Create at
  most one per project and stack, reuse it, and never create one "to try
  something". Prefer site modules, a container image, or a venv packed into one
  tarball that the job unpacks onto node-local disk.
- **Caches**: pip, uv, conda, Hugging Face, wandb and torch caches default to
  `$HOME` and grow without limit. Point them at a deliberate location
  (node-local disk for throwaway, a workspace for reuse) and clean them
  (`pip cache purge`, `conda clean --all`).
- **Outputs**: no file-per-sample or file-per-step. Use Parquet, HDF5,
  WebDataset/tar shards. Keep the last N checkpoints, not all of them. Log to
  one file per job.
- **High-churn or tiny-file I/O belongs on node-local disk** inside the job;
  copy results back as a few large files at the end.
- Never `git clone` or `npm install` large trees just to look at them.

## 4. Storage is finite and mostly not backed up

- Know the quota and lifetime of the directory you write to before writing.
- Do not duplicate datasets for convenience. Check whether the group already
  has a copy.
- Deleting is destructive and often expensive: **ask before any recursive
  delete**, show the path and what it contains, and use the site's recommended
  method (on Lustre: `munlink`, not `rm -r`).
- Never touch other users' or other groups' files, even where permissions
  allow it.

## 5. Scheduler etiquette

- **Show the job script and the resource request, then ask before submitting,
  cancelling, holding or requeueing**, unless the cluster skill grants a
  standing exception (BAF and Marvin: submissions within the limits). State
  partition/queue, time, cores, memory, GPUs and where the output goes.
- **Tag every job you submit as the agent's** - Slurm `--comment=claude-agent`
  (`#SBATCH --comment=claude-agent`), HTCondor `+ClaudeAgent = True` in the
  submit file. You may then remove your own tagged jobs by their explicit ID
  without asking (the guard hook checks the tag); report what you removed.
  Never remove a job you did not submit, or by filter (`-u`, `--name`,
  `-all`, `-constraint`, a user name), without the user's approval.
- **Check before you request**: confirm the partition, account, limits and free
  resources with the scheduler's own commands. Never guess a partition name.
- **Request the minimum that does the job.** A GPU budget is a ceiling, not a
  target. Start with one GPU; add more only for a stated reason (the model
  or batch does not fit in one GPU's memory, or a measured speed-up the user
  needs), and say that reason when you submit. A job holding four GPUs and
  running on one blocks three for everyone.
- **Verify every requested GPU is used** in the first minutes: per-GPU
  utilisation and memory from `nvidia-smi` (on shared nodes, only the rows
  for your own GPUs). If some sit idle, the launch is wrong (for example
  `--nproc_per_node` or tensor-parallel size not matching the request):
  stop the job, fix it, and resubmit smaller or correctly.
- **Smoke-test small first**: debug queue, minutes, a tiny slice of the data.
  For anything long, the smoke test must also write a checkpoint and resume
  from it. Only then submit the real run.
- **Many similar tasks are one array job** with a concurrency cap, not hundreds
  of separate submissions. Never generate jobs in a loop.
- **Do not poll.** No `watch`, no loops around `squeue`/`sacct`/`condor_q`.
  Prefer a blocking wait (`sbatch --wait`, `condor_wait`) or read the job's own
  log file. If you must check status, query only your own jobs, at most once a
  minute for the first few minutes and every 5-10 minutes after that.
- **Submission limits, even when the user has waived per-job approval.** Stop
  and ask when any of these would be exceeded: more than 6 submissions in an
  hour; an array or `queue N` of more than 20 tasks without a concurrency cap;
  more than 50 of your jobs queued; or the cluster skill's GPU budget. Count
  what you already hold before every submission - the budget is about the
  total, not about the new job. Held and pending jobs count as held: never use
  hold/release, delayed starts or edits of the resource request to get round a
  limit in steps.
- One agent session per user per cluster. Do not fan out sub-agents that each
  open their own connection.
- Long jobs must checkpoint and be able to resume.

### Write submit commands the guard hook can read

The guard hook checks every submission **before the command runs**: it reads
the job file to count GPUs, tasks and runtime. A command it cannot read is
sent to the user for approval even when the job is small, which stops
autonomous work. Shape submissions so they pass on their own:

- **The job file must exist before the submitting command runs.** Create,
  edit, copy or generate it in one tool call (Write, `scp`, `sed`, `cp`, ...),
  and submit in the **next** call. Never `sed ... > job.jdl && condor_submit
  job.jdl`, `cp a.sh job.sh && sbatch job.sh` or `python gen.py && sbatch
  job.sh` in one command: the hook treats a file changed by the same command
  as unknown and asks.
- Two same-command forms the hook can read: a **here-document** that writes
  the file (`cat > job.jdl <<'EOF' ... EOF`, then submit), or an `scp`/`rsync`
  of an **unchanged local file** followed by the submit (the hook reads the
  local copy).
- **Literal resource numbers** in the file or on the command line:
  `request_gpus = 1`, `--gpus=1`, `queue 10`. No `$(N)` macros or shell
  variables for GPUs, tasks or `MaxRuntimeHours`.
- **One submit per command**, never in a loop (use an array or `queue N`
  with a cap). Name the file explicitly: `cd <dir> && condor_submit job.jdl`
  or an absolute path, inside or outside `bash -lc "..."`.
- Keep other work (copying outputs, waiting, reading logs) out of the
  submitting command where you can; a short submit command is easy for the
  user to review if it does get asked.

When the hook asks anyway, read its reason before involving the user:

- *"written or changed by this same command"*, *"Could not read from the
  submit file ..."* or *"Could not work out how many GPUs ..."*: your command
  shape is the problem (or the file is missing, or uses a macro for GPUs). Split it as above and
  retry; do not ask the user to approve it.
- A **limit** (GPU budget, submissions per hour, queue size, uncapped array,
  missing test run): that is a real decision. Stop, tell the user the
  numbers, and wait.

## 6. Software

- Look for a site module first (`module spider <name>`); ask before installing
  anything large.
- Install on the same hardware the job will run on (GPU variant, CPU
  architecture). Login nodes usually have no GPU, so installers pick CPU-only
  builds there.
- No `sudo`, no system package managers, no Docker. Respect licence rules (for
  example, many sites forbid Anaconda's `defaults` channel).
- Do not edit `~/.bashrc`, `~/.profile` or module defaults without asking.

## 7. Secrets and data

- Never read, print or copy `~/.ssh`, Kerberos tickets, `.netrc`, `.env`,
  token files or API keys. Never put a token on a command line or in a job
  script; reference an environment variable the user has set.
- Before reading data files into context, consider whether the content may
  leave the site. Do not open unpublished or personal data unless the user
  asked for it.

## 8. Always confirm first

Ask the user before: submitting jobs (unless the cluster skill grants an
exception), cancelling jobs you did not submit, recursive deletes, changing
permissions or ownership, creating a new environment, downloads or copies above
a few GB, releasing or creating workspaces, editing shell startup files, and
anything that affects other users.

Never use a permission-bypass mode on a cluster.

## 9. When unsure

Read the cluster skill, then the site's documentation, then ask the user.
Report exactly what you ran: commands, job IDs, output paths. If something
misbehaves (a job storm, a runaway process, a filled quota), stop, clean up
your own processes, and tell the user immediately so they can inform support.
