#!/usr/bin/env python3
"""Tests for hpc_guard.py. Run: python3 hooks/test_hpc_guard.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpc_guard.py")

LOCAL = {"HPC_GUARD_LOGIN_REGEX": "^never-matches$"}          # workstation
LOGIN = {"HPC_GUARD_LOGIN_REGEX": "."}                        # on a login node
INJOB = {"HPC_GUARD_LOGIN_REGEX": ".", "SLURM_JOB_ID": "42"}  # inside an allocation

CASES = [
    # (context, command, expected decision or None for "no opinion")
    (LOCAL, "python train.py", None),
    (LOCAL, "find / -name '*.py'", None),
    (LOCAL, "ssh marvin 'squeue --me'", None),
    (LOCAL, "ssh marvin 'torchrun --nproc_per_node=4 train.py'", "deny"),
    (LOCAL, "ssh -o BatchMode=yes me_hpc@marvin.hpc.uni-bonn.de python train.py", "ask"),
    (LOCAL, "ssh baf 'cd ~/scripts && condor_submit job.jdl'", "ask"),   # budget not checkable here
    (LOCAL, "ssh marvin 'find /lustre/scratch/data -name \"*.pt\"'", "deny"),
    (LOCAL, "ssh marvin 'ls /lustre/scratch/data/me-run1'", None),
    (LOCAL, "ssh marvin 'watch -n 2 squeue --me'", "deny"),
    (LOCAL, "ssh marvin 'while true; do squeue --me; sleep 5; done'", "deny"),
    (LOCAL, "ssh marvin 'while true; do squeue --me; sleep 300; done'", None),
    (LOCAL, "ssh example.org 'torchrun train.py'", None),
    (LOCAL, "ssh baf 'du -sh /cephfs/user'", "deny"),
    (LOCAL, "ssh marvin 'rm -rf /lustre/scratch/data/me-old'", "ask"),
    (LOCAL, "ssh marvin 'rm -rf /lustre/scratch'", "deny"),
    (LOCAL, "ssh bender 'bash Miniconda3-latest-Linux-x86_64.sh -b'", "deny"),
    (LOGIN, "torchrun train.py", "deny"),
    (LOGIN, "CUDA_LAUNCH_BLOCKING=1 nohup accelerate launch train.py &", "deny"),
    (LOGIN, "grep torchrun job.sh", None),
    (LOGIN, "cat train.py | head", None),
    (LOGIN, "python --version", None),
    (LOGIN, "python -m venv .venv", None),
    (LOGIN, "python preprocess.py", "ask"),
    (LOGIN, "pytest tests/", "ask"),
    (LOGIN, "sbatch job.sh", "ask"),
    (LOGIN, "srun --pty torchrun train.py", "ask"),
    (LOGIN, "scancel 12345", "ask"),
    (LOGIN, "squeue --me", None),
    (LOGIN, "grep -rn foo /lustre/mlnvme/data", "deny"),
    (LOGIN, "grep -rn foo src/", None),
    (LOGIN, "du -sh $HOME", None),
    (LOGIN, "du -sh /home", "deny"),
    (LOGIN, "chmod -R g+r /lustre/scratch/data/", "deny"),
    (LOGIN, "make -j64", "ask"),
    (LOGIN, "make -j4", None),
    (LOGIN, "conda install -c defaults numpy", "deny"),
    (LOGIN, "conda install -c conda-forge numpy", None),
    (LOGIN, "bash -c 'mpirun -np 64 ./a.out'", "deny"),
    (LOGIN, "bash -lc 'mpirun -np 64 ./a.out'", "deny"),
    (LOCAL, "ssh marvin 'bash -lc \"torchrun x.py\"'", "deny"),
    (LOCAL, "ssh marvin 'bash -l -c \"find /lustre/scratch/data -name x\"'", "deny"),
    (LOCAL, "ssh baf 'bash -lc \"echo \\$BUDDY\"'", None),
    (LOGIN, "find /cephfs/user/me/logs -maxdepth 1 -name 'out-*' | xargs tail -n 5", None),
    (INJOB, "torchrun train.py", None),
    (INJOB, "python train.py", None),
    (INJOB, "find /lustre -name '*.ckpt'", "deny"),
    (INJOB, "watch squeue", "deny"),
    # the documented BAF "ask every schedd once" loop is not polling
    (LOGIN, 'for sd in $(condor_status -schedd -af Name); do condor_q "$USER" -name "$sd"; done', None),
    (LOGIN, "for i in $(seq 100); do sbatch job.sh $i; done", "deny"),
    (LOCAL, "ssh baf 'for f in *.jdl; do condor_submit $f; done'", "deny"),
    (LOGIN, "ls *.sh | xargs -n1 sbatch", "deny"),
    # hold/release/edit: no staging jobs around the GPU budget
    (LOGIN, "condor_release 123", "ask"),
    (LOGIN, "condor_hold -subcode 42 123", "ask"),
    (dict(LOGIN, HPC_GUARD_CONFIRM_JOBS="0"), "condor_qedit 123 RequestGPUs 4", "ask"),
    (dict(LOGIN, HPC_GUARD_CONFIRM_JOBS="0"), "condor_qedit 123 request_gpus 1", "ask"),
    (dict(LOGIN, HPC_GUARD_CONFIRM_JOBS="0"), "condor_qedit 123 RequestMemory 4000", None),
    (LOGIN, "scontrol hold 123", "ask"),
    (LOGIN, "scontrol release 123", "ask"),
    (dict(LOGIN, HPC_GUARD_CONFIRM_JOBS="0"), "scontrol update jobid=123 gres=gpu:4", "ask"),
    (dict(LOGIN, HPC_GUARD_CONFIRM_JOBS="0"), "scontrol update jobid=123 timelimit=2:00:00", None),
    (LOGIN, "scontrol show job 123", None),
]

# Submission limits, with per-job confirmation switched off. Fake scheduler
# commands stand in for the cluster: held = "JobStatus RequestGPUs" rows.
QUIET = dict(LOGIN, HPC_GUARD_CONFIRM_JOBS="0")
CONFIRM = dict(LOGIN)
REMOTE = dict(LOCAL, HPC_GUARD_CONFIRM_JOBS="0")
JDL = "executable = run.sh\nrequest_gpus = %d\n%squeue %s\n"
SBATCH = "#!/bin/bash\n#SBATCH --time=1:00:00\n%s\nsrun true\n"

LIMIT_CASES = [
    # (context, held rows, slurm queued, file content, command, expected)
    (QUIET, "", 0, JDL % (2, "", "1"), "condor_submit {f}", None),
    (QUIET, "", 0, JDL % (3, "", "1"), "condor_submit {f}", "ask"),
    (QUIET, "2 1\n", 0, JDL % (1, "", "1"), "condor_submit {f}", None),
    (QUIET, "2 2\n", 0, JDL % (1, "", "1"), "condor_submit {f}", "ask"),
    (QUIET, "1 1\n2 1\n", 0, JDL % (1, "", "1"), "condor_submit {f}", "ask"),      # idle jobs count
    (QUIET, "5 8\n", 0, JDL % (0, "", "1"), "condor_submit {f}", None),             # CPU-only stays free
    (QUIET, "5 1\n", 0, JDL % (2, "", "1"), "condor_submit {f}", "ask"),            # held GPUs count (BAF re-releases them)
    (QUIET, "5 1\n5 1\n", 0, JDL % (1, "hold = True\n", "1"), "condor_submit {f}", "ask"),  # no parking jobs on hold
    (QUIET, "2 8\n", 0, JDL % (0, "", "1"), "condor_submit {f}", None),             # CPU-only is free
    (QUIET, "2 undefined\n" * 10, 0, JDL % (0, "", "5"), "condor_submit {f}", None),
    # formats seen on the real BAF pool (2026-11-26): CPU-only jobs print "undefined"
    (QUIET, "1 undefined\n2 undefined\n2 1\n", 0, JDL % (1, "", "1"), "condor_submit {f}", None),
    (QUIET, "1 undefined\n2 1\n1 1\n", 0, JDL % (1, "", "1"), "condor_submit {f}", "ask"),
    (QUIET, "", 0, JDL % (1, "", "4"), "condor_submit {f}", "ask"),                  # 4 x 1 GPU
    (QUIET, "", 0, JDL % (1, "max_materialize = 2\n", "40"), "condor_submit {f}", None),
    (QUIET, "", 0, JDL % (0, "", "500"), "condor_submit {f}", "ask"),                # uncapped
    (QUIET, "", 0, JDL % (0, "", "Item from items.txt"), "condor_submit {f}", "ask"),
    (QUIET, "2 0\n" * 45, 0, JDL % (0, "", "10"), "condor_submit {f}", "ask"),      # queue size
    (QUIET, "", 0, JDL % (1, "", "1"), "condor_submit -append 'request_gpus = 4' {f}", "ask"),
    (QUIET, "2 2\n", 0, None, "condor_submit -interactive -append 'request_gpus = 1'", "ask"),
    (QUIET, "", 0, None, "condor_submit -interactive -append 'request_gpus = 1'", None),
    (QUIET, "", 0, None, "condor_submit /nonexistent/job.jdl", "ask"),               # unreadable
    (REMOTE, "2 2\n", 0, JDL % (1, "", "1"), "ssh baf 'condor_submit {f}'", "ask"),
    (REMOTE, "", 0, JDL % (2, "", "1"), "ssh baf 'cd {d} && condor_submit {b}'", None),
    # the recommended BAF form: login shell so getenv = BUDDY has a value
    (REMOTE, "", 0, JDL % (2, "", "1"), "ssh baf 'bash -lc \"cd {d} && condor_submit {b}\"'", None),
    (REMOTE, "", 0, JDL % (3, "", "1"), "ssh baf 'bash -lc \"cd {d} && condor_submit {b}\"'", "ask"),
    (REMOTE, "2 2\n", 0, JDL % (1, "", "1"), "ssh baf 'bash -lc \"cd {d} && condor_submit {b}\"'", "ask"),
    (QUIET, "", 0, SBATCH % "", "sbatch {f}", None),
    (QUIET, "", 0, SBATCH % "#SBATCH --array=0-199", "sbatch {f}", "ask"),
    (QUIET, "", 0, SBATCH % "#SBATCH --array=0-39%10", "sbatch {f}", None),
    (QUIET, "", 0, SBATCH % "", "sbatch --array=1-1000 {f}", "ask"),
    (QUIET, "", 45, SBATCH % "#SBATCH --array=0-19", "sbatch {f}", "ask"),
    (REMOTE, "", 60, SBATCH % "", "ssh marvin 'sbatch {f}'", "ask"),
    # per-job approval on (default): HTCondor within budget passes, Slurm asks
    (CONFIRM, "", 0, JDL % (2, "", "1"), "condor_submit {f}", None),
    (CONFIRM, "", 0, JDL % (0, "", "3"), "condor_submit {f}", None),
    (CONFIRM, "2 2\n", 0, JDL % (1, "", "1"), "condor_submit {f}", "ask"),
    (CONFIRM, "", 0, SBATCH % "", "sbatch {f}", None),                     # CPU-only, short queue
    # Marvin rule: 4 GPUs running + pending, multi-day queues always ask
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p sgpu_devel\n#SBATCH --gpus=4", "sbatch {f}", None),
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p sgpu_short\n#SBATCH --gpus=5", "sbatch {f}", "ask"),
    (CONFIRM, "sgpu_short cpu=64,node=1,billing=192,gres/gpu=2\n", 0, SBATCH % "#SBATCH -p mlgpu_short\n#SBATCH --gres=gpu:a40:2", "sbatch {f}", None),
    (CONFIRM, "sgpu_short cpu=64,node=1,billing=192,gres/gpu=2\n", 0, SBATCH % "#SBATCH -p mlgpu_short\n#SBATCH --gres=gpu:a40:3", "sbatch {f}", "ask"),
    (CONFIRM, "mlgpu_devel,sgpu_devel cpu=16,node=1,billing=32,gres/gpu=1\n" * 3, 0, SBATCH % "#SBATCH -p mlgpu_devel\n#SBATCH --gpus-per-node=1\n#SBATCH --nodes=2", "sbatch {f}", "ask"),
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p sgpu_short\n#SBATCH --gpus-per-task=1\n#SBATCH --ntasks=4", "sbatch {f}", None),
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p sgpu_short\n#SBATCH --gpus-per-task=1\n#SBATCH --ntasks=8", "sbatch {f}", "ask"),
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p mlgpu_devel\n#SBATCH --gpus=1\n#SBATCH --array=0-9%2", "sbatch {f}", None),
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p mlgpu_devel\n#SBATCH --gpus=1\n#SBATCH --array=0-9%5", "sbatch {f}", "ask"),
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p sgpu_short", "sbatch --gpus=8 {f}", "ask"),       # command line counts
    # multi-day queues need a completed devel/short run with the same job name
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p sgpu_long\n#SBATCH --gpus=1", "sbatch {f}", "ask"),
    (CONFIRM, "", 0, SBATCH % "#SBATCH --partition=intelsr_medium", "sbatch {f}", "ask"),
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p sgpu_short", "sbatch -p sgpu_long {f}", "ask"),
    (dict(CONFIRM, FAKE_HISTORY="train|mlgpu_devel\n"), "", 0, SBATCH % "#SBATCH -p mlgpu_long\n#SBATCH -J train\n#SBATCH --gpus=4", "sbatch {f}", None),
    (dict(CONFIRM, FAKE_HISTORY="job.sh|sgpu_short\n"), "", 0, SBATCH % "#SBATCH -p sgpu_medium\n#SBATCH --gpus=2", "sbatch {f}", None),
    (dict(CONFIRM, FAKE_HISTORY="other|mlgpu_devel\n"), "", 0, SBATCH % "#SBATCH -p mlgpu_long\n#SBATCH -J train", "sbatch {f}", "ask"),
    (dict(CONFIRM, FAKE_HISTORY="train|mlgpu_long\n"), "", 0, SBATCH % "#SBATCH -p mlgpu_long\n#SBATCH -J train", "sbatch {f}", "ask"),
    (dict(CONFIRM, FAKE_HISTORY="train|mlgpu_devel\n"), "", 0, SBATCH % "#SBATCH -p mlgpu_long\n#SBATCH -J train\n#SBATCH --gpus=5", "sbatch {f}", "ask"),
    (CONFIRM, "", 0, None, "sbatch /nonexistent/job.sh", "ask"),
    (CONFIRM, "", 0, SBATCH % "#SBATCH -p sgpu_short\n#SBATCH --gres=gpu:a100:4", "sbatch {f}", None),
    # removing jobs: own tagged jobs pass, everything else asks
    (CONFIRM, "123|claude-agent\n", 0, None, "scancel 123", None),
    (CONFIRM, "123_1|claude-agent\n123_2|claude-agent\n", 0, None, "scancel 123", None),
    (CONFIRM, "123_1|claude-agent\n123_2|\n", 0, None, "scancel 123", "ask"),
    (CONFIRM, "123|my own run\n", 0, None, "scancel 123", "ask"),
    (CONFIRM, "123|claude-agent\n", 0, None, "scancel -u $USER", "ask"),
    (CONFIRM, "123|claude-agent\n", 0, None, "scancel --name train", "ask"),
    (CONFIRM, "213 0 true\n213 1 true\n", 0, None, "condor_rm 213", None),
    (CONFIRM, "213 0 true\n", 0, None, "condor_rm -name submit01.physik.uni-bonn.de 213.0", None),
    (CONFIRM, "213 0 true\n213 1 undefined\n", 0, None, "condor_rm 213", "ask"),
    (CONFIRM, "213 0 true\n", 0, None, "condor_rm -all", "ask"),
    (CONFIRM, "213 0 true\n", 0, None, "condor_rm -constraint 'JobStatus == 1'", "ask"),
    (CONFIRM, "213 0 true\n", 0, None, "condor_rm someuser", "ask"),
    (CONFIRM, "213 0 true\n", 0, None, "condor_hold 213", "ask"),
    (dict(CONFIRM, HPC_GUARD_AUTO_SUBMIT=""), "", 0, JDL % (1, "", "1"), "condor_submit {f}", "ask"),
]

FAKES = {
    "condor_q": '#!/bin/bash\nprintf "%b" "$FAKE_HELD"\n',
    "condor_history": '#!/bin/bash\nprintf "%b" "$FAKE_CHISTORY"\n',
    "sacct": '#!/bin/bash\nprintf "%b" "$FAKE_HISTORY"\n',
    "squeue": '#!/bin/bash\nprintf "%b" "$FAKE_HELD"\nfor i in $(seq 1 "$FAKE_QUEUED"); do echo "intelsr_short cpu=1,mem=1G,node=1"; done\n',
    # stands in for the cluster login node: run the remote command locally
    "ssh": '#!/bin/bash\nexec bash -c "${@: -1}"\n',
}


def run(env_extra, command, state_dir=None):
    env = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "_CONDOR_SCRATCH_DIR")}
    env["HPC_GUARD_STATE_DIR"] = state_dir or tempfile.mkdtemp(prefix="hpc-guard-test-")
    env.update(env_extra)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    out = subprocess.run([sys.executable, HOOK], input=payload, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    if not out.stdout.strip():
        return None
    return json.loads(out.stdout)["hookSpecificOutput"]["permissionDecision"]


def main():
    failed = 0
    for env_extra, command, expected in CASES:
        got = run(env_extra, command)
        if got != expected:
            failed += 1
            print("FAIL  expected %-5s got %-5s  %s" % (expected, got, command))

    tmp = tempfile.mkdtemp(prefix="hpc-guard-test-")
    bindir = os.path.join(tmp, "bin")
    os.mkdir(bindir)
    for name, body in FAKES.items():
        path = os.path.join(bindir, name)
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
    for n, (ctx, held, queued, content, command, expected) in enumerate(LIMIT_CASES):
        workdir = os.path.join(tmp, "case%d" % n)
        os.mkdir(workdir)
        name = "job.jdl" if "condor" in command else "job.sh"
        if content is not None:
            with open(os.path.join(workdir, name), "w") as fh:
                fh.write(content)
        command = command.format(f=os.path.join(workdir, name), d=workdir, b=name)
        env = dict(ctx, PATH=bindir + os.pathsep + os.environ["PATH"], FAKE_HELD=held, FAKE_QUEUED=str(queued))
        got = run(env, command)
        if got != expected:
            failed += 1
            print("FAIL  expected %-5s got %-5s  %s  [held=%r queued=%d]" % (expected, got, command, held, queued))

    # rate limit: the 7th submission within an hour asks even though each job is small
    state = os.path.join(tmp, "state")
    env = dict(QUIET, PATH=bindir + os.pathsep + os.environ["PATH"], FAKE_HELD="", FAKE_QUEUED="0")
    script = os.path.join(tmp, "ratelimit.sh")
    with open(script, "w") as fh:
        fh.write(SBATCH % "")
    results = [run(env, "sbatch " + script, state) for _ in range(7)]
    if results != [None] * 6 + ["ask"]:
        failed += 1
        print("FAIL  rate limit: %s" % results)
    # BAF: a job above 48 h needs a completed short test run with the same JobBatchName
    state = os.path.join(tmp, "state-tests")
    base = dict(CONFIRM, PATH=bindir + os.pathsep + os.environ["PATH"], FAKE_HELD="", FAKE_QUEUED="0")
    def jdl(name, hours, batch):
        path = os.path.join(tmp, name)
        with open(path, "w") as fh:
            fh.write("executable = run.sh\nrequest_gpus = 1\n+MaxRuntimeHours = %d\n%squeue 1\n" % (hours, batch))
        return path
    steps = [
        ("long job, no test on record", base, jdl("l1.jdl", 100, "JobBatchName = train\n"), "ask"),
        ("long job without a batch name", base, jdl("l2.jdl", 100, ""), "ask"),
        ("48 h is not long", base, jdl("m.jdl", 48, ""), None),
        ("short test run", base, jdl("t.jdl", 2, "JobBatchName = train\n"), None),
        ("long job, test not completed", base, jdl("l1.jdl", 100, "JobBatchName = train\n"), "ask"),
        ("long job, test completed", dict(base, FAKE_CHISTORY="213800\n"), jdl("l1.jdl", 100, "JobBatchName = train\n"), None),
        ("other name", dict(base, FAKE_CHISTORY="213800\n"), jdl("l3.jdl", 100, "JobBatchName = other\n"), "ask"),
    ]
    extra = 0
    for label, env, path, expected in steps:
        extra += 1
        got = run(env, "condor_submit " + path, state)
        if got != expected:
            failed += 1
            print("FAIL  expected %-5s got %-5s  BAF test gate: %s" % (expected, got, label))
    shutil.rmtree(tmp, ignore_errors=True)

    total = len(CASES) + len(LIMIT_CASES) + 1 + extra
    print("%d/%d passed" % (total - failed, total))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
