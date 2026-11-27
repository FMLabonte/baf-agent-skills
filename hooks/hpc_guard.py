#!/usr/bin/env python3
"""PreToolUse hook for the Bash tool: keeps an agent polite on shared HPC clusters.

It only acts when the command runs on a cluster: either this process is on a
cluster host, or the command is an `ssh <cluster> ...` call. Everything else
passes untouched.

Decisions:
  deny  compute launchers on a login node, walking a shared filesystem root,
        scheduler polling loops, Anaconda/defaults-channel installs
  ask   submitting/cancelling/holding/releasing jobs, python scripts on a login node,
        recursive deletes on Lustre/CephFS

Submission limits. These always ask, even with HPC_GUARD_CONFIRM_JOBS=0, and a
submit command inside a shell loop is denied outright:
  - more than HPC_GUARD_MAX_SUBMITS_PER_HOUR submit commands in an hour
  - a job array / `queue N` above HPC_GUARD_MAX_UNCAPPED tasks without a
    concurrency cap (`%N`, `max_materialize`)
  - more than HPC_GUARD_MAX_QUEUED jobs queued after the submission
  - HTCondor only: the GPUs you already hold (running + idle + held: BAF
    re-releases held jobs every 30 minutes) plus the new request exceed
    HPC_GUARD_GPU_BUDGET
  - HTCondor only: `condor_qedit` of a GPU request
  - HTCondor only: MaxRuntimeHours above HPC_GUARD_CONDOR_LONG_HOURS (48)
    without a completed test run (MaxRuntimeHours up to
    HPC_GUARD_CONDOR_TEST_HOURS, 8) of the same JobBatchName
  - Slurm only: GPUs in running + pending jobs plus the new request exceed
    HPC_GUARD_SLURM_GPU_BUDGET, or a multi-day queue without a completed
    devel/short test run of the same job name
Holding and releasing jobs (`condor_hold`/`condor_release`, `scontrol
hold/release/update/requeue`) asks like any other job change. Removing jobs
(`scancel`, `condor_rm`) by explicit ID needs no approval when every job is
tagged as the agent's own (Slurm `--comment=claude-agent`, HTCondor
`+ClaudeAgent = True`); anything else asks.
The last two need one scheduler query per submission (over ssh when the agent
runs off-cluster). If the query fails, the hook asks.

Configuration (environment variables, all optional):
  HPC_GUARD_SSH_HOSTS     regex of ssh host names/aliases that are clusters
  HPC_GUARD_LOGIN_REGEX   regex matched against this machine's hostname
  HPC_GUARD_CONFIRM_JOBS  "0" to stop asking before submit/cancel commands
  HPC_GUARD_AUTO_SUBMIT   submit programs that skip per-job approval while all
                          limits hold ("condor_submit,sbatch"; "" to ask for all)
  HPC_GUARD_MIN_POLL      minimum seconds between scheduler queries in a loop
  HPC_GUARD_GPU_BUDGET    GPUs one user may hold on HTCondor without asking (2)
  HPC_GUARD_SLURM_GPU_BUDGET  the same on Slurm, running + pending (4)
  HPC_GUARD_SLURM_LONG_PARTITIONS  regex of multi-day partitions ("_(medium|long)\b");
                          they ask unless a job with the same name completed
                          on a devel/short partition in the last
                          HPC_GUARD_TEST_RUN_DAYS days (7)
  HPC_GUARD_MAX_SUBMITS_PER_HOUR  submit commands per hour without asking (6)
  HPC_GUARD_MAX_UNCAPPED  array tasks allowed without a concurrency cap (20)
  HPC_GUARD_MAX_QUEUED    queued jobs per user without asking (50)
  HPC_GUARD_PROBE         "0" disables the scheduler query (then it asks)
  HPC_GUARD_STATE_DIR     where the submission log is kept

This is a seat belt, not a security boundary: it parses commands heuristically
and can be evaded. The skills carry the actual rules.
"""
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time

SSH_HOSTS = os.environ.get(
    "HPC_GUARD_SSH_HOSTS",
    r"^(marvin|bender|baf|.*\.hpc\.uni-bonn\.de|.*\.physik\.uni-bonn\.de)$",
)
LOGIN_REGEX = os.environ.get(
    "HPC_GUARD_LOGIN_REGEX",
    r"^(login\d+|marvin\S*|bender\S*|(desktop|theo|exp)\d+)(\.|$)|\.(hpc|physik)\.uni-bonn\.de$",
)
CONFIRM_JOBS = os.environ.get("HPC_GUARD_CONFIRM_JOBS", "1") != "0"
# Submit commands that need no per-job approval while every limit holds
# (the limits, incl. the GPU budget, still ask). Default: HTCondor/BAF only.
AUTO_SUBMIT = {p for p in os.environ.get("HPC_GUARD_AUTO_SUBMIT", "condor_submit,sbatch").split(",") if p}
MIN_POLL = int(os.environ.get("HPC_GUARD_MIN_POLL", "60"))
GPU_BUDGET = int(os.environ.get("HPC_GUARD_GPU_BUDGET", "2"))
SLURM_GPU_BUDGET = int(os.environ.get("HPC_GUARD_SLURM_GPU_BUDGET", "4"))
SLURM_LONG_PARTITIONS = os.environ.get("HPC_GUARD_SLURM_LONG_PARTITIONS", r"_(medium|long)\b")
TEST_RUN_DAYS = int(os.environ.get("HPC_GUARD_TEST_RUN_DAYS", "7"))
CONDOR_LONG_HOURS = int(os.environ.get("HPC_GUARD_CONDOR_LONG_HOURS", "48"))
CONDOR_TEST_HOURS = int(os.environ.get("HPC_GUARD_CONDOR_TEST_HOURS", "8"))
AGENT_TAG = os.environ.get("HPC_GUARD_AGENT_TAG", "claude-agent")
MAX_SUBMITS_PER_HOUR = int(os.environ.get("HPC_GUARD_MAX_SUBMITS_PER_HOUR", "6"))
MAX_UNCAPPED = int(os.environ.get("HPC_GUARD_MAX_UNCAPPED", "20"))
MAX_QUEUED = int(os.environ.get("HPC_GUARD_MAX_QUEUED", "50"))
PROBE = os.environ.get("HPC_GUARD_PROBE", "1") != "0"
STATE_DIR = os.environ.get("HPC_GUARD_STATE_DIR") or os.path.join(
    os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"), "hpc-guard"
)

# Paths that belong to everyone. A user's own subdirectory below them is fine.
SHARED_ROOT = re.compile(
    r"^(/|/home|/lustre(/(scratch|mlnvme|project)(/data)?)?"
    r"|/cephfs(/user)?|/opt/software|/software|/local/nvme)/?\*?$"
)
SHARED_PREFIX = re.compile(r"^(/lustre|/cephfs)/")

COMPUTE = {
    "torchrun", "mpirun", "mpiexec", "orterun", "vllm", "ollama", "tensorboard",
    "jupyter", "jupyter-lab", "jupyter-notebook", "deepspeed",
}
QUERY = {"squeue", "sacct", "sinfo", "sstat", "condor_q", "condor_status", "condor_history"}
JOB_CONTROL = {
    "sbatch", "srun", "salloc", "scancel", "condor_submit", "condor_submit_dag",
    "condor_rm", "condor_hold", "condor_release", "condor_qedit",
}
SUBMIT = {"sbatch", "condor_submit", "condor_submit_dag"}
SCONTROL_CHANGE = {"hold", "uhold", "release", "update", "requeue", "requeuehold", "suspend", "resume"}
JOB_FILE = re.compile(r"\.(sh|bash|sbatch|slurm|jdl|sub|submit|condor|dag)$")
WRAPPERS = {"nohup", "time", "env", "exec", "nice", "ionice", "command", "stdbuf", "setsid", "timeout"}
SSH_ARG_OPTS = set("bcDEeFIiJLlmOopQRSWw")
ANACONDA = re.compile(
    r"repo\.anaconda\.com|Anaconda3-|Miniconda3-|module\s+load\s+Anaconda"
    r"|(-c|--channel)[ =]+(defaults|anaconda|main)\b"
)
LOOP = re.compile(r"\b(while|until)\b[\s\S]*\bdo\b[\s\S]*\b(%s)\b" % "|".join(sorted(QUERY)))
SUBMIT_LOOP = re.compile(
    r"(\b(while|until|for)\b[\s\S]*\bdo\b|\bxargs\b|\bparallel\b)[\s\S]*\b(%s)\b" % "|".join(sorted(SUBMIT))
)


def respond(decision, reason):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": "[hpc-guard] " + reason,
        }
    }))
    sys.exit(0)


HEREDOC_MARK = "__hpcguard_heredoc_%d__"
HEREDOC_OP = re.compile(r"<<(-?)[ \t]*(?:'([^'\n]*)'|\"([^\"\n]*)\"|\\?([A-Za-z_][\w.-]*))")
SHELLS = ("bash", "sh", "zsh", "dash", "ksh")


def split_heredocs(cmd):
    """Take here-document bodies (outside quotes) out of a command line.

    Returns the command with each body replaced by a marker token right after its
    delimiter (HEREDOC_MARK % index), and the list of bodies. Also drops comments
    and joins backslash-newline continuations, so that newlines can act as
    command separators afterwards."""
    out, bodies, pending = [], [], []
    i, n, quote = 0, len(cmd), None
    while i < n:
        c = cmd[i]
        if quote:
            out.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                out.append(cmd[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            out.append(" " if cmd[i + 1] == "\n" else cmd[i:i + 2])
            i += 2
            continue
        if c in "'\"":
            quote = c
        elif c == "#" and (i == 0 or cmd[i - 1] in " \t\n;&|("):
            j = cmd.find("\n", i)
            i = n if j < 0 else j  # keep the newline itself
            continue
        elif cmd.startswith("<<", i) and not cmd.startswith("<<<", i):
            m = HEREDOC_OP.match(cmd, i)
            if m:
                word = next((g for g in m.groups()[1:] if g is not None), "")
                out.append(cmd[i:m.end()] + " " + HEREDOC_MARK % len(bodies) + " ")
                pending.append((len(bodies), word, m.group(1) == "-"))
                bodies.append("")
                i = m.end()
                continue
        elif c == "\n" and pending:
            out.append(c)
            i += 1
            for idx, word, tabs in pending:
                lines = []
                while i < n:
                    j = cmd.find("\n", i)
                    j = n if j < 0 else j
                    line, i = cmd[i:j], j + 1
                    if (line.lstrip("\t") if tabs else line) == word:
                        break
                    lines.append(line)
                bodies[idx] = "\n".join(lines)
            pending = []
            continue
        out.append(c)
        i += 1
    return "".join(out), bodies


def heredoc_body(tok, bodies):
    m = re.match(r"^__hpcguard_heredoc_(\d+)__$", tok)
    return bodies[int(m.group(1))] if m and int(m.group(1)) < len(bodies) else None


def without_heredocs(tokens, bodies):
    """(tokens minus every `<<` DELIM MARKER triple, body of the first here-document or None)."""
    marks = [k for k, t in enumerate(tokens) if heredoc_body(t, bodies) is not None]
    drop = {j for k in marks for j in (k - 2, k - 1, k)}
    rest = [t for j, t in enumerate(tokens) if j not in drop]
    return rest, (heredoc_body(tokens[marks[0]], bodies) if marks else None)


def tokenize(cmd):
    """Shell-ish tokens; a newline outside quotes is its own token (a separator)."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars="();<>|&\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = ""  # comments are removed by split_heredocs
    try:
        return list(lex)
    except ValueError:
        return re.findall(r"\n|[^\s]+", cmd)


def segments(tokens):
    seg = []
    for tok in tokens:
        if tok and set(tok) <= set(";&|()\n"):
            if seg:
                yield seg
            seg = []
        else:
            seg.append(tok)
    if seg:
        yield seg


def strip_prefix(seg):
    """Drop VAR=value assignments and wrapper commands in front of the program."""
    i = 0
    while i < len(seg):
        tok = seg[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            i += 1
        elif os.path.basename(tok) in WRAPPERS:
            i += 1
            while i < len(seg) and (seg[i].startswith("-") or re.match(r"^\d+[smhd]?$", seg[i])):
                i += 1
        else:
            break
    return seg[i:]


def ssh_remote(seg, bodies=()):
    """For `ssh [opts] host cmd...` return (host, target, remote command string).

    A here-document fed to a bare remote shell (`ssh host bash -s <<EOF`, or
    `ssh host <<EOF`) is the remote command."""
    i = 1
    while i < len(seg):
        tok = seg[i]
        if tok.startswith("-") and len(tok) > 1:
            i += 2 if (len(tok) == 2 and tok[1] in SSH_ARG_OPTS) else 1
        else:
            break
    if i >= len(seg):
        return None, None, ""
    rest, stdin = without_heredocs(seg[i + 1:], bodies)
    remote = " ".join(rest)
    if stdin is not None and (not rest or os.path.basename(rest[0]) in SHELLS):
        remote = stdin
    return seg[i].split("@")[-1], seg[i], remote


# Programs that do not change the files named in their arguments. Anything else
# (sed -i, cp, mv, python gen.py job.sh, an editor, ...) may have rewritten a file
# the hook is about to read, so a file it names counts as "content unknown".
LEAVES_FILES_ALONE = {
    "cat", "ls", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "wc", "stat", "test", "[",
    "echo", "printf", "cd", "pwd", "mkdir", "chmod", "chown", "chgrp", "file", "diff", "cmp",
    "md5sum", "sha256sum", "du", "df", "find", "tee", "module", "source", ".", "export", "true",
    "false", "sleep", "date", "hostname", "which", "type", "klist", "screen", "ssh", "nvidia-smi",
}
COPIES_TO_LAST = {"cp", "mv", "install", "ln", "scp", "rsync"}


def modified_paths(prog, args):
    """Paths in args whose content this command may change."""
    if prog in LEAVES_FILES_ALONE or prog in JOB_CONTROL or prog in QUERY or prog in SHELLS:
        return []
    paths = [a for a in args if a and not a.startswith("-") and a not in (">", ">>", ">|", "<", "&>", "&>>")]
    return paths[-1:] if prog in COPIES_TO_LAST else paths


def redirect_targets(seg):
    """Files a segment writes with > / >> / tee."""
    out = [seg[k + 1] for k, t in enumerate(seg[:-1]) if t in (">", ">>", ">|", "&>", "&>>")]
    if seg and os.path.basename(seg[0]) == "tee":
        out += [a for a in seg[1:] if not a.startswith("-")]
    return out


def has_flag(args, letters, longs=()):
    for a in args:
        if a in longs:
            return True
        if a.startswith("-") and not a.startswith("--") and any(c in a[1:] for c in letters):
            return True
    return False


class Verdict:
    def __init__(self):
        self.deny = None
        self.ask = None
        self.submits = []  # (program, args, ssh target or None, working dir or None)
        # files written earlier in the same command: (ssh target, shell path) ->
        # here-document content, or None when written some other way
        self.written = {}

    def set_deny(self, msg):
        self.deny = self.deny or msg

    def set_ask(self, msg):
        self.ask = self.ask or msg


def check(cmd, login, verdict, depth=0, host=None, cwd=None):
    """login: True when the command would run on a login node. cwd: the directory a
    preceding `cd` moved to (carried into nested `bash -c` strings)."""
    if depth > 3:
        return
    if ANACONDA.search(cmd):
        verdict.set_deny(
            "Anaconda, Miniconda and the 'defaults' channel are not allowed on the Bonn "
            "clusters. Use `module load Miniforge3` with conda-forge."
        )
    if LOOP.search(cmd):
        sleeps = [int(s) for s in re.findall(r"\bsleep\s+(\d+)", cmd)]
        if not sleeps or min(sleeps) < MIN_POLL:
            verdict.set_deny(
                "Scheduler polling loop. Use `sbatch --wait` / `condor_wait <log>` or read the "
                "job's log file; if you must loop, sleep at least %d s." % MIN_POLL
            )

    if SUBMIT_LOOP.search(cmd):
        verdict.set_deny(
            "Job submission inside a loop. Submit one job array (`--array=0-N%cap`) or one "
            "HTCondor cluster (`queue N` with `max_materialize`) instead."
        )

    stripped, bodies = split_heredocs(cmd)
    for raw in segments(tokenize(stripped)):
        seg = strip_prefix(raw)
        if not seg:
            continue
        prog, args = os.path.basename(seg[0]), seg[1:]

        plain, inline = without_heredocs(seg, bodies)
        for touched in modified_paths(prog, plain[1:]):
            verdict.written[(host, shell_path(touched, cwd))] = None
        for target in redirect_targets(seg):
            verdict.written[(host, shell_path(target, cwd))] = inline

        if prog == "cd" and args:
            cwd = args[0] if args[0].startswith(("/", "~", "$")) or not cwd else cwd.rstrip("/") + "/" + args[0]
            continue
        if prog == "ssh":
            name, target, remote = ssh_remote(seg, bodies)
            if name and remote and re.search(SSH_HOSTS, name):
                check(remote, True, verdict, depth + 1, target)
            continue
        cflag = next(
            (i for i, a in enumerate(args) if re.match(r"^-[A-Za-z]*c[A-Za-z]*$", a)), None
        )  # -c, and combined forms such as -lc / -ic
        if prog in SHELLS and cflag is not None:
            idx = cflag
            if idx + 1 < len(args):
                check(args[idx + 1], login, verdict, depth + 1, host, cwd)
            continue
        if prog in SHELLS and inline is not None and not [a for a in plain[1:] if not a.startswith("-")]:
            check(inline, login, verdict, depth + 1, host, cwd)  # `bash -s <<EOF`
            continue
        if prog == "xargs":
            rest = [a for a in args if not a.startswith("-")]
            if rest:
                check(" ".join(shlex.quote(a) for a in rest), login, verdict, depth + 1, host, cwd)
            continue

        if prog == "watch" and any(os.path.basename(a) in QUERY for a in args):
            verdict.set_deny(
                "`watch` on a scheduler command hammers the scheduler. Check once, or use "
                "`sbatch --wait` / `condor_wait <log>`."
            )

        walker = (
            prog in ("find", "du", "tree", "ncdu", "fd", "rg", "ag", "locate")
            or (prog in ("grep", "egrep", "fgrep") and has_flag(args, "rR", ("--recursive",)))
            or (prog == "ls" and has_flag(args, "R", ("--recursive",)))
            or (prog in ("chmod", "chown", "chgrp") and has_flag(args, "R", ("--recursive",)))
            or (prog == "lfs" and args[:1] == ["find"])
        )
        if walker:
            hit = next((a for a in args if SHARED_ROOT.match(a)), None)
            if hit:
                verdict.set_deny(
                    "Recursive walk of the shared path %s. Search only inside your own project or "
                    "workspace directory, with -maxdepth; for usage use `lfs quota` (Lustre) or "
                    "`getfattr -n ceph.dir.rentries` (CephFS)." % hit
                )

        if prog == "rm" and has_flag(args, "rR", ("--recursive",)):
            targets = [a for a in args if not a.startswith("-")]
            shared = next((a for a in targets if SHARED_ROOT.match(a)), None)
            if shared:
                verdict.set_deny("Refusing recursive delete of the shared path %s." % shared)
            elif any(SHARED_PREFIX.match(a) for a in targets):
                verdict.set_ask(
                    "Recursive delete on a parallel filesystem. Confirm the path with the user. On "
                    "Lustre use `find -P <dir> -type f -print0 -o -type l -print0 | xargs -0 "
                    "munlink` and then `find -P <dir> -type d -empty -delete` instead of rm -r."
                )

        if prog == "scontrol" and args[:1] and args[0] in SCONTROL_CHANGE:
            prog = "scontrol " + args[0]
            if args[0] == "update" and re.search(r"(?i)\b(gres|tres\w*|numnodes|numcpus)=", " ".join(args)):
                verdict.set_ask(
                    "`scontrol update` changes a job's resource request. Show the change and the "
                    "resources you already hold to the user and get their approval."
                )
            elif CONFIRM_JOBS:
                verdict.set_ask("`%s` changes jobs on a shared cluster. Confirm with the user." % prog)
            continue

        if prog in JOB_CONTROL:
            if prog == "condor_qedit" and re.search(r"(?i)request_?gpus", " ".join(args)):
                verdict.set_ask(
                    "`condor_qedit` of a GPU request bypasses the submission check. Tell the user "
                    "how many GPUs you hold and the new request, and get their approval."
                )
            if prog in SUBMIT:
                verdict.submits.append((prog, args, host, cwd))
            if prog in ("scancel", "condor_rm") and agent_jobs_only(prog, args, host):
                continue  # removing the agent's own tagged jobs needs no approval
            if CONFIRM_JOBS and prog not in AUTO_SUBMIT:
                verdict.set_ask(
                    "`%s` changes jobs on a shared cluster. Show the job script and the resource "
                    "request to the user first." % prog
                )
            continue  # whatever follows runs inside the allocation

        if not login:
            continue

        if prog in COMPUTE or (prog == "accelerate" and args[:1] == ["launch"]):
            verdict.set_deny(
                "`%s` would run on a login node. Put it in a batch job, or use the debug queue / "
                "an interactive job (`srun --pty`, `condor_submit -interactive`)." % prog
            )
        elif re.match(r"^(python[\d.]*|pytest|ipython|Rscript|julia|matlab)$", prog):
            light = (
                not args
                or args[0] in ("-V", "--version", "-h", "--help", "-c")
                or args[:2] in (["-m", "venv"], ["-m", "pip"], ["-m", "json.tool"])
            )
            if not light:
                verdict.set_ask(
                    "`%s` would run on a login node. Only continue if this takes seconds on one "
                    "core; otherwise submit it as a job." % " ".join(seg[:3])
                )
        elif prog == "make":
            jobs = [int(m.group(1)) for a in args for m in [re.match(r"^-j(\d+)$", a)] if m]
            if any(a == "-j" for a in args) or any(j > 8 for j in jobs):
                verdict.set_ask("Large parallel build on a login node. Use -j8 or less, or build in a job.")


def rate_limit(count, verdict):
    """Ask once more than MAX_SUBMITS_PER_HOUR submit commands were seen in the last hour."""
    now = time.time()
    path = os.path.join(STATE_DIR, "submissions.log")
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        recent = []
        if os.path.exists(path):
            with open(path) as fh:
                recent = [float(x) for x in fh.read().split() if float(x) > now - 3600]
        recent += [now] * count
        with open(path, "w") as fh:
            fh.write("\n".join("%.0f" % t for t in recent))
    except (OSError, ValueError):
        return
    if len(recent) > MAX_SUBMITS_PER_HOUR:
        verdict.set_ask(
            "%d job submissions in the last hour (limit %d without asking). Confirm with the user "
            "that more are wanted; similar tasks belong in one array job."
            % (len(recent), MAX_SUBMITS_PER_HOUR)
        )


def job_file(args):
    return next((a for a in args if not a.startswith("-") and JOB_FILE.search(a)), None)


def shell_path(path, cwd):
    if cwd and not path.startswith(("/", "~", "$")):
        path = cwd.rstrip("/") + "/" + path
    for home in ("~/", "$HOME/", "${HOME}/"):
        if path.startswith(home):
            return '"$HOME"/' + shlex.quote(path[len(home):])
    return shlex.quote(path)


def probe(host, script):
    """Run one read-only query where the job would be submitted. None on failure."""
    if not PROBE:
        return None
    argv = ["bash", "-c", script]
    if host:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, script]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=25).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out if "@@ok" in out else None


def array_tasks(spec):
    """'0-199%10' -> (200, True); unknown syntax -> (None, capped)."""
    spec, _, cap = spec.strip().partition("%")
    total = 0
    for part in spec.split(","):
        m = re.match(r"^(\d+)(?:-(\d+)(?::(\d+))?)?$", part)
        if not m:
            return None, bool(cap)
        first, last, step = int(m.group(1)), int(m.group(2) or m.group(1)), int(m.group(3) or 1)
        total += max(0, (last - first) // max(step, 1) + 1)
    return total, bool(cap)


def agent_jobs_only(prog, args, host):
    """True if the command names explicit job IDs and every matching job carries the agent tag
    (Slurm: --comment=AGENT_TAG; HTCondor: +ClaudeAgent = True). Filters (-u, -all, -constraint,
    names, partitions) never qualify."""
    ids, schedd, rest = [], None, list(args)
    while rest:
        a = rest.pop(0)
        if prog == "scancel" and a in ("-Q", "--quiet"):
            continue
        if prog == "condor_rm" and a == "-name" and rest:
            schedd = rest.pop(0)
            continue
        idpat = r"^\d+(_\d+|_\[[\d,\-]+\])?$" if prog == "scancel" else r"^\d+(\.\d+)?$"
        if not re.match(idpat, a):
            return False
        ids.append(a)
    if not ids:
        return False
    if prog == "scancel":
        out = probe(host, "command -v squeue >/dev/null || exit 1; squeue --me -h -j %s -o '%%i|%%k' 2>/dev/null; echo @@ok" % ",".join(ids))
        if out is None or "@@ok" not in out:
            return False
        rows = [l.split("|", 1) for l in out.partition("@@ok")[0].splitlines() if "|" in l]
        return all(AGENT_TAG in c for _, c in rows)
    out = probe(host, "command -v condor_q >/dev/null || exit 1; condor_q %s%s -af ClusterId ProcId ClaudeAgent 2>/dev/null; echo @@ok" % (
        "-name %s " % shlex.quote(schedd) if schedd else "", " ".join(ids)))
    if out is None or "@@ok" not in out:
        return False
    rows = [l.split() for l in out.partition("@@ok")[0].splitlines() if l.strip()]
    return all(len(r) == 3 and r[2].lower() == "true" for r in rows)


def sbatch_option(texts, longs, shorts=()):
    """Last value of an sbatch option; texts are searched in order, first hit wins."""
    pat = r"(?:--(?:%s)(?:=|\s+)|(?:^|\s)-(?:%s)\s*)(\S+)" % (
        "|".join(longs), "|".join(shorts) or "(?!)"
    )
    for text in texts:
        hits = re.findall(pat, text)
        if hits:
            return hits[-1]
    return None


def gpu_count(value):
    """'4', 'a100:4', 'gpu:a100:4', 'gpu:2(S:0)' -> int; anything else -> None."""
    m = re.match(r"^(?:gpu:)?(?:[A-Za-z][\w.-]*:)?(\d+)\b", value or "")
    return int(m.group(1)) if m else None


WRITTEN_HERE = (
    "The job file %s is written or changed by this same command, so the hook cannot know its "
    "resource request. Write or edit the file in one command and submit it in the next (or "
    "write it with a here-document, which the hook reads)."
)


def same_command_file(path, host, cwd, written):
    """False: not written by this command. None: written, content unknown. Else the content."""
    if not path:
        return False
    return (written or {}).get((host, shell_path(path, cwd)), False)


def slurm_limits(args, host, cwd, verdict, written=None):
    path = job_file(args)
    inline = same_command_file(path, host, cwd, written)
    if inline is None:
        verdict.set_ask(WRITTEN_HERE % path)
    script = (
        "command -v squeue >/dev/null && squeue --me -h -r -t PD,R,S,CF,RQ,RS "
        "-O Partition:60,tres-alloc:300 && echo @@ok"
    )
    if path and inline is False:
        script += "; test -r {0} && echo @@readable; grep -E '^#SBATCH' {0} 2>/dev/null".format(shell_path(path, cwd))
    out = probe(host, script)
    queue_text, _, directives = (out or "").partition("@@ok")
    if inline:
        directives = "@@readable\n" + "\n".join(l for l in inline.splitlines() if l.startswith("#SBATCH"))
    readable = "@@readable" in directives
    directives = "\n".join(l.strip()[len("#SBATCH"):] for l in directives.splitlines() if l.strip().startswith("#SBATCH"))
    texts = (" ".join(args), directives)  # command line overrides the script

    spec = sbatch_option(texts, ["array"], ["a"])
    tasks, capped = array_tasks(spec) if spec else (1, True)
    if tasks is None or (tasks > MAX_UNCAPPED and not capped):
        verdict.set_ask(
            "Job array of %s tasks without a concurrency cap. Add `%%N` to --array (for example "
            "--array=%s%%10) or confirm with the user." % (tasks or "unknown size", spec)
        )
    cap = re.search(r"%(\d+)", spec or "")
    concurrent = min(tasks or 10 ** 6, int(cap.group(1))) if cap else (tasks or 1)

    partition = sbatch_option(texts, ["partition"], ["p"]) or ""
    if re.search(SLURM_LONG_PARTITIONS, partition):
        name = sbatch_option(texts, ["job-name"], ["J"]) or (os.path.basename(path) if path else "wrap")
        name = name.strip("'\"")
        hist = probe(
            host,
            "sacct -X -n -P -S now-%ddays -E now -s CD -o JobName,Partition 2>/dev/null; echo @@ok" % TEST_RUN_DAYS,
        )
        tested = hist is not None and any(
            j == name and re.search(r"_(devel|short)\b", p)
            for j, _, p in (l.partition("|") for l in hist.partition("@@ok")[0].splitlines())
        )
        if not tested:
            verdict.set_ask(
                "Partition %s runs for a day or more, and no job named `%s` completed on a devel or "
                "short partition in the last %d days. Run a reduced version there first (start, GPU "
                "use, output, checkpoint and resume), with the same --job-name, or get the user's "
                "approval." % (partition, name, TEST_RUN_DAYS)
            )

    nodes = gpu_count((sbatch_option(texts, ["nodes"], ["N"]) or "1").split("-")[0]) or 1
    ntasks = gpu_count(sbatch_option(texts, ["ntasks"], ["n"]) or "1") or 1
    per_job = sbatch_option(texts, ["gpus"], ["G"])
    per_node = sbatch_option(texts, ["gpus-per-node"])
    per_task = sbatch_option(texts, ["gpus-per-task"])
    gres = sbatch_option(texts, ["gres"])
    gres_gpu = re.search(r"(?:^|,)(gpu(?::[^,]*)?)", gres or "")
    if per_job:
        new_gpus = gpu_count(per_job)
    elif per_node:
        new_gpus = (gpu_count(per_node) or 0) * nodes if gpu_count(per_node) is not None else None
    elif per_task:
        new_gpus = (gpu_count(per_task) or 0) * ntasks if gpu_count(per_task) is not None else None
    elif gres_gpu:
        n = gpu_count(gres_gpu.group(1)) if ":" in gres_gpu.group(1) else 1
        new_gpus = n * nodes if n is not None else None
    elif path and out is not None and not readable:
        new_gpus = None  # job script not readable from here
    else:
        new_gpus = 0

    if out is None:
        verdict.set_ask("Could not check how many jobs and GPUs you already have. Confirm this submission with the user.")
        return
    lines = [l for l in queue_text.splitlines() if l.strip()]
    queued = len(lines)
    held_gpus = sum(int(n) for l in lines for n in re.findall(r"gres/gpu=(\d+)", l))
    if queued + (tasks or 1) > MAX_QUEUED:
        verdict.set_ask(
            "You have %d jobs queued and this adds %s (limit %d without asking). Confirm with the user."
            % (queued, tasks or "an unknown number", MAX_QUEUED)
        )
    if new_gpus is None:
        verdict.set_ask(
            "Could not work out how many GPUs this job requests; you currently hold %d. Confirm "
            "this submission with the user." % held_gpus
        )
    elif new_gpus > 0 and held_gpus + new_gpus * concurrent > SLURM_GPU_BUDGET:
        verdict.set_ask(
            "GPU budget: you hold %d GPU(s) in running or pending jobs and this requests %d more "
            "(budget %d without asking). Tell the user both numbers and get their approval."
            % (held_gpus, new_gpus * concurrent, SLURM_GPU_BUDGET)
        )


def condor_test_gate(text, args, host, gpus, verdict):
    """Jobs above CONDOR_LONG_HOURS need a short test run of the same JobBatchName that
    completed with exit code 0. Short runs are noted locally at submission, so the history
    scan on the submit host only reaches back to that moment."""
    hours = re.findall(r"(?im)^\s*\+?MaxRuntimeHours\s*=\s*(\d+)", text)
    if not hours:
        return
    hours = int(hours[-1])
    names = re.findall(r'(?im)^\s*(?:\+?JobBatchName|batch_name)\s*=\s*"?([^"\n]+?)"?\s*$', text)
    name = names[-1] if names else None
    for i, a in enumerate(args[:-1]):
        if a == "-batch-name":
            name = args[i + 1]
    path = os.path.join(STATE_DIR, "condor_tests.log")
    now = int(time.time())
    if hours <= CONDOR_TEST_HOURS:
        if name:
            try:
                os.makedirs(STATE_DIR, exist_ok=True)
                with open(path, "a") as fh:
                    fh.write("%d\t%s\n" % (now, name))
            except OSError:
                pass
        return
    if hours <= CONDOR_LONG_HOURS:
        return
    if not name or not re.match(r"^[\w.\-]+$", name):
        verdict.set_ask(
            "This job may run for %d hours. Jobs above %d hours need a JobBatchName (letters, "
            "digits, . _ -) shared with a short test run that completed first; set one, or get "
            "the user's approval." % (hours, CONDOR_LONG_HOURS)
        )
        return
    since = None
    try:
        with open(path) as fh:
            for line in fh:
                ts, _, n = line.rstrip("\n").partition("\t")
                if n == name and ts.isdigit() and int(ts) > now - TEST_RUN_DAYS * 86400:
                    since = int(ts) if since is None else min(since, int(ts))
    except OSError:
        pass
    done = None
    if since is not None:
        gpu_clause = " && RequestGPUs >= 1" if gpus else ""
        done = probe(
            host,
            "condor_history \"$USER\" -since 'EnteredCurrentStatus < %d' -constraint "
            "'JobBatchName == \"%s\" && JobStatus == 4 && ExitCode == 0 && MaxRuntimeHours <= %d%s' "
            "-limit 1 -af ClusterId 2>/dev/null; echo @@ok" % (since - 60, name, CONDOR_TEST_HOURS, gpu_clause),
        )
    if not (done and re.search(r"^\d+", done.partition("@@ok")[0].strip())):
        verdict.set_ask(
            "This job may run for %d hours, and no test run named `%s` (MaxRuntimeHours <= %d, "
            "same GPU type, exit code 0) has completed in the last %d days. Run a reduced version "
            "first (start, GPU use, output, checkpoint and resume) under the same JobBatchName, "
            "or get the user's approval." % (hours, name, CONDOR_TEST_HOURS, TEST_RUN_DAYS)
        )


CONDOR_KEYS = r"^[[:space:]]*(request_gpus|queue|max_materialize|[+]?MaxRuntimeHours|[+]?JobBatchName|batch_name)"


def condor_limits(args, host, cwd, verdict, written=None):
    path = job_file(args)
    inline = same_command_file(path, host, cwd, written)
    if inline is None:
        verdict.set_ask(WRITTEN_HERE % path)
    script = 'command -v condor_q >/dev/null || exit 1; condor_q -global "$USER" -af JobStatus RequestGPUs 2>/dev/null; echo @@ok'
    if path and inline is False:
        script += "; grep -iE '%s' %s 2>/dev/null" % (CONDOR_KEYS, shell_path(path, cwd))
    out = probe(host, script)
    if out is not None and inline:
        keys = re.compile(CONDOR_KEYS.replace("[[:space:]]", r"\s"), re.I)
        out = out.partition("@@ok")[0] + "@@ok\n" + "\n".join(l for l in inline.splitlines() if keys.match(l))
    if out is None:
        verdict.set_ask(
            "Could not check how many GPUs and jobs you currently hold. Run "
            "`condor_q -global $USER -af JobStatus RequestGPUs`, report it, and confirm this "
            "submission with the user."
        )
        return
    queue_text, _, file_text = out.partition("@@ok")
    active = [g for st, g in re.findall(r"(?m)^\s*(\d)\s+(\S+)\s*$", queue_text) if st in "125"]
    held_gpus = sum(int(g) for g in active if g.isdigit())

    text = file_text + "\n" + "\n".join(args)  # -append values come last and win
    gpus = re.findall(r"(?im)^\s*request_gpus\s*=\s*(\S*)", text)
    if gpus and gpus[-1].isdigit():
        new_gpus = int(gpus[-1])
    elif gpus:
        new_gpus = None  # a macro or variable, e.g. request_gpus = $(N)
    elif path and not file_text.strip():
        new_gpus = None  # submit file not readable from here
    else:
        new_gpus = 0

    condor_test_gate(text, args, host, new_gpus or 0, verdict)

    cap = re.search(r"(?im)^\s*max_materialize\s*=\s*(\d+)", text)
    queue = re.search(r"(?im)^\s*queue\b[ \t]*(\d+)?(.*)$", text)
    if queue and queue.group(1):
        procs = int(queue.group(1))
    elif queue and re.search(r"\b(from|in|matching)\b", queue.group(2)):
        procs = None
    else:
        procs = 1
    if (procs is None or procs > MAX_UNCAPPED) and not cap:
        verdict.set_ask(
            "This submit file queues %s jobs without `max_materialize`. Add a cap or confirm with the user."
            % (procs or "a list of")
        )
    concurrent = min(procs or 10 ** 6, int(cap.group(1))) if cap else (procs or 1)
    if len(active) + (procs or concurrent) > MAX_QUEUED:
        verdict.set_ask(
            "You have %d jobs queued and this adds %s (limit %d without asking). Confirm with the user."
            % (len(active), procs or "a list", MAX_QUEUED)
        )
    if new_gpus is None:
        verdict.set_ask(
            "Could not read from the submit file how many GPUs it requests; you currently hold "
            "%d. Confirm this submission with the user." % held_gpus
        )
    elif new_gpus > 0 and held_gpus + new_gpus * concurrent > GPU_BUDGET:
        verdict.set_ask(
            "GPU budget: you hold %d GPU(s) in running, idle or held jobs and this requests %d more "
            "(budget %d without asking). Tell the user both numbers and get their approval."
            % (held_gpus, new_gpus * concurrent, GPU_BUDGET)
        )


SCP_ARG_OPTS = set("cFiJloPSX")


def local_path(path, base):
    return os.path.normpath(os.path.join(base, os.path.expanduser(path)))


def record_copy(prog, args, local_dir, local_written, verdict):
    """`scp|rsync SRC... host:DEST` to a cluster: note DEST as written by this command,
    with the content of SRC (from a here-document in this command, or read from local
    disk), or None when SRC cannot be read here."""
    paths, i = [], 0
    while i < len(args):
        a = args[i]
        if a.startswith("-") and len(a) > 1:
            if prog == "scp" and len(a) == 2 and a[1] in SCP_ARG_OPTS:
                i += 1
            elif prog == "rsync" and a in ("-e", "--rsh"):
                i += 1
        else:
            paths.append(a)
        i += 1
    if len(paths) < 2:
        return
    m = re.match(r"^(?:[^@/:]+@)?([^/:]+):(.*)$", paths[-1])
    if not m or not re.search(SSH_HOSTS, m.group(1)):
        return
    host, dest = paths[-1].split(":", 1)[0], m.group(2)
    if dest and not dest.startswith(("/", "~")):
        dest = "~/" + dest  # scp destinations are relative to the remote home
    for src in paths[:-1]:
        if re.match(r"^[^/]+:", src):
            continue  # remote source
        lp = local_path(src, local_dir)
        content = local_written.get(lp, False)
        if content is False:
            try:
                with open(lp, errors="replace") as fh:
                    content = fh.read(1 << 20)
            except OSError:
                content = None
        name = os.path.basename(lp)
        if not dest or dest.endswith("/") or dest in ("~", "."):
            targets = [(dest or "~/").rstrip("/") + "/" + name]
        else:
            targets = [dest, dest.rstrip("/") + "/" + name]  # DEST may be a directory
        for t in targets:
            verdict.written[(host, shell_path(t, None))] = content


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        sys.exit(0)
    if payload.get("tool_name") != "Bash":
        sys.exit(0)
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd.strip():
        sys.exit(0)

    in_job = bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("_CONDOR_SCRATCH_DIR"))
    on_cluster = in_job or bool(re.search(LOGIN_REGEX, socket.gethostname()))

    verdict = Verdict()
    if on_cluster:
        check(cmd, not in_job, verdict)
    else:
        # Off-cluster: ssh calls to a cluster, plus files copied there with
        # scp/rsync in the same command (read locally, since the copy has not
        # happened yet when the hook runs).
        local_dir = payload.get("cwd") or os.getcwd()
        local_written = {}
        stripped, bodies = split_heredocs(cmd)
        for raw in segments(tokenize(stripped)):
            seg = strip_prefix(raw)
            if not seg:
                continue
            plain, inline = without_heredocs(seg, bodies)
            prog = os.path.basename(seg[0])
            if prog not in ("scp", "rsync"):
                for touched in modified_paths(prog, plain[1:]):
                    local_written[local_path(touched, local_dir)] = None
            for target in redirect_targets(seg):
                local_written[local_path(target, local_dir)] = inline
            if prog == "ssh":
                name, target, remote = ssh_remote(seg, bodies)
                if name and remote and re.search(SSH_HOSTS, name):
                    check(remote, True, verdict, 1, target)
            elif prog in ("scp", "rsync"):
                record_copy(prog, plain[1:], local_dir, local_written, verdict)

    if verdict.submits and not verdict.deny:
        limits = Verdict()
        rate_limit(len(verdict.submits), limits)
        for prog, args, host, cwd in verdict.submits:
            if prog == "sbatch":
                slurm_limits(args, host, cwd, limits, verdict.written)
            elif prog == "condor_submit":
                condor_limits(args, host, cwd, limits, verdict.written)
        if limits.ask:
            verdict.ask = limits.ask  # more specific than the generic confirmation

    if verdict.deny:
        respond("deny", verdict.deny)
    if verdict.ask:
        respond("ask", verdict.ask)
    sys.exit(0)


if __name__ == "__main__":
    main()
