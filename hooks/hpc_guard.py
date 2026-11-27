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

Submissions fail closed: one passes without a prompt only when the whole
command was parsed and every submit in it was read. Several submits in one
command are added up. A submit the hook cannot read (inside quotes, backticks,
a function, behind an unknown command), a request it cannot count (shell
variables, SBATCH_* variables, unknown or abbreviated options, macros,
`include`, a job file on a pipe), a command that may have changed the job file
first (git pull, tar, a script), or a script that itself submits, asks.

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
BUDDY_ROOT = re.compile(r"^(\$BUDDY|\$\{BUDDY\}|/cephfs/user/[^/]+)/?$")

COMPUTE = {
    "torchrun", "mpirun", "mpiexec", "orterun", "vllm", "ollama", "tensorboard",
    "jupyter", "jupyter-lab", "jupyter-notebook", "deepspeed", "horovodrun",
    "text-generation-launcher", "papermill", "dask-scheduler", "dask-worker",
    "streamlit", "gradio", "uvicorn", "gunicorn", "condor_watch_q",
}
# launchers that are only a problem with certain sub-commands
COMPUTE_SUB = {"accelerate": {"launch"}, "ray": {"start", "up", "submit"}, "lightning": {"run"},
               "dask": {"scheduler", "worker"}}
# programs that run another command given as their arguments
RUNNERS = {"conda", "mamba", "micromamba", "uv", "poetry", "pipenv", "pixi", "hatch", "pdm"}
CARRIERS = {"tmux", "screen", "sg", "sudo", "su", "script", "numactl", "taskset", "chrt", "xvfb-run", "watch"}
KEYWORDS = {"then", "else", "elif", "do", "if", "while", "until", "{", "!"}
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
_QW = "|".join(sorted(QUERY | {"scontrol"}))
LOOP = re.compile(r"\b(while|until)\b[\s\S]*\b(%s)\b[\s\S]*\bdone\b" % _QW)   # query in condition or body
FOR_LOOP = re.compile(r"\bfor\b[\s\S]*\bdo\b[\s\S]*\b(%s)\b[\s\S]*\bdone\b" % _QW)
SUBMIT_LOOP = re.compile(
    r"(\b(while|until|for)\b[\s\S]*\bdo\b|\bxargs\b|\bparallel\b|-exec(dir)?\b)[\s\S]*\b(%s)\b"
    % "|".join(sorted(SUBMIT))
)
SUBMIT_WORD = re.compile(r"(?<![\w./-])(%s)(?![\w.-])" % "|".join(sorted(SUBMIT)))
ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)
VAR = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")

UNREAD_SUBMIT = (
    "A submit command (sbatch / condor_submit) appears where the hook cannot read it: inside "
    "quotes, backticks, a variable, a function, or behind a command it does not know. Run the "
    "submit as a plain command of its own, e.g. `ssh <cluster> 'cd <dir> && sbatch job.sh'`."
)
BLIND_WRITER = (
    "`%s` runs before the submit in the same command and may create or change the job file, so "
    "the hook cannot know what would be submitted. Run it in a command of its own, then submit."
)


def count_submits(text):
    return len(SUBMIT_WORD.findall(text))


def expand(tok, env):
    """Replace $NAME / ${NAME} by values assigned earlier in the same command."""
    def rep(m):
        v = env.get(m.group(1) or m.group(2))
        return v if isinstance(v, str) else m.group(0)
    return VAR.sub(rep, tok)


def prefix_env(seg):
    """NAME=value words in front of the program (or making up the whole segment)."""
    out = {}
    for tok in seg:
        if tok in KEYWORDS or os.path.basename(tok) in WRAPPERS:
            continue
        m = ASSIGN.match(tok)
        if not m:
            break
        out[m.group(1)] = m.group(2)
    return out


def remember(env, assigns):
    for k, v in assigns.items():
        env[k] = None if re.search(r"[$`]", v) else v


def loop_rules(text, verdict):
    """Polling loops and submissions in loops, on the text of one command."""
    sleeps = [int(x) for x in re.findall(r"\bsleep\s+(\d+)", text)]
    tight = not sleeps or min(sleeps) < MIN_POLL
    if (LOOP.search(text) and tight) or (FOR_LOOP.search(text) and sleeps and tight):
        verdict.set_deny(
            "Scheduler polling loop. Use `sbatch --wait` / `condor_wait <log>` or read the "
            "job's log file; if you must loop, sleep at least %d s." % MIN_POLL
        )
    if SUBMIT_LOOP.search(text):
        verdict.set_deny(
            "Job submission inside a loop. Submit one job array (`--array=0-N%cap`) or one "
            "HTCondor cluster (`queue N` with `max_materialize`) instead."
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
        if ASSIGN.match(tok) or tok in KEYWORDS:
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
# Programs that may precede a submit in the same command without making the job
# file's content unknown. Writers that name their target (sed, cp, ...) are
# tracked per file by modified_paths; anything not listed here could rewrite
# files it does not name (git pull, tar, a script), so the submit asks.
SAFE_BEFORE_SUBMIT = LEAVES_FILES_ALONE | {
    "sed", "perl", "awk", "gawk", "cp", "mv", "ln", "install",
    "set", "unset", "umask", "ulimit", ":", "exit", "return", "rm", "rmdir", "touch", "wait",
    "fi", "done", "}", "esac", "for", "case", "function", "local", "readonly", "declare",
    "alias", "trap", "shopt", "ml", "id", "whoami", "env", "printenv", "basename", "dirname",
    "realpath", "readlink", "sort", "uniq", "cut", "tr", "scontrol", "ws_list", "ws_find",
    "lfs", "getfattr", "bquota", "condor_wait", "condor_tail", "squeue", "sacct", "sinfo",
}


def blind_writer(prog, args):
    """Name of the program if it could change files it does not name, else None."""
    if prog in ("conda", "mamba", "micromamba") and args[:1] and args[0] in ("activate", "deactivate"):
        return None
    if prog == "git" and next((a for a in args if not a.startswith("-")), "") in (
        "status", "log", "diff", "show", "branch", "rev-parse", "remote", "describe", "fetch", "ls-files"
    ):
        return None
    if prog in COPIES_TO_LAST:
        paths = [a for a in args if not a.startswith("-")]
        dest = paths[-1] if paths else ""
        many = len(paths) > 2 or any(re.search(r"[*?\[]", a) for a in paths[:-1])
        if has_flag(args, "rRa", ("--recursive", "--archive")) or dest in (".", "..") or dest.endswith("/") or many:
            return prog
        return None
    if prog in SAFE_BEFORE_SUBMIT or prog in QUERY or prog in JOB_CONTROL:
        return None
    return prog


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
        self.submits = []  # dicts: prog, args, host, cwd, stdin, piped, why
        self.blind = None  # a program seen so far that may change files it does not name
        self.scripts = []  # (host, path) of scripts the command runs on the cluster
        # files written earlier in the same command: (ssh target, shell path) ->
        # here-document content, or None when written some other way
        self.written = {}

    def set_deny(self, msg):
        self.deny = self.deny or msg

    def set_ask(self, msg):
        self.ask = self.ask or msg


HARMLESS_MENTION = {"which", "type", "man", "whatis", "apropos", "command", "module", "ml", "grep", "egrep", "rpm", "alias"}
NOT_A_SUBMISSION = {
    "sbatch": {"--test-only", "--help", "-h", "--usage", "--version", "-V"},
    "condor_submit": {"-dry-run", "-help", "-version"},
    "condor_submit_dag": {"-help", "-version", "-no_submit"},
}


def segments_sep(tokens):
    """(separator before the segment, segment)."""
    seg, sep, last = [], "", ""
    for tok in tokens:
        if tok and set(tok) <= set(";&|()\n"):
            if seg:
                yield sep, seg
                seg = []
            last = tok
        else:
            if not seg:
                sep = last
            seg.append(tok)
    if seg:
        yield sep, seg


def nested_commands(prog, args):
    """Command strings that `prog` would run, for programs that run their arguments."""
    if prog in RUNNERS:
        if "run" not in args:
            return []
        rest, out, skip = args[args.index("run") + 1:], [], False
        for k, a in enumerate(rest):
            if skip:
                skip = False
            elif a in ("-n", "-p", "--name", "--prefix", "--cwd", "--with", "--python", "--project", "--directory"):
                skip = True
            elif not a.startswith("-"):
                out = rest[k:]
                break
        return [" ".join(shlex.quote(a) for a in out)] if out else []
    if prog in CARRIERS:
        found = [a for a in args if " " in a.strip()]
        tail = list(args)
        while tail and (tail[0].startswith("-") or re.match(r"^\d+(\.\d+)?$", tail[0])):
            tail.pop(0)
        if prog in ("tmux", "sg", "su") and tail:
            tail = tail[1:]  # sub-command, group or user name
        if tail and not found:
            found.append(" ".join(shlex.quote(a) for a in tail))
        return found
    return []


def check(cmd, login, verdict, depth=0, host=None, cwd=None, env=None):
    """login: True when the command would run on a login node. cwd: the directory a
    preceding `cd` moved to (carried into nested `bash -c` strings). env: variables
    assigned earlier in the same command. Returns nothing; fills verdict."""
    if depth > 4:
        verdict.set_ask(UNREAD_SUBMIT) if count_submits(cmd) else None
        return
    env = {} if env is None else env
    if ANACONDA.search(cmd):
        verdict.set_deny(
            "Anaconda, Miniconda and the 'defaults' channel are not allowed on the Bonn "
            "clusters. Use `module load Miniforge3` with conda-forge."
        )
    loop_rules(cmd, verdict)

    stripped, bodies = split_heredocs(cmd)
    total, accounted, queries = count_submits(stripped), 0, 0
    for sep, raw in segments_sep(tokenize(stripped)):
        raw = [expand(t, env) for t in raw]
        assigns = prefix_env(raw)
        seg = strip_prefix(raw)
        if not seg:
            remember(env, assigns)
            continue
        prog, args = os.path.basename(seg[0]), seg[1:]
        plain, inline = without_heredocs(seg, bodies)
        pargs = plain[1:]  # arguments without here-document tokens

        if prog == "export":
            remember(env, {m.group(1): m.group(2) for m in map(ASSIGN.match, args) if m})
        for touched in modified_paths(prog, pargs):
            verdict.written[(host, shell_path(touched, cwd))] = None
        for target in redirect_targets(seg):
            verdict.written[(host, shell_path(target, cwd))] = inline

        if prog == "cd" and args:
            cwd = args[0] if args[0].startswith(("/", "~", "$")) or not cwd else cwd.rstrip("/") + "/" + args[0]
            continue
        if prog == "ssh":
            name, target, remote = ssh_remote(seg, bodies)
            if name and remote and re.search(SSH_HOSTS, name):
                check(remote, True, verdict, depth + 1, target, env=dict(env))
                accounted += count_submits(" ".join(pargs))
            continue
        cflag = next(
            (i for i, a in enumerate(args) if re.match(r"^-[A-Za-z]*c[A-Za-z]*$", a)), None
        )  # -c, and combined forms such as -lc / -ic
        if prog in SHELLS and cflag is not None:
            if cflag + 1 < len(args):
                check(args[cflag + 1], login, verdict, depth + 1, host, cwd, dict(env))
                accounted += count_submits(args[cflag + 1])
            continue
        if prog in SHELLS and inline is not None and not [a for a in pargs if not a.startswith("-")]:
            check(inline, login, verdict, depth + 1, host, cwd, dict(env))  # `bash -s <<EOF`
            continue
        if prog == "xargs":
            rest = [a for a in args if not a.startswith("-")]
            if rest:
                check(" ".join(shlex.quote(a) for a in rest), login, verdict, depth + 1, host, cwd, dict(env))
                accounted += count_submits(" ".join(rest))
            continue
        if prog in HARMLESS_MENTION:
            accounted += count_submits(" ".join(seg))

        joined = " ".join(args)
        if (prog == "watch" and re.search(r"(?<![\w-])(%s)(?![\w-])" % _QW, joined)) or (
            prog == "squeue" and any(re.match(r"^(-i\d*|--iterate(=.*)?)$", a) for a in args)
        ):
            verdict.set_deny(
                "`%s` polls the scheduler continuously. Check once, or use `sbatch --wait` / "
                "`condor_wait <log>`." % prog
            )
        inner = nested_commands(prog, args)
        if inner:
            for sub in inner:
                check(sub, login, verdict, depth + 1, host, cwd, dict(env))
            accounted += count_submits(joined)
            continue

        walker = (
            prog in ("find", "du", "tree", "ncdu", "fd", "rg", "ag", "locate")
            or (prog in ("grep", "egrep", "fgrep") and has_flag(args, "rR", ("--recursive",)))
            or (prog == "ls" and has_flag(args, "R", ("--recursive",)))
            or (prog in ("chmod", "chown", "chgrp") and has_flag(args, "R", ("--recursive",)))
            or (prog == "lfs" and args[:1] == ["find"])
        )
        bulk = prog in ("tar", "rsync") or (prog == "cp" and has_flag(args, "rRa", ("--recursive", "--archive")))
        if prog in QUERY or prog == "scontrol":
            queries += 1
        if prog in SHELLS or "/" in seg[0]:
            verdict.scripts += [(host, shell_path(a, cwd)) for a in ([seg[0]] if "/" in seg[0] else pargs[:1])
                                if a and not a.startswith("-")]
        if walker or bulk:
            hit = next((a for a in args if SHARED_ROOT.match(a)), None)
            if not hit and cwd and SHARED_ROOT.match(cwd) and not any(a.startswith(("/", "~", "$")) for a in args):
                hit = cwd  # `cd /lustre/scratch/data && find .`
            if hit:
                verdict.set_deny(
                    "Recursive walk of the shared path %s. Search only inside your own project or "
                    "workspace directory, with -maxdepth; for usage use `lfs quota` (Lustre) or "
                    "`ls -ld` / `bquota` in a job (CephFS)." % hit
                )
            elif any(BUDDY_ROOT.match(a) for a in args):
                verdict.set_ask(
                    "This walks your whole BUDDY directory on CephFS. For its size use `ls -ld "
                    "\"$BUDDY\"`, for the file count `bquota` inside a job; search one "
                    "subdirectory instead, or confirm with the user."
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
                accounted += count_submits(" ".join(seg))
                if any(a in NOT_A_SUBMISSION.get(prog, ()) for a in pargs):
                    continue  # dry run or help: nothing is submitted
                why = None
                if any(re.search(r"[$`*?]|\[", a) for a in pargs):
                    why = "it uses a shell variable, command substitution or wildcard the hook cannot resolve"
                elif prog == "sbatch" and any(k.startswith("SBATCH_") for k in list(env) + list(assigns)):
                    why = "SBATCH_* environment variables override the job script and are not read by the hook"
                if verdict.blind:
                    verdict.set_ask(BLIND_WRITER % verdict.blind)
                verdict.submits.append(dict(prog=prog, args=pargs, host=host, cwd=cwd, stdin=inline,
                                            piped=(sep == "|"), why=why))
            if prog in ("scancel", "condor_rm") and agent_jobs_only(prog, args, host):
                continue  # removing the agent's own tagged jobs needs no approval
            if CONFIRM_JOBS and prog not in AUTO_SUBMIT:
                verdict.set_ask(
                    "`%s` changes jobs on a shared cluster. Show the job script and the resource "
                    "request to the user first." % prog
                )
            continue  # whatever follows runs inside the allocation

        if not (prog in SHELLS and (cflag is not None or inline is not None)):
            verdict.blind = verdict.blind or blind_writer(prog, pargs)

        if not login:
            continue

        sub = args[0] if args else ""
        if (
            (prog in COMPUTE and sub not in ("--version", "--help", "-h", "-V"))
            or sub in COMPUTE_SUB.get(prog, ())
            or re.match(r"^mpi(run|exec)\b", prog)
        ):
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

    sleeps = [int(x) for x in re.findall(r"\bsleep\s+(\d+)", stripped)]
    if queries >= 3 and sleeps and min(sleeps) < MIN_POLL:
        verdict.set_deny(
            "Several scheduler queries with short sleeps in one command are polling. Use `sbatch "
            "--wait` / `condor_wait <log>`, or check once and wait at least %d s." % MIN_POLL
        )
    if total > accounted:
        verdict.set_ask(UNREAD_SUBMIT)


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
    written = written or {}
    keys = [(host, shell_path(path, cwd))]
    joined = path if not cwd or path.startswith(("/", "~", "$")) else cwd.rstrip("/") + "/" + path
    if not joined.startswith(("/", "~", "$")):
        keys.append((host, shell_path("~/" + joined, None)))  # an ssh session starts in the home directory
    for key in keys:
        if key in written:
            return written[key]
    return False


UNKNOWN_REQUEST = (
    "The hook cannot work out what this submission requests (%s); you currently hold %d GPU(s). "
    "Write the request with literal numbers in the job file or on the command line and submit "
    "it as a plain command, or get the user's approval."
)

SB_LONG = set("""account acctg-freq array batch bb bbf begin chdir cluster-constraint clusters comment
constraint container container-id contiguous core-spec cores-per-socket cpu-freq cpus-per-gpu
cpus-per-task deadline delay-boot dependency distribution error exclude exclusive export export-file
extra extra-node-info get-user-env gid gpu-bind gpu-freq gpus gpus-per-node gpus-per-socket
gpus-per-task gres gres-flags help hint hold ignore-pbs input job-name kill-on-invalid-dep licenses
mail-type mail-user mcs-label mem mem-bind mem-per-cpu mem-per-gpu mincpus network nice no-kill
no-requeue nodefile nodelist nodes ntasks ntasks-per-core ntasks-per-gpu ntasks-per-node
ntasks-per-socket open-mode output overcommit oversubscribe parsable partition prefer priority profile
propagate qos quiet reboot requeue reservation resv-ports segment signal sockets-per-node spread-job
switches test-only thread-spec threads-per-core time time-min tmp tres-bind tres-per-task tres-per-job
tres-per-node tres-per-socket uid usage use-min-nodes verbose version wait wait-all-nodes wckey
wrap""".split())
SB_BOOL_LONG = set("""contiguous exclusive get-user-env help hold ignore-pbs kill-on-invalid-dep nice
no-kill no-requeue overcommit oversubscribe parsable quiet reboot requeue spread-job test-only usage
use-min-nodes verbose version wait""".split())
SB_BOOL_SHORT = set("hHOQsvVWk")
SB_SHORT = {"G": "gpus", "N": "nodes", "n": "ntasks", "p": "partition", "J": "job-name", "a": "array",
            "w": "nodelist", "F": "nodefile", "D": "chdir"}


def parse_sbatch(tokens):
    """sbatch arguments (or the words of #SBATCH lines) -> (options, script, problems).

    Everything after the script name belongs to the script. Options the hook does not
    know, including abbreviated long options, are reported as problems."""
    opts, script, bad, i = {}, None, [], 0
    while i < len(tokens):
        a = tokens[i]
        if a == "<":
            script = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2
            continue
        if re.match(r"^\d*[<>&]", a):  # another redirection
            i += 2 if a.endswith(">") else 1
            continue
        if re.match(r"^\d+$", a) and i + 1 < len(tokens) and tokens[i + 1].startswith(">"):
            i += 1
            continue
        if a == ":":
            bad.append("a heterogeneous job")
        elif a == "--":
            script = tokens[i + 1] if i + 1 < len(tokens) else None
            break
        elif a.startswith("--"):
            name, eq, val = a[2:].partition("=")
            if name not in SB_LONG:
                bad.append("the option --%s, which is abbreviated or unknown to the hook" % name)
            if not eq and name not in SB_BOOL_LONG:
                i += 1
                val = tokens[i] if i < len(tokens) else ""
            opts[name] = val
        elif a.startswith("-") and len(a) > 1:
            if a[1] not in SB_BOOL_SHORT:
                val = a[2:]
                if not val:
                    i += 1
                    val = tokens[i] if i < len(tokens) else ""
                opts[SB_SHORT.get(a[1], a[1])] = val
        else:
            script = a
            break
        i += 1
    return opts, script, bad


def sbatch_directives(text):
    """Words of the #SBATCH lines that Slurm reads: those before the first command."""
    words = []
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("#SBATCH"):
            body = st[len("#SBATCH"):]
            try:
                words += shlex.split(body, comments=True)
            except ValueError:
                words += body.split()
        elif st and not st.startswith("#"):
            break
    return words


def entries_sum(value):
    """'2', 'a40:2', 'a40:2,a100:3' -> total, or None if an entry cannot be read."""
    total = 0
    for entry in value.split(","):
        n = gpu_count(entry)
        if n is None:
            return None
        total += n
    return total


def slurm_request(opts):
    """Total GPUs one job (or one array task) asks for -> (gpus, problems)."""
    bad = []
    for k in opts:
        if k.startswith("tres-per-") or k in ("gpus-per-socket", "ntasks-per-gpu"):
            bad.append("the option --%s" % k)
    try:
        nodes = max(int(x) for x in opts.get("nodes", "1").split("-"))
    except ValueError:
        nodes, _ = 1, bad.append("the node count %r" % opts.get("nodes"))
    try:
        if "ntasks" in opts:
            tasks = int(opts["ntasks"])
        elif "ntasks-per-node" in opts:
            tasks = int(opts["ntasks-per-node"]) * nodes
        else:
            tasks = 1
    except ValueError:
        tasks, _ = 1, bad.append("the task count")
    counts = []
    if "gpus" in opts:
        counts.append(entries_sum(opts["gpus"]))
    if "gpus-per-node" in opts:
        n = entries_sum(opts["gpus-per-node"])
        counts.append(None if n is None else n * nodes)
    if "gpus-per-task" in opts:
        n = entries_sum(opts["gpus-per-task"])
        counts.append(None if n is None else n * tasks)
    gres = [e for e in opts.get("gres", "").split(",") if e.split(":")[0] == "gpu"]
    if gres:
        per_node = 0
        for e in gres:
            n = 1 if e == "gpu" else gpu_count(e)
            per_node = None if n is None or per_node is None else per_node + n
        counts.append(None if per_node is None else per_node * nodes)
    if (gres or "gpus-per-node" in opts) and ("nodelist" in opts or "nodefile" in opts):
        bad.append("a node list together with a per-node GPU request")
    if None in counts:
        bad.append("the GPU count")
    elif len(counts) > 1:
        bad.append("several different GPU options at once (use one of --gpus, --gres, --gpus-per-node, --gpus-per-task)")
    return (max(counts) if counts and None not in counts else 0), bad


def slurm_limits(sub, verdict, written, totals):
    """totals: GPUs and jobs requested by earlier submits in the same command."""
    args, host, cwd = sub["args"], sub["host"], sub["cwd"]
    copts, path, bad = parse_sbatch(args)
    if sub["why"]:
        bad.append(sub["why"])
    text, readable = None, True
    if path:
        inline = same_command_file(path, host, cwd, written)
        if inline is None:
            verdict.set_ask(WRITTEN_HERE % path)
        elif inline is not False:
            text = inline
    elif sub["stdin"] is not None:
        text = sub["stdin"]  # sbatch <<EOF ... EOF
    elif "wrap" not in copts:
        bad.append("the job script arrives on standard input")
    script = (
        "command -v squeue >/dev/null && squeue --me -h -r -t PD,R,S,CF,RQ,RS "
        "-O Partition:60,tres-alloc:300 && echo @@ok"
    )
    if path and text is None:
        script += "; test -r {0} && echo @@readable; head -n 400 {0} 2>/dev/null".format(shell_path(path, cwd))
    out = probe(host, script)
    queue_text, _, remote_text = (out or "").partition("@@ok")
    if path and text is None:
        readable = "@@readable" in remote_text
        text = remote_text.replace("@@readable", "", 1)
    dopts, stray, dbad = parse_sbatch(sbatch_directives(text or ""))
    if stray:
        dbad.append("a heterogeneous job or stray word (%s) in the #SBATCH lines" % stray)
    opts = dict(dopts, **copts)  # the command line overrides the script
    bad += dbad

    spec = opts.get("array")
    tasks, capped = array_tasks(spec) if spec else (1, True)
    if tasks is None or (tasks > MAX_UNCAPPED and not capped):
        verdict.set_ask(
            "Job array of %s tasks without a concurrency cap. Add `%%N` to --array (for example "
            "--array=%s%%10) or confirm with the user." % (tasks or "unknown size", spec)
        )
    cap = re.search(r"%(\d+)", spec or "")
    concurrent = min(tasks or 10 ** 6, int(cap.group(1))) if cap else (tasks or 1)

    partition = opts.get("partition", "")
    if re.search(SLURM_LONG_PARTITIONS, partition):
        name = opts.get("job-name") or (os.path.basename(path) if path else ("wrap" if "wrap" in opts else "sbatch"))
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

    new_gpus, rbad = slurm_request(opts)
    bad += rbad
    if path and out is not None and not readable:
        bad.append("the job script %s cannot be read" % path)

    if out is None:
        verdict.set_ask("Could not check how many jobs and GPUs you already have. Confirm this submission with the user.")
        return
    lines = [l for l in queue_text.splitlines() if l.strip()]
    queued = len(lines) + totals["jobs"]
    held_gpus = 0
    for l in lines:
        plain_n = re.findall(r"gres/gpu=(\d+)", l)
        held_gpus += sum(int(n) for n in (plain_n or re.findall(r"gres/gpu:[\w.-]+=(\d+)", l)))
    if queued + (tasks or 1) > MAX_QUEUED:
        verdict.set_ask(
            "You have %d jobs queued and this adds %s (limit %d without asking). Confirm with the user."
            % (queued, tasks or "an unknown number", MAX_QUEUED)
        )
    if bad:
        verdict.set_ask(UNKNOWN_REQUEST % ("; ".join(dict.fromkeys(bad)), held_gpus))
    elif new_gpus > 0 and held_gpus + totals["gpus"] + new_gpus * concurrent > SLURM_GPU_BUDGET:
        verdict.set_ask(
            "GPU budget: you hold %d GPU(s) in running or pending jobs and this command requests %d more "
            "(budget %d without asking). Tell the user both numbers and get their approval."
            % (held_gpus, totals["gpus"] + new_gpus * concurrent, SLURM_GPU_BUDGET)
        )
    totals["gpus"] += new_gpus * concurrent
    totals["jobs"] += tasks or 1


def condor_test_gate(hours, name, host, gpus, verdict):
    """Jobs above CONDOR_LONG_HOURS need a short test run of the same JobBatchName that
    completed with exit code 0. Short runs are noted locally at submission, so the history
    scan on the submit host only reaches back to that moment."""
    if hours is None:
        return
    path = os.path.join(STATE_DIR, "condor_tests.log")
    now = int(time.time())
    if hours <= CONDOR_TEST_HOURS:
        if name:
            NOTED_NOW.add(name)
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
    if name in NOTED_NOW:
        verdict.set_ask(
            "The test run `%s` is submitted by this same command, so it cannot have completed. "
            "Wait for it, check its output, then submit the %d-hour job." % (name, hours)
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


NOTED_NOW = set()  # test runs noted during this invocation
CONDOR_KEYS = (
    r"^[[:space:]]*([+]|my[.])?[[:space:]]*"
    r"(request_gpus|requestgpus|queue|max_materialize|maxruntimehours|jobbatchname|batch_name|include|hold)"
)
CS_VALUE = {"-a", "-append", "-batch-name", "-name", "-remote", "-pool", "-maxjobs", "-addr", "-dump", "-digest"}


def parse_condor_args(tokens):
    """-> (submit file or None, appended commands, batch name, problems)."""
    path, appends, name, bad, i = None, [], None, [], 0
    while i < len(tokens):
        a = tokens[i]
        if a == "<":
            path = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2
            continue
        if re.match(r"^\d*[<>&]", a):
            i += 2 if a.endswith(">") else 1
            continue
        if a in ("-queue", "-q"):
            bad.append("a `-queue` on the command line")
            break
        if a in CS_VALUE:
            val = tokens[i + 1] if i + 1 < len(tokens) else ""
            if a in ("-a", "-append"):
                appends.append(val)
            elif a == "-batch-name":
                name = val
            i += 2
            continue
        if not a.startswith("-") and path is None:
            path = a
        i += 1
    return path, appends, name, bad


def condor_request(file_text, appends):
    """Read the lines of a submit description that matter.

    -> dict(gpus, jobs, uncapped, hours, name, bad). gpus/jobs are totals over all
    queue statements; jobs is None for a list the hook cannot count."""
    bad, cur, cap, hours, name = [], {"gpus": 0}, [None], [None], [None]
    forced = {}

    def setting(key, val, into):
        key, val = key.lower(), val.strip()
        if key in ("request_gpus", "requestgpus"):
            into["gpus"] = int(val) if val.isdigit() else None
        elif key == "max_materialize":
            into["cap"] = int(val) if val.isdigit() else None
            if not val.isdigit():
                bad.append("max_materialize = %s" % val)
        elif key == "maxruntimehours":
            if val.isdigit():
                into["hours"] = int(val)
            else:
                into["hours"] = None
                bad.append("MaxRuntimeHours = %s" % val)
        elif key in ("jobbatchname", "batch_name"):
            into["name"] = val.strip("\"'")
        elif key == "hold" and val.lower().startswith("t"):
            bad.append("the job is submitted on hold, which the group's rules do not allow")

    def split(line):
        m = re.match(r"(?i)^\s*(?:\+|my\.)?\s*([A-Za-z_]\w*)\s*=\s*(.*)$", line)
        return (m.group(1), m.group(2)) if m else None

    queue_lines = []
    for a in appends:
        if re.match(r"(?i)^\s*queue\b", a):
            queue_lines.append(a)
        elif split(a):
            setting(*split(a), into=forced)  # -append applies to every queue statement
    state, total_gpus, total_jobs, uncapped, seen = {"gpus": 0}, 0, 0, False, 0
    for line in file_text.splitlines() + queue_lines:
        if re.match(r"(?i)^\s*include\b", line):
            bad.append("an `include` line")
            continue
        q = re.match(r"(?i)^\s*queue\b(.*)$", line)
        if not q:
            if split(line):
                setting(*split(line), into=state)
            continue
        seen += 1
        eff = dict(state, **forced)
        rest = q.group(1).strip()
        m = re.match(r"^(\d+)?\s*[\w,\s]*?\bin\s*\(?([^()$]*)\)?\s*$", rest)
        if not rest:
            n = 1
        elif rest.isdigit():
            n = int(rest)
        elif m and "$" not in rest and not re.search(r"\b(from|matching)\b", rest):
            items = [x for x in re.split(r"[,\s]+", m.group(2).strip()) if x]
            n = (int(m.group(1)) if m.group(1) else 1) * len(items) if items else None
        else:
            n = None
        g, c = eff.get("gpus", 0), eff.get("cap")
        if n is None:
            total_jobs = None
            if c is None:
                uncapped = True
            if g is None or (g and c is None):
                total_gpus = None
            elif total_gpus is not None:
                total_gpus += g * (c or 0)
            continue
        if n > MAX_UNCAPPED and c is None:
            uncapped = True
        if total_jobs is not None:
            total_jobs += n
        if g is None:
            total_gpus = None
        elif total_gpus is not None:
            total_gpus += g * (min(n, c) if c else n)
    final = dict(state, **forced)
    if total_jobs is not None and total_jobs > MAX_UNCAPPED and final.get("cap") is None:
        uncapped = True
    if not seen:  # no queue statement: one job with the final settings
        total_jobs = 1
        total_gpus = final.get("gpus", 0)
    return dict(gpus=total_gpus, jobs=total_jobs, uncapped=uncapped, hours=final.get("hours"),
                name=final.get("name"), bad=bad)


def condor_limits(sub, verdict, written, totals):
    args, host, cwd = sub["args"], sub["host"], sub["cwd"]
    path, appends, batch, bad = parse_condor_args(args)
    if path == "-":
        path = None
    if sub["why"]:
        bad.append(sub["why"])
    interactive = any(a in ("-interactive", "-i") for a in args)
    text = None
    if path:
        inline = same_command_file(path, host, cwd, written)
        if inline is None:
            verdict.set_ask(WRITTEN_HERE % path)
        elif inline is not False:
            text = inline
    elif sub["stdin"] is not None:
        text = sub["stdin"]  # condor_submit <<EOF ... EOF
    elif not interactive:
        bad.append("the submit description arrives on standard input")
    script = 'command -v condor_q >/dev/null || exit 1; condor_q -global "$USER" -af JobStatus RequestGPUs 2>/dev/null; echo @@ok'
    if path and text is None:
        script += "; test -r {1} && echo @@readable; grep -iE '{0}' {1} 2>/dev/null".format(CONDOR_KEYS, shell_path(path, cwd))
    out = probe(host, script)
    if out is None:
        verdict.set_ask(
            "Could not check how many GPUs and jobs you currently hold. Run "
            "`condor_q -global $USER -af JobStatus RequestGPUs`, report it, and confirm this "
            "submission with the user."
        )
        return
    queue_text, _, remote_text = out.partition("@@ok")
    if path and text is None:
        if "@@readable" not in remote_text:
            bad.append("the submit file %s cannot be read" % path)
        text = remote_text.replace("@@readable", "", 1)
    active = [g for st, g in re.findall(r"(?m)^\s*(\d)\s+(\S+)\s*$", queue_text) if st in "125"]
    held_gpus = sum(int(float(g)) for g in active if re.match(r"^\d+(\.\d+)?$", g))

    req = condor_request(text or "", appends)
    bad += req["bad"]
    if req["gpus"] is None:
        bad.append("the GPU count")
    name = batch or req["name"]
    condor_test_gate(req["hours"], name, host, req["gpus"] or 0, verdict)

    if req["uncapped"]:
        verdict.set_ask(
            "This submit file queues %s jobs without `max_materialize`. Add a cap or confirm with the user."
            % (req["jobs"] or "a list of")
        )
    if len(active) + totals["jobs"] + (req["jobs"] or 0) > MAX_QUEUED:
        verdict.set_ask(
            "You have %d jobs queued and this adds %s (limit %d without asking). Confirm with the user."
            % (len(active) + totals["jobs"], req["jobs"] or "a list", MAX_QUEUED)
        )
    if bad:
        verdict.set_ask(UNKNOWN_REQUEST % ("; ".join(dict.fromkeys(bad)), held_gpus))
    elif req["gpus"] > 0 and held_gpus + totals["gpus"] + req["gpus"] > GPU_BUDGET:
        verdict.set_ask(
            "GPU budget: you hold %d GPU(s) in running, idle or held jobs and this command requests %d more "
            "(budget %d without asking). Tell the user both numbers and get their approval."
            % (held_gpus, totals["gpus"] + req["gpus"], GPU_BUDGET)
        )
    totals["gpus"] += req["gpus"] or 0
    totals["jobs"] += req["jobs"] or 0


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
        if os.path.isdir(lp) or src.endswith("/") or re.search(r"[*?\[]", src):
            verdict.blind = verdict.blind or "%s (copying a directory or several files)" % prog
            continue
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


def check_local(cmd, verdict, local_dir, local_written, depth=0, env=None):
    """Off-cluster: ssh calls to a cluster, plus files copied there with scp/rsync in the
    same command (read locally, since the copy has not happened yet when the hook runs)."""
    env = {} if env is None else env
    stripped, bodies = split_heredocs(cmd)
    if re.search(r"(?<![\w-])ssh(?![\w-])", stripped):
        loop_rules(stripped, verdict)  # loops on the workstation around an ssh call
    total, accounted, queries = count_submits(stripped), 0, 0
    for sep, raw in segments_sep(tokenize(stripped)):
        raw = [expand(t, env) for t in raw]
        assigns = prefix_env(raw)
        seg = strip_prefix(raw)
        if not seg:
            remember(env, assigns)
            continue
        plain, inline = without_heredocs(seg, bodies)
        prog, args = os.path.basename(seg[0]), seg[1:]
        if prog == "export":
            remember(env, {m.group(1): m.group(2) for m in map(ASSIGN.match, args) if m})
        if prog not in ("scp", "rsync"):
            for touched in modified_paths(prog, plain[1:]):
                local_written[local_path(touched, local_dir)] = None
        for target in redirect_targets(seg):
            local_written[local_path(target, local_dir)] = inline
        cflag = next((i for i, a in enumerate(args) if re.match(r"^-[A-Za-z]*c[A-Za-z]*$", a)), None)
        if prog == "ssh":
            name, target, remote = ssh_remote(seg, bodies)
            if name and remote and re.search(SSH_HOSTS, name):
                check(remote, True, verdict, 1, target, env=dict(env))
                accounted += count_submits(" ".join(plain[1:]))
        elif prog in ("scp", "rsync"):
            record_copy(prog, plain[1:], local_dir, local_written, verdict)
        elif prog in SHELLS and cflag is not None and cflag + 1 < len(args) and depth < 3:
            check_local(args[cflag + 1], verdict, local_dir, local_written, depth + 1, dict(env))
            accounted += count_submits(args[cflag + 1])
        elif prog in HARMLESS_MENTION:
            accounted += count_submits(" ".join(seg))
        elif prog == "watch" and re.search(r"\bssh\b", " ".join(args)) and re.search(r"(?<![\w-])(%s)(?![\w-])" % _QW, " ".join(args)):
            verdict.set_deny(
                "`watch` around a scheduler query polls the scheduler continuously. Check once, "
                "or use `sbatch --wait` / `condor_wait <log>`."
            )
    sleeps = [int(x) for x in re.findall(r"\bsleep\s+(\d+)", stripped)]
    if queries >= 3 and sleeps and min(sleeps) < MIN_POLL:
        verdict.set_deny(
            "Several scheduler queries with short sleeps in one command are polling. Use `sbatch "
            "--wait` / `condor_wait <log>`, or check once and wait at least %d s." % MIN_POLL
        )
    if total > accounted:
        verdict.set_ask(UNREAD_SUBMIT)


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
        check_local(cmd, verdict, payload.get("cwd") or os.getcwd(), {})

    if verdict.submits and not verdict.deny:
        limits, totals = Verdict(), {}
        rate_limit(len(verdict.submits), limits)
        for sub in verdict.submits:
            t = totals.setdefault(sub["prog"], {"gpus": 0, "jobs": 0})
            if sub["prog"] == "sbatch":
                slurm_limits(sub, limits, verdict.written, t)
            elif sub["prog"] == "condor_submit":
                condor_limits(sub, limits, verdict.written, t)
        if limits.ask:
            verdict.ask = limits.ask  # more specific than the generic confirmation

    if verdict.scripts and not verdict.deny and not verdict.ask:
        for h in dict.fromkeys(h for h, _ in verdict.scripts):
            files = " ".join(dict.fromkeys(f for hh, f in verdict.scripts if hh == h))
            found = probe(h, "grep -lE '(^|[^[:alnum:]_.-])(sbatch|condor_submit)([^[:alnum:]_.-]|$)' %s 2>/dev/null; echo @@ok" % files)
            if found and found.partition("@@ok")[0].strip():
                verdict.set_ask(
                    "The script %s submits jobs itself, so the hook cannot check those submissions. "
                    "Run each submit as a plain command instead, or get the user's approval."
                    % found.partition("@@ok")[0].split()[0]
                )

    if verdict.deny:
        respond("deny", verdict.deny)
    if verdict.ask:
        respond("ask", verdict.ask)
    sys.exit(0)


if __name__ == "__main__":
    main()
