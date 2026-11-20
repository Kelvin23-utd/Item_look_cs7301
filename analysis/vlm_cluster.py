#!/usr/bin/env python3
"""Runs the VLM naming experiment on a university Slurm cluster, driven from the Mac over SSH.

    export FB_CLUSTER=<ssh host of the cluster's login nodes>
    export FB_CLUSTER_DIR='<work or scratch storage>/$USER/fb_vlm'   # private folder; $USER expands there
    export FB_SLURM_OPTS='-A <account> -q <qos> -p <gpu partition> --gres=gpu:<gpu type>:1'   # GPU jobs
    export FB_SLURM_CPU_OPTS='-A <account> -q <qos> -p <cpu partition>'   # install and downloads (needs internet)
    export FB_CLUSTER_PYTHON='<path of a full Python 3.12 there>'      # optional; default python3.12 on PATH;
                                                     #   'uv:3.12.8' installs that CPython with uv in the folder
    export FB_CLUSTER_BOOT='<commands that put a python3 on PATH>'     # optional; only to bootstrap uv
    export FB_CLUSTER_CONSTRAINTS='<local file with name==version lines>'   # optional: every package at
                                                     #   exactly these versions (for example another site's
                                                     #   environment.txt), checked after the install
    export FB_CLUSTER_PRE='<shell commands run first in every job>'      # optional, written into env.sh
    export FB_CLUSTER_SITE=cluster2                  # optional (default cluster): the site tag of the runs and
                                                     #   the local mirror analysis/out/vlm/<site>_runs
    export FB_CLUSTER_SCRUB='word1,word2'            # optional: more words replaced in pulled text
    export FB_VLM_OUT=analysis/out/vlm_names/<pull folder>   # optional: local root (package/, <site>_runs/)
                                                     #   instead of analysis/out/vlm (a names run of another pull)
    export FB_CLUSTER_SUB=names/<pull folder>        # optional: data/, code/, runs/ and jobs/ in this subfolder
                                                     #   of the folder; weights, environment, caches stay shared

A names package (vlm_prepare.py --names-only; items.json "names_only": true) syncs without the judges' inputs,
is submitted with --no-judges, and is never a reason to download weights (submit stops when they are missing).

    python3 analysis/vlm_cluster.py check            # start estimate of a GPU job, storage
    python3 analysis/vlm_cluster.py setup            # folder (mode 700, fixed layout), env.sh; vLLM env in a CPU job
    python3 analysis/vlm_cluster.py sync             # package, pool inputs, names of primary runs, code, model file
    python3 analysis/vlm_cluster.py pack-env         # archive of the environment, unpacked on node-local storage
                                                     #   by every job (imports from the network FS are slow)
    python3 analysis/vlm_cluster.py fetch [MODEL ...]   # pinned weights, one CPU job per missing model (130 GB)
    python3 analysis/vlm_cluster.py submit --one-gpu    # one GPU job per session; the judges wait for the contestants
    python3 analysis/vlm_cluster.py submit --models model_a --no-judges [--after JOB]   # additional runs (a
                                                     #   second cluster), optionally after a smoke test
    python3 analysis/vlm_cluster.py smoke MODEL      # software test on a synthetic image (FB_SLURM_SMOKE_OPTS)
    python3 analysis/vlm_cluster.py jobs             # our jobs and the state of every run folder
    python3 analysis/vlm_cluster.py set-aside NAME preempted   # keep an interrupted attempt aside before resubmitting
    python3 analysis/vlm_cluster.py wait             # poll the queue every few minutes until our jobs are gone
    python3 analysis/vlm_cluster.py pull             # mirror runs/ to analysis/out/vlm/<site>_runs (sanitised)
    python3 analysis/vlm_cluster.py cancel           # cancel every queued or running job of this experiment
    python3 analysis/vlm_cluster.py cleanup --yes    # delete the whole folder on the cluster

Standard library only. Nothing about the cluster (host, account, partitions, folder) is written to a
repository file: it comes from the environment, and text pulled back has the folder, account, user
and node names replaced by placeholders. Every model run is a Slurm batch job that does not depend on
the Mac (the judges are chained to the contestants with Slurm dependencies); the login node only
receives files and submissions. Inside a job the session uses GPU index 0 (the allocated GPU).

Every job is submitted with --no-requeue and refuses to run when Slurm restarted it anyway, so a
preempted session is never resumed: its folder is set aside (set-aside) and the session resubmitted.
Every remote command runs under umask 077; the folder is created with mode 700.

The folder's layout is the workstation's (see vlm_ws.py) plus jobs/ (batch scripts, Slurm output).
"""
import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlm_common as vc  # noqa: E402
import vlm_ws  # noqa: E402

OUT = os.path.abspath(os.environ.get("FB_VLM_OUT") or os.path.join(HERE, "out", "vlm"))   # see the docstring
PKG = os.path.join(OUT, "package")
MIRROR = os.path.join(OUT, "cluster_runs")      # the first cluster's mirror (site "cluster"); see mirror()
SITE_RE = re.compile(r"^cluster[0-9]*$")
SET_ASIDE_TAGS = ("preempted", "cancelled", "failed", "interrupted")
RUN_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z]+_[a-z0-9]+(_smoke)?$")
DISK_BUDGET_GB = 130       # weights (models/); the environment and caches add about 20 GB
LAYOUT = vlm_ws.LAYOUT + ("jobs",)
GPU_DEFAULTS = "-N 1 -n 1 -c 8 --mem=64G"
CPU_DEFAULTS = "-N 1 -n 1 -c 4 --mem=16G"
SESSION_MIN = 60          # Slurm time limit of a naming session (the slowest took about 10 minutes on the 4090)
JUDGE_MIN = 75          # a 27 B judge: start-up from the network file system plus 88 judging requests
FETCH_MIN = 90
SETUP_MIN = 90
GENERIC_WORDS = {"normal", "debug", "dev", "gpu", "cpu", "low", "high", "short", "long", "main", "default"}


def env(name, required=True):
    v = os.environ.get(name, "")
    if required and not v:
        sys.exit("set %s first (see the docstring)" % name)
    return v


def safe_opts(name, required=True):
    v = env(name, required)
    if re.search(r"[;&|`$<>\\\n]", v):
        sys.exit("%s must hold sbatch options only" % name)
    return v


def site():
    """Site tag of this cluster's runs (FB_CLUSTER_SITE, default "cluster"): written into every job as
    FB_SITE (status.json) and naming the local mirror out/vlm/<site>_runs, so that the runs of two clusters
    can never be mixed up."""
    s = os.environ.get("FB_CLUSTER_SITE", "") or "cluster"
    if not SITE_RE.match(s):
        sys.exit("FB_CLUSTER_SITE must be 'cluster' or 'cluster' followed by a number")
    return s


def mirror():
    """Local mirror of this cluster's runs/ folder."""
    return os.path.join(OUT, site() + "_runs")


def python_spec():
    """('uv', version) when FB_CLUSTER_PYTHON is 'uv:3.12.8' (a CPython that uv installs inside the folder),
    else ('path', interpreter)."""
    v = env("FB_CLUSTER_PYTHON", False) or "python3.12"
    m = re.match(r"^uv:(3\.\d+\.\d+)$", v)
    if m:
        return "uv", m.group(1)
    if v.startswith("uv:"):
        sys.exit("FB_CLUSTER_PYTHON: write uv:<major.minor.patch>, for example uv:3.12.8")
    return "path", v


_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!_-]+)$")


def canon(name):
    """Package name as pip compares it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def constraint_lines(text):
    """The `name==version` lines of a pip freeze listing (other lines, such as the Python and GPU lines of an
    environment.txt, are skipped); sorted, one per package."""
    out = {}
    for line in text.splitlines():
        m = _PIN.match(line.strip())
        if m:
            out[canon(m.group(1))] = "%s==%s" % (m.group(1), m.group(2))
    return [out[k] for k in sorted(out)]


def compare_freeze(want_lines, got_text):
    """(missing, extra, different) between the wanted pins and an installed environment's pip freeze:
    names wanted but not installed, installed but not wanted, and (name, wanted, installed) versions."""
    want = {canon(x.split("==")[0]): x.split("==")[1] for x in want_lines}
    got = {canon(x.split("==")[0]): x.split("==")[1] for x in constraint_lines(got_text)}
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    diff = sorted((k, want[k], got[k]) for k in set(want) & set(got) if want[k] != got[k])
    return missing, extra, diff


def ssh_opts():
    sock = os.path.join(os.path.expanduser("~"), ".ssh", "fbvlmc-%C")
    return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=30",
            "-o", "ControlMaster=auto", "-o", "ControlPersist=10m", "-o", "ControlPath=" + sock]


def cluster_sub():
    """FB_CLUSTER_SUB (a relative subfolder of the folder for data/, code/, runs/ and jobs/), or ""."""
    s = os.environ.get("FB_CLUSTER_SUB", "")
    if s and not vlm_ws.SUB_RE.match(s):
        sys.exit("FB_CLUSTER_SUB must be a relative path of letters, digits, '.', '_' and '-' (no '..')")
    return s


def names_only():
    """True for a names package (vlm_prepare.py --names-only)."""
    try:
        with open(os.path.join(PKG, "items.json")) as f:
            return bool(json.load(f).get("names_only"))
    except (OSError, ValueError):
        return False


def remote_prefix(check=True):
    d = env("FB_CLUSTER_DIR")
    if '"' in d or "`" in d:
        sys.exit("FB_CLUSTER_DIR must not contain quotes")
    # nothing created there is ever readable by other accounts; $W: data/, code/, runs/ and jobs/ of this package
    p = 'umask 077; D="%s"; W="$D%s"; ' % (d, "/" + cluster_sub() if cluster_sub() else "")
    if check:
        p += '[ -f "$D/.fb_vlm_marker" ] || { echo "folder missing; run setup" >&2; exit 9; }; '
    return p


def ssh(cmd, check=True, capture=False, timeout=None, marker=True):
    full = remote_prefix(marker) + cmd
    r = subprocess.run(["ssh"] + ssh_opts() + [env("FB_CLUSTER"), full], capture_output=capture, text=True,
                       timeout=timeout)
    if check and r.returncode != 0:
        if capture:
            sys.stderr.write((r.stdout or "") + (r.stderr or ""))
        raise SystemExit("remote command failed (%d)" % r.returncode)
    return r


def ssh_out(cmd, **kw):
    return ssh(cmd, capture=True, **kw).stdout


def rsync(src, dst, delete=False, extra=()):
    args = ["rsync", "-a", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", "-e", " ".join(["ssh"] + ssh_opts())] + list(extra)
    if delete:
        args.append("--delete")
    subprocess.run(args + [src, dst], check=True)


def remote_dir():
    return ssh_out('echo "$D"').strip()


def remote_work():
    return ssh_out('echo "$W"').strip()


# ----------------------------------------------------------------------------- jobs

def env_sh(pre):
    """env.sh on the cluster: the workstation's (every cache inside the folder) after `pre`, which puts
    Python 3.12 on PATH; module systems that set PYTHONHOME or PYTHONPATH would break the venv."""
    body = vlm_ws.ENV_SH.split("\n", 1)[1]
    return ("# FindBack VLM experiment on a cluster: every weight, cache and temporary file stays inside this folder.\n"
            "type module >/dev/null 2>&1 || { [ -f /etc/profile.d/lmod.sh ] && . /etc/profile.d/lmod.sh; }\n"
            "%s\nunset PYTHONHOME PYTHONPATH\n%s"
            'export TORCH_HOME="$S/torch" HF_XET_CACHE="$S/xet"\n' % (pre, body))


def job_script(d, what, body, w=None):
    # A job that Slurm restarted (requeued after preemption, although it was submitted with --no-requeue)
    # stops before doing anything: an interrupted session is never resumed or run again under the same job.
    # FB_WORK: the session reads data/ and writes runs/ of $W (vlm_remote.py); weights and environment are $D's.
    return ("#!/bin/bash\n# FindBack VLM naming (generated by analysis/vlm_cluster.py): %s\n"
            'if [ "${SLURM_RESTART_COUNT:-0}" != "0" ]; then echo "restarted by Slurm (count $SLURM_RESTART_COUNT): '
            'not run; set the attempt aside and resubmit" >&2; exit 3; fi\n'
            'umask 077\nD="%s"\nW="%s"\n. "$D/env/env.sh" || exit 9\nexport FB_SITE=%s\nexport FB_WORK="$W"\n'
            'cd "$W/code" || exit 9\n%s\n' % (what, d, w or d, site(), body))


def submit_job(name, script, opts, defaults, minutes, deps=(), kind="", model=""):
    """Write jobs/<name>.sh, sbatch it (never requeued), record it in runs/jobs.tsv; returns the job id."""
    d, w = remote_dir(), remote_work()
    path = "$W/jobs/%s.sh" % name
    ssh("cat > \"%s\" <<'EOF'\n%sEOF\nchmod 700 \"%s\"" % (path, job_script(d, name, script, w), path))
    dep = (" --dependency=afterok:" + ":".join(str(x) for x in deps)) if deps else ""
    out = ssh_out('cd "$W/jobs" && sbatch --parsable --no-requeue %s %s -t %d -J fbvlm-%s -o "$W/jobs/%%x_%%j.out"%s '
                  '"%s"' % (defaults, opts, minutes, name, dep, path)).strip()
    jid = out.split(";")[0].split()[-1]
    if not jid.isdigit():
        raise SystemExit("sbatch: " + out)
    ssh('printf "%%s\\t%%s\\t%%s\\t%%s\\t%%s\\n" %s %s %s %s "$(date +%%Y-%%m-%%dT%%H:%%M)" >> "$W/runs/jobs.tsv"' % (
        jid, name, kind or "-", model or "-"))
    print("submitted %s: job %s%s" % (name, jid, (" (after %s)" % ",".join(map(str, deps))) if deps else ""))
    return jid


def recorded_jobs():
    rows = []
    for line in ssh_out('cat "$W/runs/jobs.tsv" 2>/dev/null || true').splitlines():
        p = line.split("\t")
        if len(p) >= 4 and p[0].isdigit():
            rows.append({"id": p[0], "name": p[1], "kind": p[2], "model": p[3]})
    return rows


def queued(ids):
    if not ids:
        return {}
    out = ssh_out('squeue -h -j %s -o "%%i|%%T|%%M|%%r" 2>/dev/null || true' % ",".join(ids))
    res = {}
    for line in out.splitlines():
        p = line.split("|")
        if len(p) == 4:
            res[p[0]] = {"state": p[1], "time": p[2], "reason": p[3]}
    return res


def accounting(ids):
    if not ids:
        return {}
    out = ssh_out('sacct -X -n -P -j %s -o JobID,JobName,State,Elapsed,ExitCode 2>/dev/null || true' % ",".join(ids))
    res = {}
    for line in out.splitlines():
        p = line.split("|")
        if len(p) == 5:
            res[p[0]] = {"name": p[1], "state": p[2], "elapsed": p[3], "exit": p[4]}
    return res


def wait_jobs(ids, every=120, quiet=False):
    last = None
    while True:
        q = queued(ids)
        line = ", ".join("%s %s %s" % (j, v["state"], v["time"] if v["state"] == "RUNNING" else v["reason"])
                         for j, v in sorted(q.items()))
        if line != last and not quiet:
            print(time.strftime("%H:%M"), line or "no job left in the queue", flush=True)
            last = line
        if not q:
            return accounting(ids)
        time.sleep(every)


# ----------------------------------------------------------------------------- commands

def cmd_check(args):
    print(ssh_out('cd /tmp; sbatch --test-only %s %s -t %d --wrap=true 2>&1 | tail -1 | sed "s/ on nodes.*//"; '
                  'df -h "$(dirname "$D")" | tail -1 | awk \'{print "storage free: "$4}\'; '
                  '[ -d "$D" ] && du -sh "$D" | cut -f1 | sed "s/^/folder size: /"; true' % (
                      GPU_DEFAULTS, safe_opts("FB_SLURM_OPTS"), SESSION_MIN), marker=False))


def setup_folder_cmd(layout=LAYOUT):
    """Shell command that creates the private folder: mode 700 from the first moment (umask 077 and
    mkdir -m 700, so no parent or child is ever created readable by others), the marker and the layout."""
    return ('umask 077; mkdir -p -m 700 "$D" && [ "$(stat -c %%a "$D")" = 700 ] && '
            'echo "FindBack VLM naming experiment; remove this whole folder to clean up" > "$D/.fb_vlm_marker" && '
            'cd "$D" && mkdir -p %s && chmod go-rwx "$D" %s' % (" ".join(layout), " ".join(layout)))


def setup_body(spec, pins, constraints, boot=""):
    """The setup job: the Python (an interpreter on the cluster, or a CPython that uv installs inside the
    folder), the venv, then vLLM and its dependencies with uv, every package held to `constraints` when
    there are any, and the resulting pip freeze."""
    con = ' -c "$D/env/constraints.txt"' if constraints else ""
    if spec[0] == "uv":
        # uv itself is bootstrapped in a throw-away venv from any python3 (FB_CLUSTER_BOOT puts one on PATH)
        py = ('BOOT="$D/scratch/uvboot"\n'
              'if [ ! -x "$BOOT/bin/uv" ]; then ( %s\n  python3 -m venv "$BOOT" && "$BOOT/bin/python" -m pip install -q '
              '--no-cache-dir%s uv ); fi\n'
              'export UV_PYTHON_INSTALL_DIR="$D/env/python" UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR="$S/uv"\n'
              '"$BOOT/bin/uv" python install %s\nPY="$("$BOOT/bin/uv" python find %s)"\necho "python: $PY"\n'
              % (boot or "true", con, spec[1], spec[1]))
    else:
        py = "PY=%s\n" % shlex.quote(spec[1])
    venv = (  # the venv is rebuilt when its Python cannot load ctypes (a Python built without libffi cannot run torch)
        'if [ ! -x "$D/env/bin/python" ] || ! "$D/env/bin/python" -c "import ctypes" 2>/dev/null; then\n'
        # moved aside, not deleted: deleting tens of thousands of files on a network file system takes minutes
        '  X="$D/scratch/old_env_$(date +%H%M%S)"; mkdir -p "$X"; for p in bin lib lib64 include pyvenv.cfg; do '
        'if [ -e "$D/env/$p" ]; then mv "$D/env/$p" "$X/"; fi; done\n'
        '  "$PY" -m venv "$D/env"; . "$D/env/env.sh"\nfi\n')
    # uv instead of pip: the same index and pins, but parallel downloads (PyPI gave about 5 MB/s per connection
    # on the first cluster, so pip's one-at-a-time downloads took most of an hour)
    install = ('export UV_CACHE_DIR="$S/uv" UV_CONCURRENT_DOWNLOADS=16\n'
               "python -m pip install -q --no-cache-dir --upgrade pip\n"
               "python -m pip install -q --no-cache-dir%s %suv\n" % (con, "" if con else "--upgrade ") +
               'python -m uv pip install --python "$D/env/bin/python"%s %s\n' % (
                   con, " ".join(shlex.quote(p) for p in pins)))
    check = ("python -c \"import sys, vllm, torch, pynvml; print('python', sys.version.split()[0], 'vllm', "
             "vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)\"\n"
             'python -m pip freeze > "$D/env/pip_freeze.txt"\n'
             'du -sh "$D/env" | cut -f1 | sed "s/^/environment: /"\necho SETUP DONE')
    return "set -e\n" + py + venv + install + check


def check_freeze(want):
    """Compare the cluster's pip freeze with the wanted pins; prints the differences, returns True when
    every package is installed at exactly the wanted version and nothing else is installed."""
    got = ssh_out('cat "$D/env/pip_freeze.txt" 2>/dev/null || true')
    missing, extra, diff = compare_freeze(want, got)
    for k, a, b in diff:
        print("version differs: %s wanted %s, installed %s" % (k, a, b))
    for k in missing:
        print("not installed: %s" % k)
    for k in extra:
        print("installed but not in the list: %s" % k)
    ok = not (missing or extra or diff)
    print("environment %s the %d listed versions" % ("matches" if ok else "does NOT match", len(want)))
    return ok


def cmd_setup(args):
    ssh(setup_folder_cmd(), marker=False)
    ssh("cat > \"$D/env/env.sh\" <<'EOF'\n%sEOF\nchmod 600 \"$D/env/env.sh\"" % env_sh(env("FB_CLUSTER_PRE", False)))
    if args.env_only:
        print(ssh_out('. "$D/env/env.sh"; echo "env.sh rewritten; python: $(command -v python)"; '
                      'command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "no nvcc on PATH"'))
        return
    want = []
    if env("FB_CLUSTER_CONSTRAINTS", False):
        with open(env("FB_CLUSTER_CONSTRAINTS")) as f:
            want = constraint_lines(f.read())
        pinned = dict(p.split("==") for p in vlm_ws.PIP_PACKAGES)
        clash = [x for x in want if canon(x.split("==")[0]) in {canon(k) for k in pinned}
                 and x.split("==")[1] != pinned.get(x.split("==")[0], x.split("==")[1])]
        if clash or not want:
            sys.exit("constraints: %s" % (clash or "no name==version line"))
        ssh("cat > \"$D/env/constraints.txt\" <<'EOF'\n%s\nEOF" % "\n".join(want))
    body = setup_body(python_spec(), vlm_ws.PIP_PACKAGES, bool(want), env("FB_CLUSTER_BOOT", False))
    jid = submit_job("setup", body, safe_opts("FB_SLURM_CPU_OPTS"), CPU_DEFAULTS, SETUP_MIN, kind="setup")
    if args.no_wait:
        return
    acc = wait_jobs([jid], every=30)
    print(acc.get(jid))
    print(ssh_out('tail -n 5 "$W/jobs/fbvlm-setup_%s.out"' % jid, check=False))
    if want:
        check_freeze(want)


def pool_runs_record(out=OUT):
    """Names of the primary contestant runs that were made elsewhere than on the cluster (the judging pool
    needs them; names only)."""
    import vlm_report as vr
    rec = {}
    for m in vc.CONTESTANTS:
        p = vr.primary_run(out, m)
        if p and p["site"] != "cluster":
            names = {}
            for r in vr.load_jsonl(os.path.join(p["dir"], "requests.jsonl")):
                if r.get("phase") == "seq" and r.get("name"):
                    names.setdefault(r["item"], []).append(r["name"])
            rec[p["id"]] = {"model": m, "names": names}
    return {"version": 1, "runs": rec}


def cmd_sync(args):
    if not os.path.exists(os.path.join(PKG, "items.json")):
        sys.exit("no package; run vlm_prepare.py first")
    # the images go only into the private folder: refuse unless it is mode 700 and owned by this account
    mode = ssh_out('stat -c "%a %U" "$D"; id -un').split()
    if len(mode) != 3 or mode[0] != "700" or mode[1] != mode[2]:
        sys.exit("the folder on the cluster is not private (mode 700, ours); nothing copied")
    if not names_only():                    # a names package has no judges: no judging pool inputs
        r = subprocess.run([sys.executable, os.path.join(HERE, "vlm_report.py"), "pool-inputs"], capture_output=True,
                           text=True)
        if r.returncode != 0:
            sys.exit("pool inputs: " + r.stderr[-500:])
        with open(os.path.join(PKG, "pool_runs.json"), "w") as f:
            json.dump(pool_runs_record(), f, indent=1)
    vc.private_models()                     # fails here, with a clear message, when the model file is missing
    if cluster_sub():
        # the subfolder is private from the first moment, inside the private folder
        ssh('mkdir -p -m 700 "$W" && cd "$W" && mkdir -p data code runs jobs && chmod 700 "$W" data code runs jobs')
    w = remote_work()
    host = env("FB_CLUSTER")
    rsync(PKG + "/", "%s:%s/data/" % (host, w), delete=True, extra=["--exclude", "judge_pool.json"])
    for f in ("vlm_remote.py", "vlm_common.py"):
        rsync(os.path.join(HERE, f), "%s:%s/code/" % (host, w))
    rsync(vc.models_path(), "%s:%s/code/%s" % (host, w, vc.MODELS_FILE))
    # the Mac's rsync may ignore --chmod: make sure nothing in the folder is readable by others
    print(ssh_out('chmod -R go-rwx "$W/data" "$W/code"; chmod 600 "$W/code/%s"; '
                  'echo "data: $(find "$W/data" -type f | wc -l) files, $(du -sh "$W/data" | cut -f1); '
                  'model file mode $(stat -c %%a "$W/code/%s"); folder mode $(stat -c %%a "$D"), work folder mode '
                  '$(stat -c %%a "$W")"' % (vc.MODELS_FILE, vc.MODELS_FILE)))


def downloaded(key):
    """Weights complete: the fetch command's marker and the pinned snapshot are both there."""
    return ssh('. "$D/env/env.sh"; [ -f "$HF_HOME/%s.fetched" ] && cd "$D/code" && python -c "import vlm_remote as v, '
               'sys; sys.exit(0 if v.snapshot_path(\'%s\') else 1)"' % (key, key), check=False,
               capture=True).returncode == 0


def folder_gb():
    """Size of the weights folder (a du over the whole folder crawls the environment for minutes)."""
    return float(ssh_out('du -sk "$D/models" | cut -f1').strip()) / 1e6


def fetch_jobs(keys, after=()):
    """{model: job id} of the download job of every model whose weights are missing: a fetch job still in
    the queue is reused, otherwise one CPU job is submitted per model (after the `after` jobs)."""
    missing = [k for k in keys if not downloaded(k)]
    if not missing:
        return {}
    rec = [r for r in recorded_jobs() if r["kind"] == "fetch"]
    q = queued([r["id"] for r in rec])
    out = {}
    for r in rec:
        if r["model"] in missing and r["id"] in q:
            out[r["model"]] = r["id"]
    new = [k for k in missing if k not in out]
    if new:
        need = sum(vc.private_models()[k]["gb"] for k in new)
        used = folder_gb()
        if used + need > DISK_BUDGET_GB:
            sys.exit("disk budget: weights %.1f GB + %.1f GB > %d GB" % (used, need, DISK_BUDGET_GB))
        print("weights %.1f GB; downloading about %.1f GB (%s)" % (used, need, ", ".join(new)))
    for k in new:
        # plain HTTPS: Hugging Face's Xet transfer wrote at about 8 MB/s to the cluster's network file system,
        # plain downloads at about 54 MB/s per file
        out[k] = submit_job("fetch_" + k, "export FB_FETCH_WORKERS=8 HF_HUB_DISABLE_XET=1\n"
                            "python vlm_remote.py fetch %s" % k,
                            safe_opts("FB_SLURM_CPU_OPTS"), CPU_DEFAULTS, FETCH_MIN, deps=after, kind="fetch",
                            model=k)
    return out


def cmd_fetch(args):
    ids = fetch_jobs([vc.canonical_key(m) for m in (args.models or vc.CONTESTANTS + vc.JUDGES)],
                     after=[args.after] if args.after else ())
    if ids and not args.no_wait:
        print(wait_jobs(list(ids.values()), every=60))


def cmd_submit(args):
    """Contestant sessions (the model-a one is an additional run: its primary run is the workstation's),
    then the judges, which build the pool from the primary runs and wait for the contestants' jobs."""
    import vlm_report as vr
    models = [vc.canonical_key(m) for m in (args.models if args.models is not None else vc.CONTESTANTS)]
    judges = [] if args.no_judges else [vc.canonical_key(j) for j in (args.judges or vc.JUDGES)]
    if getattr(args, "after", None) and not args.after.isdigit():
        sys.exit("--after takes a Slurm job id")
    if judges and site() in vr.ADDITIONAL_SITES:
        sys.exit("site %s gives additional runs only (cost and name agreement): no judge runs there; "
                 "add --no-judges" % site())
    if judges and names_only():
        sys.exit("a names package has no judges: add --no-judges")
    date = datetime.date.today().isoformat()
    # every attempt gets its own folder: an earlier attempt of the same name must have been set aside first
    # (set-aside), and a job of that name must not be queued or running
    names = ["%s_%s" % (date, m) for m in models + judges]
    held = ssh_out('cd "$W/runs" 2>/dev/null || exit 0; for n in %s; do [ -e "$n" ] && echo "$n"; done; true'
                   % " ".join(names)).split()
    busy = [r["name"] for r in recorded_jobs() if r["name"] in names and r["id"] in queued(
        [x["id"] for x in recorded_jobs() if x["name"] in names])]
    if held or busy:
        sys.exit("not submitted: %s%s" % (("run folder exists (set it aside first): " + ", ".join(held)) if held else "",
                                          ("; job queued or running: " + ", ".join(busy)) if busy else ""))
    if names_only():
        # a names run never downloads weights: they must be in the folder already
        missing = [m for m in models if not downloaded(m)]
        if missing:
            sys.exit("weights missing for %s; not submitted (fetch them first if that is intended)" % ", ".join(missing))
    cmd_sync(args)
    fetch = fetch_jobs(models + judges)
    gpu_opts = safe_opts("FB_SLURM_OPTS")
    jobs = {}
    for m in models:
        one = " --gpus 1" if args.one_gpu and vc.PUBLIC_MODELS[m]["gpus"] > 1 else ""
        jobs[m] = submit_job("%s_%s" % (date, m), "python vlm_remote.py job %s --name %s_%s --time-limit-s %d%s" % (
            m, date, m, SESSION_MIN * 60, one), gpu_opts, GPU_DEFAULTS, SESSION_MIN,
            deps=([fetch[m]] if m in fetch else []) + ([args.after] if args.after else []), kind="session", model=m)
    if not judges:
        return
    sessions = [r for r in recorded_jobs() if r["kind"] == "session"]
    q = queued([r["id"] for r in sessions])
    waiting = {r["model"]: r for r in sessions if r["id"] in q}
    pool_runs, deps = [], []
    for c in vc.CONTESTANTS:
        p = vr.primary_run(OUT, c)
        if p and p["site"] == "cluster":
            pool_runs.append(os.path.basename(p["dir"]))
        elif p:
            print("pool: %s from %s (names in pool_runs.json)" % (vc.alias_label(c), p["id"]))
        elif c in jobs:
            pool_runs.append("%s_%s" % (date, c))
            deps.append(jobs[c])
        elif c in waiting:
            pool_runs.append(waiting[c]["name"])         # a session submitted earlier, still queued or running
            deps.append(waiting[c]["id"])
            print("pool: %s from job %s (still running)" % (vc.alias_label(c), waiting[c]["id"]))
        else:
            print("pool: no run of %s; the pool will lack it" % vc.alias_label(c))
    for j in judges:
        one = " --gpus 1" if args.one_gpu and vc.PUBLIC_MODELS[j]["gpus"] > 1 else ""
        script = "python vlm_remote.py job %s --name %s_%s --time-limit-s %d%s --names-file \"$W/data/pool_runs.json\"%s" % (
            j, date, j, JUDGE_MIN * 60, one, "".join(" --pool-run %s" % r for r in pool_runs))
        submit_job("%s_%s" % (date, j), script, gpu_opts, GPU_DEFAULTS, JUDGE_MIN,
                   deps=deps + ([fetch[j]] if j in fetch else []), kind="judge", model=j)


def cmd_pack_env(args):
    """An archive of the Python environment (scratch/env.tar) that each job unpacks on node-local storage
    (importing from the network file system took minutes per server start). A CPU job."""
    body = ('tar -C "$D" -cf "$D/scratch/env.tar.part" env && mv "$D/scratch/env.tar.part" "$D/scratch/env.tar" && '
            'ls -la "$D/scratch/env.tar"')
    jid = submit_job("pack_env", body, safe_opts("FB_SLURM_CPU_OPTS"), CPU_DEFAULTS, 60, kind="setup")
    if not args.no_wait:
        print(wait_jobs([jid], every=30))


def cmd_smoke(args):
    """A server start and three requests on a synthetic image (no home photo) to test the software on a GPU
    type; its output is not used. Slurm options: FB_SLURM_SMOKE_OPTS, else FB_SLURM_OPTS."""
    key = vc.canonical_key(args.model)
    opts = safe_opts("FB_SLURM_SMOKE_OPTS", False) or safe_opts("FB_SLURM_OPTS")
    name = "%s_%s_smoke" % (datetime.date.today().isoformat(), key)
    submit_job(name, "python vlm_remote.py smoke %s --run \"$W/runs/%s\" --no-hold --max-seconds 1500" % (key, name),
               opts, GPU_DEFAULTS, 30, kind="smoke", model=key)


def cmd_jobs(args):
    rows = recorded_jobs()
    ids = [r["id"] for r in rows]
    q, acc = queued(ids), accounting(ids)
    for r in rows:
        s = q.get(r["id"]) or {}
        a = acc.get(r["id"]) or {}
        print("%-8s %-34s %-10s %s" % (r["id"], r["name"], s.get("state") or a.get("state", "?"),
                                       s.get("time") if s.get("state") == "RUNNING" else
                                       (s.get("reason") or "%s exit %s" % (a.get("elapsed", ""), a.get("exit", "")))))
    print()
    print(ssh_out('cd "$W/runs" && for f in */status.json; do [ -f "$f" ] || continue; '
                  'python3 -c "import json,sys; s=json.load(open(sys.argv[1])); n=s.get(\'notes\') or [\'\']; '
                  'print(\'%-44s %-9s %s\' % (sys.argv[1][:-12], s.get(\'state\'), n[-1]))" "$f"; done; '
                  'ls "$W/runs" | grep "\\.dropped$" || true', check=False).rstrip())


def set_aside_cmd(name, tag, stamp):
    """Shell command that renames runs/<name> to runs/<name>_<tag>_<stamp> unless it finished ('done')."""
    if not RUN_NAME_RE.match(name) or tag not in SET_ASIDE_TAGS or not re.match(r"^\d{6}$", stamp):
        sys.exit("set-aside: NAME is a run folder name (<date>_<alias>), TAG one of %s" % ", ".join(SET_ASIDE_TAGS))
    return ('cd "$W/runs" && [ -d "%s" ] || { echo "no run folder %s" >&2; exit 3; }; '
            'if grep -q \'"state": "done"\' "%s/status.json" 2>/dev/null; then echo "%s finished: not set aside" >&2; '
            'exit 4; fi; mv "%s" "%s_%s_%s" && echo "set aside: %s_%s_%s"' % (
                name, name, name, name, name, name, tag, stamp, name, tag, stamp))


def cmd_set_aside(args):
    """Keep an interrupted attempt (preempted, cancelled, failed) aside under a clear tag, so that the session
    can be resubmitted into a fresh folder; never for a run that finished or whose job is still in the queue."""
    rows = [r for r in recorded_jobs() if r["name"] == args.name]
    if queued([r["id"] for r in rows]):
        sys.exit("a job of %s is still queued or running; cancel it first" % args.name)
    print(ssh_out(set_aside_cmd(args.name, args.tag, time.strftime("%H%M%S"))).strip())


def cmd_wait(args):
    ids = [r["id"] for r in recorded_jobs()]
    acc = wait_jobs(ids, every=args.every)
    for j, a in sorted(acc.items()):
        print(j, a)


def cmd_cancel(args):
    ids = list(queued([r["id"] for r in recorded_jobs()]))
    if ids:
        ssh("scancel %s" % " ".join(ids))
    print("cancelled:", ", ".join(ids) or "nothing")


def slurm_names():
    """Account, partition and QOS names in the Slurm options (to be replaced in pulled text)."""
    names = set()
    for v in (os.environ.get("FB_SLURM_OPTS", ""), os.environ.get("FB_SLURM_CPU_OPTS", "")):
        toks = shlex.split(v)
        for i, t in enumerate(toks):
            val = None
            if t in ("-A", "--account", "-p", "--partition", "-q", "--qos") and i + 1 < len(toks):
                val = toks[i + 1]
            elif re.match(r"--(account|partition|qos)=", t):
                val = t.split("=", 1)[1]
            if val:
                names.update(x for x in val.split(",") if x and x.lower() not in GENERIC_WORDS)
    return names


def replacements(d, home, user, hosts, nodes, cluster, slurm, extra=(), real_dirs=()):
    """(compiled pattern, placeholder) pairs, longest first; short names only as whole words. `real_dirs`:
    the folder's other spellings (for example the target of a symbolic link), `extra`: more words."""
    pairs = [(d, "<CL_DIR>"), (home, "<CL_HOME>"), (cluster, "<CL>")]
    pairs += [(x, "<CL_DIR>") for x in real_dirs if x != d]
    pairs += [(h, "<CL_NODE>") for h in hosts | nodes]
    if cluster and "." in cluster:
        labels = cluster.split(".")
        pairs += [(".".join(labels[-2:]), "<CL_DOMAIN>"), (labels[0], "<CL>")]
    pairs += [(user, "<CL_USER>")] + [(s, "<CL_SLURM>") for s in slurm] + [(w, "<CL>") for w in extra]
    out = []
    for a, b in sorted({p for p in pairs if p[0]}, key=lambda x: -len(x[0])):
        pat = re.escape(a) if "/" in a or "." in a else r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(a)
        out.append((re.compile(pat), b))
    return out


def sanitize(root, pats):
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if not fn.endswith((".json", ".jsonl", ".log", ".out", ".csv", ".txt", ".tsv", ".sh", ".dropped")):
                continue
            p = os.path.join(dp, fn)
            with open(p, encoding="utf-8", errors="replace") as f:
                s = f.read()
            t = s
            for pat, rep in pats:
                t = pat.sub(rep, t)
            if t != s:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(t)


def cmd_pull(args):
    d, w = remote_dir(), remote_work()
    info = ssh_out('echo "$HOME"; id -un; hostname -s; hostname -f; hostname', marker=False).split()
    ids = [r["id"] for r in recorded_jobs()]
    nodes = set()
    if ids:
        lst = ssh_out('sacct -X -n -P -j %s -o NodeList 2>/dev/null | sort -u | grep -v "None" | '
                      'xargs -r -n1 scontrol show hostnames 2>/dev/null || true' % ",".join(ids))
        nodes = {x for x in lst.split() if x}
    host = env("FB_CLUSTER")
    real = ssh_out('readlink -f "$D"').strip()
    scrub = [w.strip() for w in env("FB_CLUSTER_SCRUB", False).split(",") if w.strip()]
    out = mirror()
    os.makedirs(os.path.join(out, "_slurm"), exist_ok=True)
    rsync("%s:%s/runs/" % (host, w), out + "/", extra=["--exclude", "*.lock"])
    rsync("%s:%s/jobs/" % (host, w), os.path.join(out, "_slurm") + "/")
    pats = replacements(d, info[0], info[1], set(info[2:]), nodes, host, slurm_names(), scrub, [real])
    sanitize(out, pats)
    print("pulled to", os.path.relpath(out, os.path.join(HERE, "..")))


def cmd_cleanup(args):
    if not args.yes:
        sys.exit("this deletes the whole folder on the cluster; add --yes")
    left = queued([r["id"] for r in recorded_jobs()])
    if left:
        sys.exit("jobs of the experiment are still queued or running (cancel them first): %s" % ", ".join(left))
    ssh('rm -rf "$D" && echo removed')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "sync", "jobs", "pull", "cancel"):
        sub.add_parser(name)
    p = sub.add_parser("setup")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--env-only", action="store_true", help="only rewrite env.sh (for example after changing "
                   "FB_CLUSTER_PRE)")
    p = sub.add_parser("fetch")
    p.add_argument("models", nargs="*")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--after", help="start the downloads after this job (for example the setup job)")
    p = sub.add_parser("submit")
    p.add_argument("--models", nargs="*", help="contestant sessions to run (default: all four)")
    p.add_argument("--judges", nargs="*", help="judges to run (default: both)")
    p.add_argument("--no-judges", action="store_true")
    p.add_argument("--after", help="start the sessions only after this job ended well (for example a smoke test)")
    p.add_argument("--one-gpu", action="store_true", help="run a model the design gives two GPUs on one GPU "
                   "(a large-memory GPU; recorded in the run's status as a deviation)")
    p = sub.add_parser("pack-env")
    p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("smoke")
    p.add_argument("model")
    p = sub.add_parser("wait")
    p.add_argument("--every", type=int, default=180)
    p = sub.add_parser("set-aside")
    p.add_argument("name", help="run folder name in runs/ (<date>_<alias>)")
    p.add_argument("tag", choices=SET_ASIDE_TAGS)
    p = sub.add_parser("cleanup")
    p.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    fn = {"check": cmd_check, "setup": cmd_setup, "sync": cmd_sync, "fetch": cmd_fetch, "submit": cmd_submit,
          "jobs": cmd_jobs, "wait": cmd_wait, "smoke": cmd_smoke, "set-aside": cmd_set_aside,
          "pack-env": cmd_pack_env, "pull": cmd_pull, "cancel": cmd_cancel, "cleanup": cmd_cleanup}
    sys.exit(fn[args.cmd](args) or 0)


if __name__ == "__main__":
    main()
