#!/usr/bin/env python3
"""Runs the VLM naming experiment on a university Slurm cluster, driven from the Mac over SSH.

    export FB_CLUSTER=<ssh host of the cluster's login nodes>
    export FB_CLUSTER_DIR='<work or scratch storage>/$USER/fb_vlm'   # private folder; $USER expands there
    export FB_SLURM_OPTS='-A <account> -q <qos> -p <gpu partition> --gres=gpu:<gpu type>:1'   # GPU jobs
    export FB_SLURM_CPU_OPTS='-A <account> -q <qos> -p <cpu partition>'   # install and downloads (needs internet)
    export FB_CLUSTER_PYTHON='<path of a full Python 3.12 there>'      # optional; default python3.12 on PATH
    export FB_CLUSTER_PRE='<shell commands run first in every job>'      # optional, written into env.sh

    python3 analysis/vlm_cluster.py check            # start estimate of a GPU job, storage
    python3 analysis/vlm_cluster.py setup            # folder (mode 700, fixed layout), env.sh; vLLM env in a CPU job
    python3 analysis/vlm_cluster.py sync             # package, pool inputs, names of primary runs, code, model file
    python3 analysis/vlm_cluster.py pack-env         # archive of the environment, unpacked on node-local storage
                                                     #   by every job (imports from the network FS are slow)
    python3 analysis/vlm_cluster.py fetch [MODEL ...]   # pinned weights, one CPU job per missing model (130 GB)
    python3 analysis/vlm_cluster.py submit --one-gpu    # one GPU job per session; the judges wait for the contestants
    python3 analysis/vlm_cluster.py smoke MODEL      # software test on a synthetic image (FB_SLURM_SMOKE_OPTS)
    python3 analysis/vlm_cluster.py jobs             # our jobs and the state of every run folder
    python3 analysis/vlm_cluster.py wait             # poll the queue every few minutes until our jobs are gone
    python3 analysis/vlm_cluster.py pull             # mirror runs/ to analysis/out/vlm/cluster_runs (sanitised)
    python3 analysis/vlm_cluster.py cancel           # cancel every queued or running job of this experiment
    python3 analysis/vlm_cluster.py cleanup --yes    # delete the whole folder on the cluster

Standard library only. Nothing about the cluster (host, account, partitions, folder) is written to a
repository file: it comes from the environment, and text pulled back has the folder, account, user
and node names replaced by placeholders. Every model run is a Slurm batch job that does not depend on
the Mac (the judges are chained to the contestants with Slurm dependencies); the login node only
receives files and submissions. Inside a job the session uses GPU index 0 (the allocated GPU).

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

OUT = os.path.join(HERE, "out", "vlm")
PKG = os.path.join(OUT, "package")
MIRROR = os.path.join(OUT, "cluster_runs")
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


def ssh_opts():
    sock = os.path.join(os.path.expanduser("~"), ".ssh", "fbvlmc-%C")
    return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=30",
            "-o", "ControlMaster=auto", "-o", "ControlPersist=10m", "-o", "ControlPath=" + sock]


def remote_prefix(check=True):
    d = env("FB_CLUSTER_DIR")
    if '"' in d or "`" in d:
        sys.exit("FB_CLUSTER_DIR must not contain quotes")
    p = 'D="%s"; ' % d
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


# ----------------------------------------------------------------------------- jobs

def env_sh(pre):
    """env.sh on the cluster: the workstation's (every cache inside the folder) after `pre`, which puts
    Python 3.12 on PATH; module systems that set PYTHONHOME or PYTHONPATH would break the venv."""
    body = vlm_ws.ENV_SH.split("\n", 1)[1]
    return ("# FindBack VLM experiment on a cluster: every weight, cache and temporary file stays inside this folder.\n"
            "type module >/dev/null 2>&1 || { [ -f /etc/profile.d/lmod.sh ] && . /etc/profile.d/lmod.sh; }\n"
            "%s\nunset PYTHONHOME PYTHONPATH\n%s"
            'export TORCH_HOME="$S/torch" HF_XET_CACHE="$S/xet"\n' % (pre, body))


def job_script(d, what, body):
    return ("#!/bin/bash\n# FindBack VLM naming (generated by analysis/vlm_cluster.py): %s\n"
            'D="%s"\n. "$D/env/env.sh" || exit 9\nexport FB_SITE=cluster\ncd "$D/code" || exit 9\n%s\n'
            % (what, d, body))


def submit_job(name, script, opts, defaults, minutes, deps=(), kind="", model=""):
    """Write jobs/<name>.sh, sbatch it, record it in runs/jobs.tsv; returns the job id."""
    d = remote_dir()
    path = "$D/jobs/%s.sh" % name
    ssh("cat > \"%s\" <<'EOF'\n%sEOF\nchmod 700 \"%s\"" % (path, job_script(d, name, script), path))
    dep = (" --dependency=afterok:" + ":".join(str(x) for x in deps)) if deps else ""
    out = ssh_out('cd "$D/jobs" && sbatch --parsable %s %s -t %d -J fbvlm-%s -o "$D/jobs/%%x_%%j.out"%s "%s"' % (
        defaults, opts, minutes, name, dep, path)).strip()
    jid = out.split(";")[0].split()[-1]
    if not jid.isdigit():
        raise SystemExit("sbatch: " + out)
    ssh('printf "%%s\\t%%s\\t%%s\\t%%s\\t%%s\\n" %s %s %s %s "$(date +%%Y-%%m-%%dT%%H:%%M)" >> "$D/runs/jobs.tsv"' % (
        jid, name, kind or "-", model or "-"))
    print("submitted %s: job %s%s" % (name, jid, (" (after %s)" % ",".join(map(str, deps))) if deps else ""))
    return jid


def recorded_jobs():
    rows = []
    for line in ssh_out('cat "$D/runs/jobs.tsv" 2>/dev/null || true').splitlines():
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


def cmd_setup(args):
    ssh('mkdir -p "$D" && chmod 700 "$D" && '
        'echo "FindBack VLM naming experiment; remove this whole folder to clean up" > "$D/.fb_vlm_marker" && '
        # mode 700 on the folder keeps everything inside private; new files get umask 077 from env.sh
        'cd "$D" && mkdir -p %s && chmod go-rwx "$D" %s' % (" ".join(LAYOUT), " ".join(LAYOUT)), marker=False)
    ssh("cat > \"$D/env/env.sh\" <<'EOF'\n%sEOF\nchmod 600 \"$D/env/env.sh\"" % env_sh(env("FB_CLUSTER_PRE", False)))
    if args.env_only:
        print(ssh_out('. "$D/env/env.sh"; echo "env.sh rewritten; python: $(command -v python)"; '
                      'command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "no nvcc on PATH"'))
        return
    # uv instead of pip: the same index and pins, but parallel downloads (PyPI gave about 5 MB/s per
    # connection on the cluster, so pip's one-at-a-time downloads took most of an hour)
    # the venv is rebuilt when its Python cannot load ctypes (a Python built without libffi cannot run torch)
    py = shlex.quote(env("FB_CLUSTER_PYTHON", False) or "python3.12")
    body = ("set -e\n"
            'if [ ! -x "$D/env/bin/python" ] || ! "$D/env/bin/python" -c "import ctypes" 2>/dev/null; then\n'
            # moved aside, not deleted: deleting tens of thousands of files on a network file system takes minutes
            '  X="$D/scratch/old_env_$(date +%%H%%M%%S)"; mkdir -p "$X"; for p in bin lib lib64 include pyvenv.cfg; do '
            'if [ -e "$D/env/$p" ]; then mv "$D/env/$p" "$X/"; fi; done\n'
            '  %s -m venv "$D/env"; . "$D/env/env.sh"\nfi\n' % py +
            'export UV_CACHE_DIR="$S/uv" UV_CONCURRENT_DOWNLOADS=16\n'
            "python -m pip install -q --no-cache-dir --upgrade pip uv\n"
            'python -m uv pip install --python "$D/env/bin/python" %s\n'
            "python -c \"import vllm, torch, pynvml; print('vllm', vllm.__version__, 'torch', torch.__version__, "
            "'cuda', torch.version.cuda)\"\n"
            'python -m pip freeze > "$D/env/pip_freeze.txt"\n'
            'du -sh "$D/env" | cut -f1 | sed "s/^/environment: /"\necho SETUP DONE' % " ".join(
                shlex.quote(p) for p in vlm_ws.PIP_PACKAGES))
    jid = submit_job("setup", body, safe_opts("FB_SLURM_CPU_OPTS"), CPU_DEFAULTS, SETUP_MIN, kind="setup")
    if args.no_wait:
        return
    acc = wait_jobs([jid], every=30)
    print(acc.get(jid))
    print(ssh_out('tail -n 5 "$D/jobs/fbvlm-setup_%s.out"' % jid, check=False))


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
    r = subprocess.run([sys.executable, os.path.join(HERE, "vlm_report.py"), "pool-inputs"], capture_output=True,
                       text=True)
    if r.returncode != 0:
        sys.exit("pool inputs: " + r.stderr[-500:])
    with open(os.path.join(PKG, "pool_runs.json"), "w") as f:
        json.dump(pool_runs_record(), f, indent=1)
    vc.private_models()                     # fails here, with a clear message, when the model file is missing
    d = remote_dir()
    host = env("FB_CLUSTER")
    rsync(PKG + "/", "%s:%s/data/" % (host, d), delete=True, extra=["--exclude", "judge_pool.json"])
    for f in ("vlm_remote.py", "vlm_common.py"):
        rsync(os.path.join(HERE, f), "%s:%s/code/" % (host, d))
    rsync(vc.models_path(), "%s:%s/code/%s" % (host, d, vc.MODELS_FILE))
    # the Mac's rsync may ignore --chmod: make sure nothing in the folder is readable by others
    print(ssh_out('chmod -R go-rwx "$D/data" "$D/code"; '
                  'echo "data: $(find "$D/data" -type f | wc -l) files, $(du -sh "$D/data" | cut -f1)"'))


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
    cmd_sync(args)
    models = [vc.canonical_key(m) for m in (args.models if args.models is not None else vc.CONTESTANTS)]
    judges = [] if args.no_judges else [vc.canonical_key(j) for j in (args.judges or vc.JUDGES)]
    date = datetime.date.today().isoformat()
    fetch = fetch_jobs(models + judges)
    gpu_opts = safe_opts("FB_SLURM_OPTS")
    jobs = {}
    for m in models:
        one = " --gpus 1" if args.one_gpu and vc.PUBLIC_MODELS[m]["gpus"] > 1 else ""
        jobs[m] = submit_job("%s_%s" % (date, m), "python vlm_remote.py job %s --name %s_%s --time-limit-s %d%s" % (
            m, date, m, SESSION_MIN * 60, one), gpu_opts, GPU_DEFAULTS, SESSION_MIN,
            deps=[fetch[m]] if m in fetch else [], kind="session", model=m)
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
        script = "python vlm_remote.py job %s --name %s_%s --time-limit-s %d%s --names-file \"$D/data/pool_runs.json\"%s" % (
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
    submit_job(name, "python vlm_remote.py smoke %s --run \"$D/runs/%s\" --no-hold --max-seconds 1500" % (key, name),
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
    print(ssh_out('cd "$D/runs" && for f in */status.json; do [ -f "$f" ] || continue; '
                  'python3 -c "import json,sys; s=json.load(open(sys.argv[1])); n=s.get(\'notes\') or [\'\']; '
                  'print(\'%-44s %-9s %s\' % (sys.argv[1][:-12], s.get(\'state\'), n[-1]))" "$f"; done; '
                  'ls "$D/runs" | grep "\\.dropped$" || true', check=False).rstrip())


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


def replacements(d, home, user, hosts, nodes, cluster, slurm):
    """(compiled pattern, placeholder) pairs, longest first; short names only as whole words."""
    pairs = [(d, "<CL_DIR>"), (home, "<CL_HOME>"), (cluster, "<CL>")]
    pairs += [(h, "<CL_NODE>") for h in hosts | nodes]
    if cluster and "." in cluster:
        labels = cluster.split(".")
        pairs += [(".".join(labels[-2:]), "<CL_DOMAIN>"), (labels[0], "<CL>")]
    pairs += [(user, "<CL_USER>")] + [(s, "<CL_SLURM>") for s in slurm]
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
    d = remote_dir()
    info = ssh_out('echo "$HOME"; id -un; hostname -s; hostname -f; hostname', marker=False).split()
    ids = [r["id"] for r in recorded_jobs()]
    nodes = set()
    if ids:
        lst = ssh_out('sacct -X -n -P -j %s -o NodeList 2>/dev/null | sort -u | grep -v "None" | '
                      'xargs -r -n1 scontrol show hostnames 2>/dev/null || true' % ",".join(ids))
        nodes = {x for x in lst.split() if x}
    host = env("FB_CLUSTER")
    os.makedirs(os.path.join(MIRROR, "_slurm"), exist_ok=True)
    rsync("%s:%s/runs/" % (host, d), MIRROR + "/", extra=["--exclude", "*.lock"])
    rsync("%s:%s/jobs/" % (host, d), os.path.join(MIRROR, "_slurm") + "/")
    pats = replacements(d, info[0], info[1], set(info[2:]), nodes, host, slurm_names())
    sanitize(MIRROR, pats)
    print("pulled to", os.path.relpath(MIRROR, os.path.join(HERE, "..")))


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
    p.add_argument("--one-gpu", action="store_true", help="run a model the design gives two GPUs on one GPU "
                   "(a large-memory GPU; recorded in the run's status as a deviation)")
    p = sub.add_parser("pack-env")
    p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("smoke")
    p.add_argument("model")
    p = sub.add_parser("wait")
    p.add_argument("--every", type=int, default=180)
    p = sub.add_parser("cleanup")
    p.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    fn = {"check": cmd_check, "setup": cmd_setup, "sync": cmd_sync, "fetch": cmd_fetch, "submit": cmd_submit,
          "jobs": cmd_jobs, "wait": cmd_wait, "smoke": cmd_smoke,
          "pack-env": cmd_pack_env, "pull": cmd_pull, "cancel": cmd_cancel, "cleanup": cmd_cleanup}
    sys.exit(fn[args.cmd](args) or 0)


if __name__ == "__main__":
    main()
