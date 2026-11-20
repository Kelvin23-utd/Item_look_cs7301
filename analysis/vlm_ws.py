#!/usr/bin/env python3
"""Drives the VLM naming experiment on the lab workstation from the Mac, over SSH.

    export FB_WORKSTATION=<ssh alias of the lab workstation>
    # MODEL below is an alias (model_a, model_b, model_c, model_d, judge_a, judge_b)
    export FB_WS_DIR='/data/$USER/fb_vlm'       # private folder there; $USER expands on the workstation

    python3 analysis/vlm_ws.py status            # who is logged in, load, GPUs, disk use of the folder
    python3 analysis/vlm_ws.py setup             # folder (mode 700, fixed layout), env.sh, Python env with vLLM
    python3 analysis/vlm_ws.py sync              # package, pool inputs, code and the private model file
    python3 analysis/vlm_ws.py fetch MODEL       # pinned weights into the folder's models/
    python3 analysis/vlm_ws.py queue-start       # detached queue on the workstation: waits for free GPUs and
                                                 #   runs every remaining session (survives the Mac sleeping)
    python3 analysis/vlm_ws.py queue-start --models model_b model_c --until 2026-11-23T23:59
                                                 # only these contestants (additional runs: no judge, no pool),
                                                 #   giving up at that workstation time instead of after 24 h
    python3 analysis/vlm_ws.py queue-status      # the queue's runs/STATUS
    python3 analysis/vlm_ws.py queue-stop        # ask the queue (and its running session) to stop
    python3 analysis/vlm_ws.py roundtrip MODEL   # short session with the round trip from the Mac (needs the Mac)
    python3 analysis/vlm_ws.py run MODEL         # one full session by hand, with the round trip
    python3 analysis/vlm_ws.py judge JUDGE       # one judging session by hand
    python3 analysis/vlm_ws.py pull              # mirror runs/ to analysis/out/vlm/ws_runs (paths replaced)
    python3 analysis/vlm_ws.py rm-weights MODEL  # delete one model's weights
    python3 analysis/vlm_ws.py procs             # processes of this experiment still running
    python3 analysis/vlm_ws.py cleanup --yes     # delete the whole folder on the workstation

A names run of another pull (vlm_prepare.py --names-only), kept apart from the experiment on both sides:

    export FB_VLM_OUT=analysis/out/vlm_names/<pull folder>   # local root: package/, ws_runs/ (default analysis/out/vlm)
    export FB_WS_SUB=names/<pull folder>          # data/, code/ and runs/ in this subfolder of the folder;
                                                 #   weights, environment and caches stay shared
    python3 analysis/vlm_ws.py run model_c        # a names package: no judges' inputs, no round trip (--no-hold)
    python3 analysis/vlm_ws.py pull

Standard library only. The SSH alias and the folder are read from the environment and never written
to a file; text pulled back has the workstation's paths and account replaced by placeholders. One SSH
connection is shared (ControlMaster) because opening one can take up to a minute.

The folder's layout (one directory, mode 700; `cleanup` removes exactly this directory):
  data/      what was copied from the Mac (images, item list, pool inputs, judge pool)
  code/      vlm_remote.py, vlm_common.py and the private model file
  models/    every weight cache (HF_HOME)
  env/       the Python environment and env.sh
  runs/      <date>_<name>/ per session, queue.log, STATUS, queue.lock
  scratch/   HOME, caches and temporary files of the processes started there
"""
import argparse
import datetime
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlm_common as vc  # noqa: E402

OUT = os.path.abspath(os.environ.get("FB_VLM_OUT") or os.path.join(HERE, "out", "vlm"))   # see the docstring
PKG = os.path.join(OUT, "package")
MIRROR = os.path.join(OUT, "ws_runs")
DISK_BUDGET_GB = 80
VLLM_VERSION = "0.33.0"
PIP_PACKAGES = ["vllm==" + VLLM_VERSION, "nvidia-ml-py==13.615.71"]
MODEL_KEYS = sorted(vc.PUBLIC_MODELS)      # aliases; identifiers in the private model file (vlm_models.private.json)
POLL_S = 300            # while waiting by hand: one nvidia-smi query every 5 minutes
FREE_POLLS = 3          # three free polls in a row = at least 10 minutes without any process
LAYOUT = ("data", "code", "models", "env", "runs", "scratch/home", "scratch/cache", "scratch/tmp")
ENV_SH = r'''# FindBack VLM experiment: every weight, cache and temporary file stays inside this folder.
export FBV="$D"
export HOME="$D/scratch/home"
S="$D/scratch/cache"
export XDG_CACHE_HOME="$S" XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share"
export HF_HOME="$D/models" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TMPDIR="$D/scratch/tmp" PIP_CACHE_DIR="$S/pip" MPLCONFIGDIR="$S/mpl"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT="$S/vllm" VLLM_CONFIG_ROOT="$HOME/.config/vllm"
export TRITON_CACHE_DIR="$S/triton" TORCHINDUCTOR_CACHE_DIR="$S/inductor" CUDA_CACHE_PATH="$S/nv"
export FLASHINFER_WORKSPACE_BASE="$S" OUTLINES_CACHE_DIR="$S/outlines" NUMBA_CACHE_DIR="$S/numba"
export PATH="$D/env/bin:$PATH"
# light on a shared 48-core machine: a handful of CPU threads per process
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 RAYON_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
umask 077
'''


def env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit("set %s first (see the docstring)" % name)
    return v


def ssh_opts():
    """Shared connection: the first call opens it, later calls reuse it for 10 minutes."""
    sock = os.path.join(os.path.expanduser("~"), ".ssh", "fbvlm-%C")
    return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=90", "-o", "ServerAliveInterval=30",
            "-o", "ControlMaster=auto", "-o", "ControlPersist=10m", "-o", "ControlPath=" + sock]


SUB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")


def sub():
    """FB_WS_SUB (a relative subfolder of the folder for data/, code/ and runs/), or "" for the folder itself."""
    s = os.environ.get("FB_WS_SUB", "")
    if s and not SUB_RE.match(s):
        sys.exit("FB_WS_SUB must be a relative path of letters, digits, '.', '_' and '-' (no '..')")
    return s


def remote_prefix():
    # FB_WS_DIR goes in double quotes so that $USER expands on the workstation, not here. $W: data/, code/ and
    # runs/ of this package (the folder itself, or FB_WS_SUB inside it).
    d = env("FB_WS_DIR")
    if '"' in d or "`" in d:
        sys.exit("FB_WS_DIR must not contain quotes")
    return ('D="%s"; [ -f "$D/.fb_vlm_marker" ] || { echo "folder missing; run setup" >&2; exit 9; }; W="$D%s"; '
            % (d, "/" + sub() if sub() else ""))


def names_only():
    """True for a names package (vlm_prepare.py --names-only)."""
    try:
        with open(os.path.join(PKG, "items.json")) as f:
            return bool(json.load(f).get("names_only"))
    except (OSError, ValueError):
        return False


def ssh(cmd, env_sh=True, check=True, capture=False, timeout=None, quiet=False, prefix=True):
    full = (remote_prefix() if prefix else "") + ('. "$D/env/env.sh"; ' if env_sh else "") + cmd
    r = subprocess.run(["ssh"] + ssh_opts() + [env("FB_WORKSTATION"), full], capture_output=capture, text=True,
                       timeout=timeout)
    if check and r.returncode != 0:
        if capture and not quiet:
            sys.stderr.write(r.stdout + r.stderr)
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
    """The folder's absolute path, read from the workstation (kept in memory only)."""
    return ssh_out('echo "$D"', env_sh=False).strip()


def remote_work():
    """Absolute path of $W (see remote_prefix)."""
    return ssh_out('echo "$W"', env_sh=False).strip()


# ----------------------------------------------------------------------------- setup, sync, weights

def cmd_status(args):
    print(ssh_out('echo "logged-in sessions: $(who | wc -l)"; uptime | sed "s/.*load/load/"; '
                  'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader; '
                  'me=$(id -un); for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do '
                  '[ "$(ps -o user= -p $p)" = "$me" ] && echo "compute process $p: ours" || echo "compute process $p: '
                  'other account"; done; du -sh "$D" | cut -f1 | sed "s/^/folder size: /"; '
                  'df -h "$D" | tail -1 | awk \'{print "disk free: "$4}\'; [ -f "$D/runs/STATUS" ] && cat "$D/runs/STATUS"',
                  env_sh=False, check=False))


def cmd_setup(args):
    d = env("FB_WS_DIR")
    subprocess.run(["ssh"] + ssh_opts() + [env("FB_WORKSTATION"),
                    'D="%s"; mkdir -p "$D" && chmod 700 "$D" && '
                    'echo "FindBack VLM naming experiment; remove this whole folder to clean up" > "$D/.fb_vlm_marker" && '
                    'cd "$D" && mkdir -p %s && chmod -R go-rwx "$D"' % (d, " ".join(LAYOUT))], check=True)
    ssh("cat > \"$D/env/env.sh\" <<'EOF'\n%sEOF\nchmod 600 \"$D/env/env.sh\"" % ENV_SH, env_sh=False)
    ssh('if [ ! -x "$D/env/bin/python" ]; then "$(command -v python3.12 || echo /usr/bin/python3)" -m venv "$D/env"; fi; '
        'python -m pip install -q --no-cache-dir --upgrade pip && python -m pip install -q --no-cache-dir %s && '
        'python -c "import vllm, torch, pynvml; print(\'vllm\', vllm.__version__, \'torch\', torch.__version__, '
        '\'cuda\', torch.version.cuda)"; du -sh "$D/env" | cut -f1 | sed "s/^/environment: /"'
        % " ".join(shlex.quote(p) for p in PIP_PACKAGES), timeout=7200)


def cmd_sync(args):
    if not os.path.exists(os.path.join(PKG, "items.json")):
        sys.exit("no package; run vlm_prepare.py first")
    if not names_only():                    # a names package has no judges, so no judging pool inputs
        r = subprocess.run([sys.executable, os.path.join(HERE, "vlm_report.py"), "pool-inputs"], capture_output=True,
                           text=True)
        if r.returncode != 0:
            sys.exit("pool inputs: " + r.stderr[-500:])
    # the images go only into the private folder: refuse unless it is mode 700 and owned by this account
    mode = ssh_out('stat -c "%a %U" "$D"; id -un', env_sh=False).split()
    if len(mode) != 3 or mode[0] != "700" or mode[1] != mode[2]:
        sys.exit("the folder on the workstation is not private (mode 700, ours); nothing copied")
    if sub():
        # the subfolder is private from the first moment, inside the private folder
        ssh('umask 077; mkdir -p -m 700 "$W" && cd "$W" && mkdir -p data code runs && chmod 700 "$W" data code runs',
            env_sh=False)
    w = remote_work()
    host = env("FB_WORKSTATION")
    # the judge pool may have been built on the workstation: never delete it from here
    rsync(PKG + "/", "%s:%s/data/" % (host, w), delete=True, extra=["--exclude", "judge_pool.json"])
    for f in ("vlm_remote.py", "vlm_common.py"):
        rsync(os.path.join(HERE, f), "%s:%s/code/" % (host, w))
    vc.private_models()                     # the remote code reads the models' identifiers from this file
    rsync(vc.models_path(), "%s:%s/code/%s" % (host, w, vc.MODELS_FILE))
    print(ssh_out('chmod -R go-rwx "$W/data" "$W/code"; echo "data: $(find "$W/data" -type f | wc -l) files, '
                  '$(du -sh "$W/data" | cut -f1); modes: folder $(stat -c %a "$D"), work $(stat -c %a "$W"), '
                  'model file $(stat -c %a "$W/code/' + vc.MODELS_FILE + '")"', env_sh=False))


def folder_gb():
    return float(ssh_out('du -sk "$D" | cut -f1', env_sh=False).strip()) / 1e6


def downloaded(model):
    return ssh('cd "$D/code" && python -c "import vlm_remote as v, sys; sys.exit(0 if v.snapshot_path(\'%s\') else 1)"'
               % model, check=False, capture=True).returncode == 0


def cmd_fetch(args):
    if downloaded(args.model):
        print("%s already downloaded" % args.model)
        return
    used = folder_gb()
    need = vc.private_models()[args.model]["gb"]
    if used + need > DISK_BUDGET_GB:
        sys.exit("disk budget: folder %.1f GB + %.1f GB > %d GB; remove weights first" % (used, need, DISK_BUDGET_GB))
    print("folder %.1f GB, downloading about %.1f GB for %s" % (used, need, args.model))
    print(ssh_out('cd "$D/code" && python vlm_remote.py fetch %s' % args.model, timeout=7200).strip())


def cmd_rm_weights(args):
    repo = vc.private_models()[args.model]["repo"]
    ssh('rm -rf "$D/models/hub/models--%s"; du -sh "$D" | cut -f1 | sed "s/^/folder size: /"' % repo.replace("/", "--"),
        env_sh=False)


def cmd_du(args):
    print(ssh_out('du -sh "$D"/* 2>/dev/null | sed "s|$D/||"; du -sh "$D" | cut -f1 | sed "s/^/total: /"; '
                  'df -h "$D" | tail -1 | awk \'{print "disk free: "$4}\'', env_sh=False))


# ----------------------------------------------------------------------------- the detached queue

def queue_flags(args):
    """The queue's options as they go on its command line (aliases and a time only)."""
    flags = ["--shared"] if args.shared else []
    if args.models:
        flags += ["--models"] + list(args.models)
    if args.until:
        try:
            datetime.datetime.fromisoformat(args.until)
        except ValueError:
            sys.exit("--until: not a date and time like 2026-11-23T23:59: %r" % args.until)
        flags += ["--until", args.until]
    return " ".join(shlex.quote(f) for f in flags)


def cmd_queue_start(args):
    if sub() or names_only():
        sys.exit("the queue runs the experiment in the folder itself: unset FB_WS_SUB and FB_VLM_OUT")
    flags = queue_flags(args)
    cmd_sync(args)
    out = ssh_out('cd "$D/code"; if flock -n "$D/runs/queue.lock" true; then '
                  'nohup setsid python vlm_remote.py queue %s >> "$D/runs/queue.log" 2>&1 < /dev/null & sleep 8; '
                  'else echo "a queue is already running"; fi; cat "$D/runs/STATUS" 2>/dev/null' % flags, check=False)
    print(out.strip())


def cmd_queue_status(args):
    print(ssh_out('cat "$D/runs/STATUS" 2>/dev/null || echo "no STATUS file"; echo; tail -n %d "$D/runs/queue.log" '
                  '2>/dev/null | sed "s|$D|<WS_DIR>|g"' % args.lines, env_sh=False, check=False).strip())


def cmd_queue_stop(args):
    """Ask the queue to stop: it stops its running session after the item in flight, then exits."""
    ssh('touch "$D/runs/STOP"', env_sh=False)
    for _ in range(40):
        r = ssh('flock -n "$D/runs/queue.lock" true', env_sh=False, check=False)
        if r.returncode == 0:
            print("queue stopped")
            cmd_procs(args)
            return
        time.sleep(15)
    print("the queue has not stopped yet; its pid is in runs/queue.lock (send it SIGTERM if it must stop now)")


# ----------------------------------------------------------------------------- sessions driven from the Mac

def gpu_free_count():
    out = ssh_out('n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); '
                  'u=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u | wc -l); '
                  'echo $((n - u))', env_sh=False, check=False)
    try:
        return int(out.strip())
    except ValueError:
        return 0


def wait_free(need, polls, wait_max):
    """Wait until `need` GPUs have no process of anyone at `polls` polls in a row (5 minutes apart)."""
    t0, run = time.time(), 0
    while True:
        free = gpu_free_count()
        run = run + 1 if free >= need else 0
        print("  %s: %d GPU(s) free, %d free poll(s) in a row" % (time.strftime("%H:%M"), free, run), flush=True)
        if run >= polls:
            return True
        if time.time() - t0 + POLL_S > wait_max:
            return False
        time.sleep(POLL_S)


def check_free_or_exit(model, args):
    if args.no_wait:
        return
    if not wait_free(vc.PUBLIC_MODELS[model]["gpus"], FREE_POLLS, args.wait_max):
        sys.exit("the GPUs did not stay free; nothing was started")


def start_remote(kind, model, run, flags=""):
    # FB_WORK: the session reads data/ and writes runs/ of $W (vlm_remote.py)
    ssh('X="$W/runs/%s"; [ -d "$X" ] && mv "$X" "${X}_attempt_$(date +%%H%%M%%S)"; mkdir -p "$X" && cd "$W/code" && '
        'export FB_WORK="$W" && '
        # The braces matter: with a bare "a && b &" the shell keeps a waiting copy of itself that holds this
        # SSH call open until the session ends, so the caller never reaches the round trip.
        '{ nohup setsid python vlm_remote.py %s %s --run "$X" %s > "$X/session.out" 2>&1 < /dev/null & }'
        % (run, kind, model, flags))


def remote_status(run):
    r = ssh('cat "$W/runs/%s/status.json" 2>/dev/null' % run, capture=True, check=False, env_sh=False)
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {}


TERMINAL = ("done", "error", "failed", "busy", "yielded", "nofit")


def wait_state(run, states, timeout, every=30):
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        st = remote_status(run)
        notes = st.get("notes") or []
        if notes and notes[-1] != last:
            last = notes[-1]
            print("  [ws]", last, flush=True)
        if st.get("state") in states:
            return st
        time.sleep(every)
    return remote_status(run)


def open_tunnel(remote_port):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    lp = s.getsockname()[1]
    s.close()
    p = subprocess.Popen(["ssh", "-N"] + ssh_opts() + ["-o", "ExitOnForwardFailure=yes",
                                                       "-L", "127.0.0.1:%d:127.0.0.1:%d" % (lp, remote_port),
                                                       env("FB_WORKSTATION")])
    for _ in range(240):
        try:
            socket.create_connection(("127.0.0.1", lp), timeout=1).close()
            return p, lp
        except OSError:
            time.sleep(0.5)
    p.terminate()
    raise SystemExit("tunnel did not open")


def roundtrip(model, remote_port, out_dir):
    """Pings and every R and S item (crop-full) from the Mac through an SSH tunnel, one at a time."""
    import base64
    with open(os.path.join(PKG, "items.json")) as f:
        man = json.load(f)
    items = [it for it in man["items"] if it["set"] in ("R", "S")]
    tun, lp = open_tunnel(remote_port)
    rows = []
    try:
        for _ in range(20):
            r = vc.post_stream("127.0.0.1", lp, "/v1/chat/completions", vc.ping_body(model))
            rows.append({"phase": "ping", "latency_s": r["t_last"] - r["t_send"], "status": r.get("status"),
                         "error": r.get("error")})
        for it in items:
            t0 = time.perf_counter()
            encs = []
            for fname in it["variants"]["crop-full"]:
                with open(os.path.join(PKG, "images", fname), "rb") as f:
                    encs.append(base64.b64encode(f.read()).decode("ascii"))
            body = vc.chat_body(model, vc.NAME_PROMPT_CROP, encs, vc.NAME_SCHEMA, vc.NAME_MAX_TOKENS)
            prep = time.perf_counter() - t0
            r = vc.post_stream("127.0.0.1", lp, "/v1/chat/completions", body)
            ans = vc.parse_answer(r.get("text"))
            rows.append({"phase": "item", "item": it["id"], "prep_s": prep, "latency_s": r["t_last"] - r["t_send"],
                         "ttft_s": (r["t_first"] - r["t_send"]) if r.get("t_first") else None,
                         "request_bytes": r["request_bytes"], "response_bytes": r["response_bytes"],
                         "usage": r.get("usage"), "name": vc.answer_name(ans), "status": r.get("status"),
                         "error": r.get("error")})
    finally:
        tun.terminate()
        tun.wait(timeout=10)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "roundtrip_mac.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print("round trip: 20 pings, %d items, %d errors" % (
        sum(1 for r in rows if r["phase"] == "item"), sum(1 for r in rows if r["error"])))


def mac_session(model, kind, suffix, flags):
    """One session started from the Mac; when it holds, run the round trip, then release and pull."""
    cmd_sync(None)
    if names_only() and kind == "session" and "--no-hold" not in flags:
        flags = (flags + " --no-hold").strip()           # a names run needs no round trip from the Mac
    run = "%s_%s%s" % (time.strftime("%Y-%m-%d"), model, suffix)
    start_remote(kind, model, run, flags)
    st = wait_state(run, ("hold",) + TERMINAL, timeout=4 * 3600)
    if st.get("state") == "hold":
        try:
            roundtrip(model, st["port"], os.path.join(MIRROR, run))
        finally:
            ssh('touch "$W/runs/%s/release"' % run, env_sh=False)
        st = wait_state(run, TERMINAL, timeout=900, every=10)
    state = st.get("state") or "unknown"
    if state != "done":
        ssh('mv "$W/runs/%s" "$W/runs/%s_%s"' % (run, run, "contaminated" if state in ("yielded", "busy") else state),
            env_sh=False, check=False)
        local = os.path.join(MIRROR, run)
        if os.path.isdir(local):
            os.rename(local, local + "_" + ("contaminated" if state in ("yielded", "busy") else state))
    pull()
    print("remote state:", state)
    cmd_procs(None)
    return state


def cmd_run(args):
    check_free_or_exit(args.model, args)
    flags = ("--extra " + " ".join(shlex.quote(x) for x in args.extra)) if args.extra else ""
    mac_session(args.model, "smoke" if args.smoke else "session", "_smoke" if args.smoke else "", flags)


def cmd_roundtrip(args):
    if args.shared:
        # Shared mode: the workstation side does the idle check, the memory cap and the watchdog, and
        # refuses models that are not allowed to share a GPU.
        mac_session(args.model, "session", "_roundtrip", "--only-roundtrip --shared")
        return
    check_free_or_exit(args.model, args)
    mac_session(args.model, "session", "_roundtrip", "--only-roundtrip")


def cmd_judge(args):
    check_free_or_exit(args.model, args)
    mac_session(args.model, "judge", "", "--no-hold")


# ----------------------------------------------------------------------------- results back

def sanitize_tree(root, replacements):
    pats = [(re.compile(re.escape(a)), b) for a, b in replacements if a]
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if not fn.endswith((".json", ".jsonl", ".log", ".out", ".csv", ".txt")) and fn != "STATUS":
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


def pull():
    """Mirror the workstation's runs/ into analysis/out/vlm/ws_runs (never deletes local files)."""
    d, w = remote_dir(), remote_work()
    info = ssh_out('echo "$HOME"; id -un; hostname -s; hostname', env_sh=False).split()
    host = env("FB_WORKSTATION")
    os.makedirs(MIRROR, exist_ok=True)
    rsync("%s:%s/runs/" % (host, w), MIRROR + "/", extra=["--exclude", "queue.lock", "--exclude", "release"])
    repl = [(d, "<WS_DIR>"), (info[0] if info else "", "<WS_HOME>")] + [(x, "<WS>") for x in info[1:]]
    repl.append((host, "<WS>"))
    sanitize_tree(MIRROR, sorted(repl, key=lambda x: -len(x[0])))
    print("pulled to", os.path.relpath(MIRROR, os.path.join(HERE, "..")))


def cmd_pull(args):
    pull()


def cmd_procs(args):
    out = ssh_out('ps -u "$(id -un)" -o pid=,etime=,args= | grep -e "$D" -e vlm_remote.py | grep -v -e grep -e "sed s|" -e "ps -u" | '
                  'cut -c1-160 | sed "s|$D|<WS_DIR>|g" || true; echo "GPU compute processes:"; '
                  'me=$(id -un); for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do '
                  '[ "$(ps -o user= -p $p)" = "$me" ] && echo "  pid $p: ours" || echo "  pid $p: other account"; done; true',
                  env_sh=False)
    print(out.strip() or "nothing running")


def cmd_cleanup(args):
    if not args.yes:
        sys.exit("this deletes the whole folder on the workstation; add --yes")
    left = ssh_out('ps -u "$(id -un)" -o pid=,args= | grep -e "$D" -e vlm_remote.py | grep -v -e grep -e "ps -u" || true',
                   env_sh=False).strip()
    if left:
        sys.exit("processes of the experiment are still running; stop them first (queue-stop):\n" + left)
    ssh('rm -rf "$D" && echo removed', env_sh=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "setup", "sync", "du", "procs", "pull", "queue-stop"):
        sub.add_parser(name)
    p = sub.add_parser("queue-start")
    p.add_argument("--shared", action="store_true", help="allowed models first share one GPU with another account's "
                   "idle process (memory cap, watchdog); the rest wait for a free GPU")
    p.add_argument("--models", nargs="+", choices=vc.CONTESTANTS, help="only these contestants (additional runs): "
                   "no judge and no judging pool on the workstation")
    p.add_argument("--until", help="the queue gives up at this workstation time (YYYY-MM-DDTHH:MM) instead of "
                   "after 24 h")
    p = sub.add_parser("queue-status")
    p.add_argument("--lines", type=int, default=8)
    for name in ("fetch", "rm-weights"):
        p = sub.add_parser(name)
        p.add_argument("model", choices=MODEL_KEYS)
    for name in ("run", "roundtrip", "judge"):
        p = sub.add_parser(name)
        p.add_argument("model", choices=MODEL_KEYS if name != "judge" else vc.JUDGES)
        p.add_argument("--no-wait", action="store_true", help="skip the 10-minute free-GPU wait (a GPU is still "
                       "checked at start)")
        p.add_argument("--wait-max", type=float, default=6 * 3600)
        if name == "roundtrip":
            p.add_argument("--shared", action="store_true", help="share one GPU with another account's idle "
                           "process (allowed models only; memory cap and watchdog on the workstation)")
        if name == "run":
            p.add_argument("--smoke", action="store_true", help="a few requests on a synthetic image only")
            p.add_argument("--extra", nargs="*", default=[], help="extra vLLM arguments (listed contingencies only)")
    p = sub.add_parser("cleanup")
    p.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    fn = {"status": cmd_status, "setup": cmd_setup, "sync": cmd_sync, "fetch": cmd_fetch, "run": cmd_run,
          "roundtrip": cmd_roundtrip, "judge": cmd_judge, "pull": cmd_pull, "rm-weights": cmd_rm_weights,
          "du": cmd_du, "procs": cmd_procs, "cleanup": cmd_cleanup, "queue-start": cmd_queue_start,
          "queue-status": cmd_queue_status, "queue-stop": cmd_queue_stop}
    sys.exit(fn[args.cmd](args) or 0)


if __name__ == "__main__":
    main()
