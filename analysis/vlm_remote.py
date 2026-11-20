#!/usr/bin/env python3
"""Runs on the GPU machine (the lab workstation, started by vlm_ws.py, or a Slurm job on a university
cluster, started by vlm_cluster.py), inside the experiment's private folder.

    python vlm_remote.py gpus                       # which GPUs are idle (JSON)
    python vlm_remote.py fetch MODEL                # download pinned weights into $HF_HOME
    python vlm_remote.py session MODEL --run DIR    # serve MODEL, run every naming phase, stop
    python vlm_remote.py judge JUDGE --run DIR      # serve a judge, judge the pool, stop
    python vlm_remote.py queue                      # detached: wait for free GPUs, run every remaining session
    python vlm_remote.py queue --models M ... [--until 2026-11-23T23:59]
                                                    # only these contestants (additional runs: no judge, no
                                                    #   judging pool); give up at that local time, not after 24 h
    python vlm_remote.py pool --out FILE --run DIR ... [--names-file FILE]   # the blind judging pool
    python vlm_remote.py job MODEL --name NAME      # inside a Slurm job: one session or judging session on
                                                    #   the allocated GPU, with the listed contingencies

MODEL is an alias (model_a, model_b, model_c, model_d, judge_a, judge_b); a former
key from the private model file is accepted too. The models' identifiers are read from
vlm_models.private.json next to this file (or $FB_VLM_MODELS).

A session starts one vLLM server bound to 127.0.0.1 on idle GPUs only, samples GPU power, energy
and memory every 100 ms, runs the phases of analysis/notes/2026-11-17_vlm_naming_design.md, then
waits (at most --hold-max seconds) for the Mac's round-trip test and stops the server in a finally
block. A hard limit (--max-seconds) stops everything even if the Mac never answers. Nothing keeps
running after a session ends.

Paths: the package is in $FBV/data, results go to --run (inside $FBV/runs). With FB_WORK set (a
subfolder of $FBV, written by vlm_cluster.py when FB_CLUSTER_SUB is set), data/ and runs/ are those of
$FB_WORK; weights, environment and caches stay in $FBV.

A package whose items.json has "names_only": true (vlm_prepare.py --names-only) gets a session of the
sequential pass with crop-full, repeat 1 only (no other input, repeat, cross-view or concurrency phase).
"""
import argparse
import base64
import concurrent.futures as cf
import datetime
import fcntl
import glob
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlm_common as vc  # noqa: E402

FBV = os.environ.get("FBV") or os.path.dirname(HERE)
WORK = os.environ.get("FB_WORK") or FBV            # data/ and runs/ (see the docstring)
DATA = os.path.join(WORK, "data")

def model(key):
    """Serving facts of a model (alias or former key): repo, rev, ignore and limit from the private model
    file, GPUs from the design."""
    key = vc.canonical_key(key)
    p = vc.private_models()[key]
    return {"key": key, "repo": p["repo"], "rev": p["rev"], "gpus": vc.PUBLIC_MODELS[key]["gpus"],
            "limit": p["limit"], "ignore": p.get("ignore"), "gb": p["gb"]}


def model_choices():
    """Aliases, and the former keys of the private file (a queue started with older code passes those)."""
    old = [m.get("old_key") for m in vc.private_models(required=False).values() if m.get("old_key")]
    return sorted(vc.PUBLIC_MODELS) + sorted(old)


SERVE_ARGS = ["--max-model-len", "8192", "--max-num-seqs", "16", "--gpu-memory-utilization", "0.90",
              "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0", "--seed", "0",
              "--generation-config", "vllm", "--disable-uvicorn-access-log"]
IDLE_UTIL = 5            # percent
IDLE_MEM_MIB = 1024
WARMUP = 3
IDLE_SECONDS = 15
REPEATS = 3
CONCURRENCY = (4, 16)
PINGS = 20


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- GPUs

def nvml():
    import pynvml
    pynvml.nvmlInit()
    return pynvml


def gpu_state():
    """Per GPU: index, memory used (MiB), utilisation (%), compute processes (pid, MiB), idle flag."""
    pn = nvml()
    out = []
    for i in range(pn.nvmlDeviceGetCount()):
        h = pn.nvmlDeviceGetHandleByIndex(i)
        mem = pn.nvmlDeviceGetMemoryInfo(h)
        util = pn.nvmlDeviceGetUtilizationRates(h).gpu
        procs = [{"pid": p.pid, "mib": (p.usedGpuMemory or 0) // 2**20}
                 for p in pn.nvmlDeviceGetComputeRunningProcesses(h)]
        used = mem.used // 2**20
        try:
            name = pn.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
        except Exception:  # noqa: BLE001
            name = None
        out.append({"index": i, "name": name, "mem_used_mib": used, "mem_total_mib": mem.total // 2**20,
                    "util": util, "compute_procs": len(procs),
                    "idle": not procs and util < IDLE_UTIL and used < IDLE_MEM_MIB})
    return out


def foreign_gpu_processes():
    """Compute processes on any GPU that belong to another account."""
    pn = nvml()
    me = os.getuid()
    out = []
    for i in range(pn.nvmlDeviceGetCount()):
        for p in pn.nvmlDeviceGetComputeRunningProcesses(pn.nvmlDeviceGetHandleByIndex(i)):
            try:
                uid = os.stat("/proc/%d" % p.pid).st_uid
            except OSError:
                uid = -1
            if uid != me:
                out.append({"gpu": i, "pid": p.pid})
    return out


class Yield(Exception):
    """Another user started a GPU process: stop and leave the machine to them."""


class Watchdog(threading.Thread):
    """Watches every GPU for a compute process of another account (every `period` seconds).

    When one appears it records it; the phases check `check_yield()` after each item, so the item
    in flight finishes and the session then stops its server. If no item boundary comes within
    `backstop` seconds (for example during server start-up), the main thread is interrupted."""

    def __init__(self, period=2.0, backstop=15.0):
        super().__init__(daemon=True)
        self.period = period
        self.backstop = backstop
        self.stop_ev = threading.Event()
        self.found = None
        self.found_at = None

    def run(self):
        while not self.stop_ev.wait(self.period):
            try:
                f = foreign_gpu_processes()
            except Exception:  # noqa: BLE001
                continue
            if f:
                self.found, self.found_at = f, time.time()
                break
        if self.found and not self.stop_ev.wait(self.backstop):
            signal.pthread_kill(threading.main_thread().ident, signal.SIGUSR1)


# ----------------------------------------------------------------------------- shared mode
# Decided 2026-11-17 with the user: a model whose measured footprint fits may share ONE GPU with another
# account's idle process, leaving at least SHARED_FLOOR_TARGET_MIB unreserved; a watchdog stops our
# server at the first sign that the other program is active again.

SHARED = {
    # model: settings that bound the server's memory (the inputs are unchanged: every image is <= 960 px)
    #   est_peak_mib: our process at its peak, estimated from the 4B smoke run (9297 MiB after weights and
    #   context) + the fixed KV cache + activations profiled with 960-px images
    "model_a": {"kv_bytes": 1 << 30, "batched": 4096, "est_peak_mib": 11300},
}
SHARED_FLOOR_TARGET_MIB = 5120      # unreserved memory left on the GPU with our model at its planned peak
SHARED_FLOOR_STOP_MIB = 4864        # the watchdog stops our server below this
SHARED_IDLE_CHECK_S = 120           # the other process must look idle this long right before a start
SHARED_REIDLE_S = 600               # after a watchdog stop: idle this long before another shared attempt
SHARED_PRELOAD_IDLE_S = 15          # GPU power right before loading (the energy baseline)


def other_account_procs():
    """{(gpu, pid): MiB} of compute processes of other accounts, and {gpu: free MiB}."""
    pn = nvml()
    me = os.getuid()
    procs, free = {}, {}
    for i in range(pn.nvmlDeviceGetCount()):
        h = pn.nvmlDeviceGetHandleByIndex(i)
        free[i] = pn.nvmlDeviceGetMemoryInfo(h).free // 2**20
        for p in pn.nvmlDeviceGetComputeRunningProcesses(h):
            try:
                mine = os.stat("/proc/%d" % p.pid).st_uid == me
            except OSError:
                mine = False
            if not mine:
                procs[(i, p.pid)] = (p.usedGpuMemory or 0) // 2**20
    return procs, free


def other_activity(pids, since_us):
    """Largest SM utilisation of the given pids in NVML's per-process samples since `since_us` on any GPU
    (0 when NVML has no sample: the process did no work)."""
    pn = nvml()
    top = 0
    for i in range(pn.nvmlDeviceGetCount()):
        try:
            for x in pn.nvmlDeviceGetProcessUtilization(pn.nvmlDeviceGetHandleByIndex(i), since_us):
                if x.pid in pids:
                    top = max(top, x.smUtil, x.memUtil)
        except pn.NVMLError:
            pass                    # NotFound: no samples since the timestamp
    return top


def shared_signs(base, procs, free, gpu, util):
    """Reasons to stop (empty when the other program still looks idle). `base`: {(gpu, pid): MiB}."""
    out = []
    for k, mib in procs.items():
        if k not in base:
            out.append("new process of another account on GPU %d (pid %d)" % k)
        elif mib != base[k]:
            out.append("other process memory changed on GPU %d: %d -> %d MiB" % (k[0], base[k], mib))
    if util > 0:
        out.append("other process GPU utilisation %d%%" % util)
    if free.get(gpu, 0) < SHARED_FLOOR_STOP_MIB:
        out.append("unreserved memory on GPU %d %d MiB < floor %d MiB" % (gpu, free.get(gpu, 0), SHARED_FLOOR_STOP_MIB))
    return out


def idle_check(seconds, gpu=None, log_path=None, abort=None):
    """Sample once a second for `seconds`: the other accounts' processes must keep their memory, show no
    utilisation and no new process may appear. Returns (ok, baseline dict, record)."""
    procs0, free0 = other_account_procs()
    pids = {pid for _, pid in procs0}
    ts = int(time.time() * 1e6)
    rec = {"seconds": seconds, "samples": 0, "max_util": 0, "reasons": [], "other": {"%d:%d" % k: v for k, v in procs0.items()},
           "free_mib": free0}
    f = open(log_path, "w") if log_path else None
    if f:
        f.write("t,free_mib,other_mib,other_util,reason\n")
    t_end = time.time() + seconds
    while time.time() < t_end:
        time.sleep(1.0)
        if abort is not None and abort():
            rec["reasons"] = ["aborted"]
            if f:
                f.close()
            return False, procs0, rec
        procs, free = other_account_procs()
        now = int(time.time() * 1e6)
        util = other_activity(pids, ts)
        ts = now
        rec["samples"] += 1
        rec["max_util"] = max(rec["max_util"], util)
        why = shared_signs(procs0, procs, {gpu: 10**9} if gpu is None else free, gpu, util)
        if f:
            f.write("%.1f,%s,%d,%d,%s\n" % (time.time(), free.get(gpu, ""), sum(procs.values()), util, "; ".join(why)))
        if why:
            rec["reasons"] = why
            if f:
                f.close()
            return False, procs0, rec
    if f:
        f.close()
    return True, procs0, rec


class SharedWatchdog(threading.Thread):
    """Shared mode: every 0.1 s during start-up (first 5 minutes), then every second, checks the other
    accounts' processes and the free memory of our GPU; at the first sign of activity it interrupts the
    main thread at once (the server is then stopped within a few seconds)."""

    def __init__(self, gpu, base, csv_path):
        super().__init__(daemon=True)
        self.gpu, self.base = gpu, base
        self.pids = {pid for _, pid in base}
        self.stop_ev = threading.Event()
        self.found = None
        self.found_at = None
        self.samples = 0
        self.min_free = None
        self.max_util = 0
        self.f = open(csv_path, "w")
        self.f.write("t,free_mib,other_mib,other_util,reason\n")

    def run(self):
        t0 = time.time()
        ts = int(t0 * 1e6)
        blind_since = None
        while not self.stop_ev.wait(0.1 if time.time() - t0 < 300 else 1.0):
            try:
                procs, free = other_account_procs()
                now = int(time.time() * 1e6)
                util = other_activity(self.pids, ts)
                ts = now
            except Exception:  # noqa: BLE001
                # Never run blind beside another account's process: 5 s without a reading stops our server.
                blind_since = blind_since or time.time()
                if time.time() - blind_since >= 5.0:
                    self.found, self.found_at = ["watchdog could not read the GPU state for 5 s"], time.time()
                    self.f.write("%.2f,,,,%s\n" % (time.time(), self.found[0]))
                    self.f.flush()
                    signal.pthread_kill(threading.main_thread().ident, signal.SIGUSR1)
                    break
                continue
            blind_since = None
            self.samples += 1
            fr = free.get(self.gpu, 0)
            self.min_free = fr if self.min_free is None else min(self.min_free, fr)
            self.max_util = max(self.max_util, util)
            why = shared_signs(self.base, procs, free, self.gpu, util)
            self.f.write("%.2f,%d,%d,%d,%s\n" % (time.time(), fr, sum(procs.values()), util, "; ".join(why)))
            if why:
                self.found, self.found_at = why, time.time()
                self.f.flush()
                signal.pthread_kill(threading.main_thread().ident, signal.SIGUSR1)
                break
        self.f.close()

    def record(self):
        return {"samples": self.samples, "min_free_mib": self.min_free, "max_other_util": self.max_util,
                "fired": bool(self.found), "reasons": self.found or [], "fired_at": self.found_at}


WATCH = None


def check_yield():
    """Raise Yield after the current item when another account's process is on a GPU."""
    if WATCH is not None and WATCH.found:
        raise Yield("another user's process appeared on a GPU: %s" % (WATCH.found,))


class Sampler(threading.Thread):
    """GPU power (W), utilisation, memory and the energy counter (J) every `period` seconds."""

    def __init__(self, gpus, path, period=0.1):
        super().__init__(daemon=True)
        self.pn = nvml()
        self.handles = [self.pn.nvmlDeviceGetHandleByIndex(g) for g in gpus]
        self.gpus = gpus
        self.period = period
        self.f = open(path, "w")
        self.f.write("t,gpu,power_w,util,mem_mib,energy_j\n")
        self.stop_ev = threading.Event()
        self.peak_mem = {g: 0 for g in gpus}

    def energy(self):
        """Sum over the GPUs of the energy counter, joules."""
        return sum(self.pn.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles) / 1000.0

    def run(self):
        while not self.stop_ev.is_set():
            t = time.perf_counter()
            for g, h in zip(self.gpus, self.handles):
                try:
                    p = self.pn.nvmlDeviceGetPowerUsage(h) / 1000.0
                    u = self.pn.nvmlDeviceGetUtilizationRates(h).gpu
                    m = self.pn.nvmlDeviceGetMemoryInfo(h).used // 2**20
                    e = self.pn.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
                except Exception:  # noqa: BLE001
                    continue
                self.peak_mem[g] = max(self.peak_mem[g], m)
                self.f.write("%.4f,%d,%.2f,%d,%d,%.3f\n" % (t, g, p, u, m, e))
            self.stop_ev.wait(self.period)
        self.f.close()

    def stop(self):
        self.stop_ev.set()
        self.join(timeout=5)


# ----------------------------------------------------------------------------- server

def snapshot_path(key):
    m = model(key)
    d = os.path.join(os.environ["HF_HOME"], "hub", "models--" + m["repo"].replace("/", "--"), "snapshots", m["rev"])
    return d if os.path.isdir(d) else None


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve_path(key):
    """A neutral path to the pinned weights, $HF_HOME/<alias> (a symlink to the snapshot), so that no
    repository name appears on the server's command line; None when the weights are not downloaded."""
    snap = snapshot_path(key)
    if not snap:
        return None
    link = os.path.join(os.environ["HF_HOME"], vc.canonical_key(key))
    target = os.path.relpath(snap, os.path.dirname(link))
    if os.path.islink(link) and os.readlink(link) != target:
        os.remove(link)
    if not os.path.lexists(link):
        os.symlink(target, link)
    return link


class Server:
    """vLLM, started as `python vlm_remote.py serve <run>/serve_args.json`: the arguments (neutral weights
    path, served name = the alias) are read from that file, so a process listing shows no model detail."""

    def __init__(self, key, gpus, port, log_path, extra=(), limit=None):
        path = serve_path(key)
        if not path:
            raise SystemExit("weights of %s are not downloaded" % key)
        m = model(key)
        argv = ["serve", path, "--served-model-name", key, "--host", "127.0.0.1", "--port", str(port),
                "--limit-mm-per-prompt", json.dumps(limit or m["limit"])]
        argv += SERVE_ARGS + (["--tensor-parallel-size", str(len(gpus))] if len(gpus) > 1 else []) + list(extra)
        args_file = os.path.join(os.path.dirname(os.path.abspath(log_path)), "serve_args.json")
        write_json(args_file, argv)
        cmd = [sys.executable, os.path.abspath(__file__), "serve", args_file]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus))
        # NVRTC (run-time kernel compilation, used for example by 4-bit kernels) looks up its builtins library
        # on the library path: add the environment's own CUDA library folder (the files torch already loads)
        cuda_lib = os.path.join(sys.prefix, "lib", "python%d.%d" % sys.version_info[:2], "site-packages", "nvidia",
                                "cu13", "lib")
        if os.path.isdir(cuda_lib):
            env["LD_LIBRARY_PATH"] = cuda_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        self.cmd, self.port, self.key = ["vllm"] + argv, port, key
        self.logf = open(log_path, "w")
        self.t_launch = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        log("server pid %d on port %d, GPUs %s" % (self.proc.pid, port, gpus))

    def wait_ready(self, timeout=1800):
        while time.perf_counter() - self.t_launch < timeout:
            if self.proc.poll() is not None:
                return None
            check_yield()
            st, _ = vc.http_get("127.0.0.1", self.port, "/health", timeout=5)
            if st == 200:
                return time.perf_counter() - self.t_launch
            time.sleep(1.0)
        return None

    def metrics(self):
        st, txt = vc.http_get("127.0.0.1", self.port, "/metrics", timeout=10)
        return vc.parse_prometheus(txt) if st == 200 else {}

    def stop(self, fast=False):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=5 if fast else 90)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=30)
        # make sure no child of the group survives
        try:
            os.killpg(self.proc.pid, 0)
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.logf.close()
        log("server stopped (exit %s)" % self.proc.returncode)


# ----------------------------------------------------------------------------- requests

def load_manifest():
    with open(os.path.join(DATA, "items.json")) as f:
        return json.load(f)


_B64 = {}


def b64(fname):
    if fname not in _B64:
        with open(os.path.join(DATA, "images", fname), "rb") as f:
            raw = f.read()
        t0 = time.perf_counter()
        enc = base64.b64encode(raw).decode("ascii")
        _B64[fname] = (enc, len(raw), time.perf_counter() - t0)
    return _B64[fname]


def name_request(model, item, variant):
    files = item["variants"][variant]
    encs = [b64(f) for f in files]
    text = vc.NAME_PROMPT_CTX if variant.startswith("ctx") else vc.NAME_PROMPT_CROP
    body = vc.chat_body(model, text, [e[0] for e in encs], vc.NAME_SCHEMA, vc.NAME_MAX_TOKENS)
    return body, {"jpeg_bytes": sum(e[1] for e in encs), "b64_s": sum(e[2] for e in encs), "n_images": len(files)}


def timed(server, sampler, body, measure_metrics=True):
    m0 = server.metrics() if measure_metrics else {}
    e0 = sampler.energy()
    t_wall0 = time.time()
    r = vc.post_stream("127.0.0.1", server.port, "/v1/chat/completions", body)
    e1 = sampler.energy()
    m1 = server.metrics() if measure_metrics else {}
    r["energy_j"] = e1 - e0
    r["wall_start"] = t_wall0
    r["vllm"] = vc.metric_deltas(m0, m1) if measure_metrics else {}
    return r


def row_from(r, **kw):
    u = r.get("usage") or {}
    lat = r["t_last"] - r["t_send"]
    ttft = (r["t_first"] - r["t_send"]) if r.get("t_first") else None
    dec = (r["t_last"] - r["t_first"]) if r.get("t_first") else None
    ans = vc.parse_answer(r.get("text"))
    out = dict(kw)
    out.update({"latency_s": lat, "ttft_s": ttft, "decode_s": dec, "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"), "finish": r.get("finish_reason"),
                "energy_j": r.get("energy_j"), "request_bytes": r.get("request_bytes"),
                "response_bytes": r.get("response_bytes"), "status": r.get("status"), "error": r.get("error"),
                "text": r.get("text"), "answer": ans, "name": vc.answer_name(ans), "vllm": r.get("vllm"),
                "t_send": r.get("t_send"), "t_last": r.get("t_last")})
    return out


# ----------------------------------------------------------------------------- phases

def phase_sequential(server, sampler, items, out_path, status, repeats=REPEATS, variants=None):
    order = list(items)
    random.Random(vc.SEED).shuffle(order)
    with open(out_path, "a") as f:
        for rep in range(1, repeats + 1):
            for variant in variants or vc.VARIANTS:
                todo = [it for it in order if variant in it["variants"]]
                for k, it in enumerate(todo):
                    body, meta = name_request(server.key, it, variant)
                    r = timed(server, sampler, body)
                    f.write(json.dumps(row_from(r, phase="seq", item=it["id"], set=it["set"], variant=variant,
                                                repeat=rep, concurrency=1, **meta)) + "\n")
                    f.flush()
                    check_yield()
                status("sequential repeat %d %s: %d items" % (rep, variant, len(todo)))


def phase_crossview(server, sampler, items, out_path):
    with open(out_path, "a") as f:
        for it in items:
            body, meta = name_request(server.key, it, "crop-full")
            r = timed(server, sampler, body)
            f.write(json.dumps(row_from(r, phase="crossview", item=it["id"], set="V", variant="crop-full", repeat=1,
                                        concurrency=1, **meta)) + "\n")
            check_yield()


def phase_concurrency(server, sampler, items, out_path, levels=CONCURRENCY, passes=2):
    work = [it for _ in range(passes) for it in items]
    summary = []
    with open(out_path, "a") as f:
        for c in levels:
            e0, t0 = sampler.energy(), time.perf_counter()

            def one(it):
                check_yield()
                body, meta = name_request(server.key, it, "crop-full")
                r = vc.post_stream("127.0.0.1", server.port, "/v1/chat/completions", body)
                return it, meta, r

            ex = cf.ThreadPoolExecutor(max_workers=c)
            try:
                results = list(ex.map(one, work))
            finally:
                ex.shutdown(wait=False, cancel_futures=True)   # an interruption does not wait for the queue
            t1, e1 = time.perf_counter(), sampler.energy()
            for it, meta, r in results:
                f.write(json.dumps(row_from(r, phase="concurrency", item=it["id"], set=it["set"], variant="crop-full",
                                            repeat=0, concurrency=c, **meta)) + "\n")
            summary.append({"concurrency": c, "requests": len(work), "wall_s": t1 - t0, "energy_j": e1 - e0,
                            "errors": sum(1 for _, _, r in results if r.get("error"))})
            log("concurrency %d: %d requests in %.1f s" % (c, len(work), t1 - t0))
    return summary


def phase_pings(server, n=PINGS):
    out = []
    for _ in range(n):
        check_yield()
        r = vc.post_stream("127.0.0.1", server.port, "/v1/chat/completions", vc.ping_body(server.key))
        out.append({"latency_s": r["t_last"] - r["t_send"], "error": r.get("error"), "status": r.get("status")})
    return out


# ----------------------------------------------------------------------------- commands

def cmd_gpus(args):
    print(json.dumps(gpu_state()))


def cmd_fetch(args):
    from huggingface_hub import snapshot_download
    m = model(args.model)
    t0 = time.time()
    p = snapshot_download(m["repo"], revision=m["rev"], ignore_patterns=m.get("ignore"),
                          max_workers=int(os.environ.get("FB_FETCH_WORKERS", "2")))
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p, followlinks=True) for f in fs)
    rec = {"model": args.model, "seconds": time.time() - t0, "bytes": size}
    # the snapshot folder appears while files are still arriving; this marker says the download ended
    write_json(os.path.join(os.environ["HF_HOME"], "%s.fetched" % vc.canonical_key(args.model)), rec)
    print(json.dumps(rec))


def run_server_session(args, body_fn):
    """Common frame: idle check, sampler, server start, body_fn(server, sampler, status), stop."""
    os.makedirs(args.run, exist_ok=True)
    stat_path = os.path.join(args.run, "status.json")
    args.model = vc.canonical_key(args.model)
    # site: "workstation" or "cluster" (set by the launcher); a Slurm job id when there is one
    state = {"model": args.model, "state": "starting", "t": time.time(), "t_start": time.time(), "notes": [],
             "site": os.environ.get("FB_SITE"), "slurm_job": os.environ.get("SLURM_JOB_ID")}

    def status(msg=None, **kw):
        if msg:
            state["notes"].append("%s %s" % (time.strftime("%H:%M:%S"), msg))
            log(msg)
        state.update(kw)
        state["t"] = time.time()
        write_json(stat_path, state)

    need = model(args.model)["gpus"]
    if getattr(args, "gpus", None):
        state["gpus_override"] = {"design": need, "used": args.gpus}
        need = args.gpus
    gs = gpu_state()
    state["gpus_before"] = gs
    if os.environ.get("SLURM_JOB_ID") and not getattr(args, "shared", False):
        # In a Slurm job NVML must see exactly the GPUs Slurm allocated, so that index 0.. are ours.
        alloc = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        if len(gs) != len(alloc) or len(gs) < need:
            status("NVML sees %d GPU(s), Slurm allocated %s, %d needed; nothing started" % (
                len(gs), alloc or "none", need), state="error")
            return 4
    shared = None
    extra, limit = list(args.extra), None
    if getattr(args, "shared", False):
        if args.model not in SHARED:
            status("%s is not allowed in shared mode" % args.model, state="nofit")
            return 2
        procs, free = other_account_procs()
        ours = {g["index"] for g in gs if g["compute_procs"] > len([1 for (gg, _) in procs if gg == g["index"]])}
        cands = [i for i in free if i not in ours]
        gpu = max(cands, key=lambda i: free[i])
        allowed = free[gpu] - SHARED_FLOOR_TARGET_MIB
        plan = SHARED[args.model]
        shared = {"gpu": gpu, "free_before_mib": free[gpu], "cap_mib": allowed, "floor_target_mib": SHARED_FLOOR_TARGET_MIB,
                  "floor_stop_mib": SHARED_FLOOR_STOP_MIB, "est_peak_mib": plan["est_peak_mib"],
                  "other_before": {"%d:%d" % k: v for k, v in procs.items()}}
        state["shared"] = shared
        if plan["est_peak_mib"] > allowed:
            status("shared mode: %s needs about %d MiB, the cap on GPU %d is %d MiB; not started" % (
                args.model, plan["est_peak_mib"], gpu, allowed), state="nofit")
            return 2
        status("shared mode: checking that the other process is idle on every GPU for %d s" % SHARED_IDLE_CHECK_S)
        ok, base, rec = idle_check(SHARED_IDLE_CHECK_S, gpu, os.path.join(args.run, "idle_check.csv"))
        shared["idle_check"] = rec
        if not ok:
            status("shared mode: the other process is not idle (%s); not started" % "; ".join(rec["reasons"]),
                   state="busy")
            return 2
        shared["baseline"] = {"%d:%d" % k: v for k, v in base.items()}
        gpus = [gpu]
        frac = int(100.0 * allowed / (gs[gpu]["mem_total_mib"])) / 100.0
        extra = ["--gpu-memory-utilization", "%.2f" % frac, "--kv-cache-memory", str(plan["kv_bytes"]),
                 "--max-num-batched-tokens", str(plan["batched"])] + extra
        limit = dict(model(args.model)["limit"])
        limit["image"] = {"count": 2, "width": 960, "height": 960}
        shared["server_args"] = extra
    else:
        idle = [g["index"] for g in gs if g["idle"]]
        if len(idle) < need:
            status("not enough idle GPUs (%d needed, idle: %s); nothing started" % (need, idle), state="busy")
            return 2
        gpus = idle[:need]
    port = args.port or free_port()
    sampler = Sampler(gpus, os.path.join(args.run, "power.csv"))
    sampler.start()
    server = None
    deadline = time.time() + args.max_seconds

    def on_alarm(signum, frame):
        raise TimeoutError("session time limit reached")

    def on_yield(signum, frame):
        if shared:
            raise Yield("shared-mode watchdog: %s" % "; ".join(watchdog.found or []))
        raise Yield("another user's process appeared on a GPU (no item boundary within the backstop): %s"
                    % (watchdog.found,))

    global WATCH
    if shared:
        watchdog = WATCH = SharedWatchdog(gpus[0], base, os.path.join(args.run, "watchdog.csv"))
    else:
        watchdog = WATCH = Watchdog()
    signal.signal(signal.SIGALRM, on_alarm)
    signal.signal(signal.SIGUSR1, on_yield)
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.alarm(int(args.max_seconds))
    code = 0
    try:
        mem_before = {g["index"]: g["mem_used_mib"] for g in gpu_state() if g["index"] in gpus}
        if shared:
            t0 = time.perf_counter()
            time.sleep(SHARED_PRELOAD_IDLE_S)
            shared["preload_idle_window"] = [t0, time.perf_counter()]
        status("GPUs %s %s; launching" % (gpus, "shared with an idle process of another account" if shared else "idle"),
               state="launching", gpus=gpus, port=port, mem_before=mem_before)
        watchdog.start()
        server = Server(args.model, gpus, port, os.path.join(args.run, "server.log"), extra=extra, limit=limit)
        state["cmd"] = server.cmd
        ready = server.wait_ready()
        if ready is None:
            status("server did not start", state="failed")
            return 3
        status("ready after %.1f s" % ready, state="ready", startup_s=ready)
        state["metric_names"] = sorted(server.metrics())
        body_fn(server, sampler, status, deadline)
        status("phases done", state="finishing")
    except Yield as e:
        own = shared and watchdog.found and all(r.startswith("unreserved memory") for r in watchdog.found)
        if own:
            # the other process is unchanged and idle: our own footprint passed the floor
            status("shared mode: our server needed more than the cap (%s); stopped" % "; ".join(watchdog.found),
                   state="nofit")
        else:
            status("yielded: %s" % e, state="yielded", foreign_seen_at=watchdog.found_at)
        code = 5
    except KeyboardInterrupt as e:
        if shared and watchdog.found:
            status("yielded (shared-mode watchdog): %s" % "; ".join(watchdog.found), state="yielded")
            code = 5
        else:
            status("stopped by KeyboardInterrupt: %s" % e, state="error")
            code = 4
    except BaseException as e:  # noqa: BLE001 - always stop the server below
        status("stopped by %s: %s" % (type(e).__name__, e), state="error")
        code = 4
    finally:
        # nothing may interrupt the cleanup below
        signal.alarm(0)
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        watchdog.stop_ev.set()
        if server is not None:
            server.stop(fast=bool(shared))
        if shared:
            shared["watchdog"] = watchdog.record()
        time.sleep(3)
        sampler.stop()
        state["peak_mem_mib"] = sampler.peak_mem
        state["gpus_after"] = gpu_state()
        if state.get("state") not in ("error", "failed", "busy", "yielded", "nofit"):
            state["state"] = "done"
        status("session over")
    return code


def cmd_session(args):
    man = load_manifest()
    rs = [it for it in man["items"] if it["set"] in ("R", "S")]
    vs = [it for it in man["items"] if it["set"] == "V"]

    def body(server, sampler, status, deadline):
        out = os.path.join(args.run, "requests.jsonl")
        for it in rs[:WARMUP]:
            b, _ = name_request(server.key, it, "crop-full")
            vc.post_stream("127.0.0.1", server.port, "/v1/chat/completions", b)
        status("warm-up done")
        t0 = time.perf_counter()
        time.sleep(IDLE_SECONDS)
        status("idle window", idle_window=[t0, time.perf_counter()])
        if man.get("names_only") and not args.only_roundtrip:
            status("names package: sequential pass, %s, repeat 1 only" % vc.PRIMARY_VARIANT)
            phase_sequential(server, sampler, rs, out, status, repeats=1, variants=[vc.PRIMARY_VARIANT])
        elif not args.only_roundtrip:
            phase_sequential(server, sampler, rs, out, status)
            phase_crossview(server, sampler, vs, out)
            status("cross-view done")
            state_conc = phase_concurrency(server, sampler, rs, out)
            write_json(os.path.join(args.run, "concurrency.json"), state_conc)
        write_json(os.path.join(args.run, "pings_local.json"), phase_pings(server))
        if args.no_hold:
            return
        # hold for the Mac's round trip
        release = os.path.join(args.run, "release")
        status("holding for the round-trip test", state="hold")
        t_hold = time.time()
        while not os.path.exists(release) and time.time() - t_hold < args.hold_max and time.time() < deadline - 120:
            time.sleep(2)
            check_yield()
        status("hold over (%s)" % ("released" if os.path.exists(release) else "timeout"))

    return run_server_session(args, body)


def cmd_judge(args):
    with open(args.pool or os.path.join(DATA, "judge_pool.json")) as f:
        pool = json.load(f)

    def body(server, sampler, status, deadline):
        out = os.path.join(args.run, "judgments.jsonl")
        first = pool["items"][0]
        b = vc.chat_body(server.key, vc.judge_text(first["names"], first["with_frame"]),
                         [b64(x)[0] for x in first["images"]], vc.judge_schema(len(first["names"])),
                         vc.JUDGE_MAX_TOKENS)
        vc.post_stream("127.0.0.1", server.port, "/v1/chat/completions", b)
        t0 = time.perf_counter()
        time.sleep(IDLE_SECONDS)
        status("idle window", idle_window=[t0, time.perf_counter()])
        with open(out, "a") as f:
            for order in ("A", "B"):
                for it in pool["items"]:
                    names = it["names"] if order == "A" else list(reversed(it["names"]))
                    b = vc.chat_body(server.key, vc.judge_text(names, it["with_frame"]),
                                     [b64(x)[0] for x in it["images"]], vc.judge_schema(len(names)),
                                     vc.JUDGE_MAX_TOKENS)
                    r = timed(server, sampler, b, measure_metrics=False)
                    verdicts = vc.parse_verdicts(r.get("text"), len(names))
                    check_yield()
                    f.write(json.dumps({"item": it["item"], "order": order, "names": names,
                                        "verdicts": {names[i]: v for i, v in verdicts.items()},
                                        "latency_s": r["t_last"] - r["t_send"], "energy_j": r.get("energy_j"),
                                        "usage": r.get("usage"), "finish": r.get("finish_reason"),
                                        "error": r.get("error"), "text": r.get("text")}) + "\n")
                    f.flush()
                status("judged order %s: %d items" % (order, len(pool["items"])))

    return run_server_session(args, body)


def cmd_smoke(args):
    """A server start and a few requests on a synthetic image (no home photo)."""
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGB", (320, 240), (240, 240, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([110, 60, 210, 200], fill=(200, 30, 30))
    d.ellipse([125, 40, 195, 80], fill=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    enc = base64.b64encode(buf.getvalue()).decode()

    def body(server, sampler, status, deadline):
        res = []
        for text, schema, imgs in ((vc.NAME_PROMPT_CROP, vc.NAME_SCHEMA, [enc]),
                                   (vc.NAME_PROMPT_CTX, vc.NAME_SCHEMA, [enc, enc]),
                                   (vc.judge_text(["red cup", "blue car"], False), vc.judge_schema(2), [enc])):
            r = timed(server, sampler, vc.chat_body(server.key, text, imgs, schema, 200))
            res.append(row_from(r, phase="smoke"))
        res.append({"pings": phase_pings(server, 3)})
        write_json(os.path.join(args.run, "smoke.json"), res)

    return run_server_session(args, body)



# ----------------------------------------------------------------------------- queue (detached)

RUNS_DIR = os.path.join(WORK, "runs")
CONTESTANTS = ["model_a", "model_b", "model_d", "model_c"]     # the queue's order (design note)
JUDGES = list(vc.JUDGES)
DISK_BUDGET_GB = 80
CONTINGENCIES = [[], ["--enforce-eager"], ["--enforce-eager", "--gpu-memory-utilization", "0.95"],
                 ["--enforce-eager", "--gpu-memory-utilization", "0.95", "--max-model-len", "4096"]]
POLL_S = 300            # one NVML query every 5 minutes while waiting
FREE_POLLS = 3          # three free polls in a row: at least 10 minutes without any process
QUEUE_HOURS = 24
TWO_GPU_WAIT_S = 2 * 3600
STOP_FILE = os.path.join(RUNS_DIR, "STOP")


def run_state(d):
    try:
        with open(os.path.join(d, "status.json")) as f:
            return json.load(f).get("state")
    except (OSError, ValueError):
        return None


def latest_done(key, suffix=""):
    """Latest finished run folder of a model (by folder name), under its alias or its former key."""
    key = vc.canonical_key(key)
    cands = []
    for k in vc.run_keys(key):
        cands += glob.glob(os.path.join(RUNS_DIR, "*_%s%s" % (k, suffix)))
    for d in sorted(cands, key=os.path.basename, reverse=True):
        if run_state(d) == "done":
            return d
    return None


def seq_names(run_dir):
    """item -> every name of the run's sequential phase (all inputs and repeats), in file order."""
    out = {}
    with open(os.path.join(run_dir, "requests.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") == "seq" and r.get("name"):
                out.setdefault(r["item"], []).append(r["name"])
    return out


pool_names = vc.pool_names


def folder_gb():
    tot = 0
    for dp, _, fs in os.walk(FBV):
        for f in fs:
            try:
                st = os.lstat(os.path.join(dp, f))
            except OSError:
                continue
            if not os.path.islink(os.path.join(dp, f)):
                tot += st.st_size
    return tot / 1e9


def weights_dir(key):
    return os.path.join(os.environ["HF_HOME"], "hub", "models--" + model(key)["repo"].replace("/", "--"))


def gpu_occupancy():
    """[(index, ['ours' | 'other', ...])] from one NVML query; no GPU memory is allocated."""
    pn = nvml()
    me = os.getuid()
    out = []
    for i in range(pn.nvmlDeviceGetCount()):
        who = []
        for p in pn.nvmlDeviceGetComputeRunningProcesses(pn.nvmlDeviceGetHandleByIndex(i)):
            try:
                who.append("ours" if os.stat("/proc/%d" % p.pid).st_uid == me else "other")
            except OSError:
                who.append("other")
        out.append((i, who))
    return out


def queue_order(models=None):
    """The sessions the queue runs, in its order: every contestant, then the judges; with `models` (a
    subset of the contestants), only those contestants in the queue's order, and no judge, so that no
    judging pool is built (used for additional runs of models whose primary run is elsewhere)."""
    if not models:
        return CONTESTANTS + JUDGES
    keys = {vc.canonical_key(m) for m in models}
    bad = sorted(keys - set(CONTESTANTS))
    if bad:
        raise SystemExit("queue --models takes contestants only, not %s" % ", ".join(bad))
    return [m for m in CONTESTANTS if m in keys]


def parse_until(text):
    """--until: a local date and time (YYYY-MM-DDTHH:MM) -> epoch seconds; it must lie in the future."""
    try:
        t = datetime.datetime.fromisoformat(text).timestamp()
    except ValueError:
        raise SystemExit("--until: not a date and time like 2026-11-23T23:59: %r" % text)
    if t <= time.time():
        raise SystemExit("--until: %s is not in the future" % text)
    return t


class QueueStatus:
    """runs/STATUS: a few plain lines for a person to read."""

    def __init__(self, deadline=None, only=None):
        self.path = os.path.join(RUNS_DIR, "STATUS")
        self.started = time.time()
        if deadline is None:
            self.deadline, self.limit = self.started + QUEUE_HOURS * 3600, "%d h limit" % QUEUE_HOURS
        else:
            self.deadline, self.limit = deadline, "limit given at start"
        self.only = only            # the contestants of a restricted queue (no judges), or None
        self.finished, self.attempts = [], []
        self.state, self.detail = "starting", ""

    def set(self, state=None, detail=None):
        if state is not None:
            self.state = state
        if detail is not None:
            self.detail = detail
        fmt = "%m-%d %H:%M:%S"
        # six lines: analysis/vlm_monitor.sh shows the last six
        lines = ["state: " + self.state,
                 "detail: " + (self.detail or "-"),
                 "times: queue started %s, last update %s, gives up at %s (%s)" % (
                     time.strftime(fmt, time.localtime(self.started)), time.strftime(fmt),
                     time.strftime(fmt, time.localtime(self.deadline)), self.limit),
                 "sessions finished%s: " % (" (this queue runs only %s; no judges)" % ", ".join(self.only)
                                            if self.only else "") + (", ".join(self.finished) or "none"),
                 "interrupted or failed attempts (kept aside): " + (", ".join(self.attempts) or "none"),
                 "stop: python3 analysis/vlm_ws.py queue-stop  (or on the workstation: touch runs/STOP)"]
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.path)


def cmd_queue(args):
    only = getattr(args, "models", None)
    order = queue_order(only)
    deadline = parse_until(args.until) if getattr(args, "until", None) else None
    os.makedirs(RUNS_DIR, exist_ok=True)
    lockf = open(os.path.join(RUNS_DIR, "queue.lock"), "a+")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another queue is running", file=sys.stderr)
        return 1
    lockf.seek(0)
    lockf.truncate()
    lockf.write("%d\n" % os.getpid())
    lockf.flush()
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    st = QueueStatus(deadline, order if only else None)
    child = {"proc": None}

    def on_term(signum, frame):
        if child["proc"] is not None and child["proc"].poll() is None:
            os.killpg(child["proc"].pid, signal.SIGTERM)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def stopped():
        return os.path.exists(STOP_FILE) or time.time() > st.deadline

    def wait_free(need, polls, until, model):
        run = 0
        while True:
            occ = gpu_occupancy()
            free = [i for i, who in occ if not who]
            run = run + 1 if len(free) >= need else 0
            held = ["GPU %d: %s" % (i, "/".join(who)) for i, who in occ if who]
            st.set("waiting for %d free GPU(s) for %s" % (need, model),
                   "%d of %d GPUs free%s; free polls in a row %d of %d; polls every %d min" % (
                       len(free), len(occ), (" (" + ", ".join(held) + ")") if held else "", run, polls,
                       POLL_S // 60))
            log("poll: %s" % st.detail)
            if run >= polls:
                return True
            t_next = time.time() + POLL_S
            if t_next > until:
                return False
            while time.time() < t_next:
                if stopped():
                    return False
                time.sleep(10)

    def make_room(key):
        need = model(key)["gb"]
        for other in CONTESTANTS + JUDGES:
            if folder_gb() + need <= DISK_BUDGET_GB:
                break
            if other != key and latest_done(other) and os.path.isdir(weights_dir(other)):
                log("deleting weights of %s (runs complete)" % other)
                shutil.rmtree(weights_dir(other))
        return folder_gb() + need <= DISK_BUDGET_GB

    def fetch(model):
        st.set("downloading %s" % model, "")
        cmd_fetch(argparse.Namespace(model=model))

    def build_pool():
        with open(os.path.join(DATA, "items.json")) as f:
            man = json.load(f)
        with open(os.path.join(DATA, "pool_inputs.json")) as f:
            extra = json.load(f)["extra_names"]
        runs = {}
        for m in CONTESTANTS:
            d = latest_done(m)
            if d:
                runs[os.path.basename(d)] = seq_names(d)
        used = sorted(runs)
        pool = vc.judge_pool(man, pool_names(extra, runs))
        write_json(os.path.join(DATA, "judge_pool.json"), {"version": 1, "runs": used, "items": pool})
        log("judge pool: %d items, %d names, from %s" % (len(pool), sum(len(p["names"]) for p in pool), used))

    def run_child(model, extra, shared=False):
        kind = "judge" if model in JUDGES else "session"
        name = "%s_%s" % (datetime.date.today().isoformat(), model)
        run = os.path.join(RUNS_DIR, name)
        if os.path.isdir(run):
            os.rename(run, run + "_attempt_" + time.strftime("%H%M%S"))
        os.makedirs(run)
        cmd = [sys.executable, os.path.join(HERE, "vlm_remote.py"), kind, model, "--run", run, "--no-hold",
               "--max-seconds", str(int(max(600, min(3 * 3600, st.deadline - time.time()))))]
        if shared:
            cmd.append("--shared")
        if extra:
            cmd += ["--extra"] + list(extra)
        out = open(os.path.join(run, "session.out"), "w")
        child["proc"] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        expected = "%d requests" % (REPEATS * 124 + 54 + 176) if kind == "session" else "judgments"
        while child["proc"].poll() is None:
            if os.path.exists(STOP_FILE):
                os.killpg(child["proc"].pid, signal.SIGTERM)
            notes = []
            try:
                with open(os.path.join(run, "status.json")) as f:
                    notes = json.load(f).get("notes") or []
            except (OSError, ValueError):
                pass
            n = 0
            for fn in ("requests.jsonl", "judgments.jsonl"):
                try:
                    with open(os.path.join(run, fn)) as f:
                        n += sum(1 for _ in f)
                except OSError:
                    pass
            st.set("running %s%s" % (model, " in SHARED mode (one GPU shared with an idle process of another account; "
                                                "memory cap and watchdog)" if shared else ""),
                   "%s; %d recorded (about %s in all)" % (notes[-1] if notes else "starting", n, expected))
            time.sleep(15)
        child["proc"] = None
        out.close()
        return run_state(run) or "failed", run

    log("queue: start (pid %d)%s; gives up at %s" % (
        os.getpid(), (", only " + " ".join(order) + ", no judges") if only else "",
        time.strftime("%Y-%m-%d %H:%M", time.localtime(st.deadline))))
    clean_last = False
    code = 0
    try:
        for mk in order:
            if latest_done(mk):
                log("%s: already done" % mk)
                continue
            if getattr(args, "shared", False) and mk in SHARED:
                while True:
                    if stopped():
                        raise KeyboardInterrupt()
                    if not snapshot_path(mk):
                        if not make_room(mk):
                            break
                        fetch(mk)
                    log("%s: starting in SHARED mode" % mk)
                    state, run = run_child(mk, [], shared=True)
                    log("%s (shared): %s" % (mk, state))
                    if state != "done" and stopped():
                        os.rename(run, run + "_stopped_" + time.strftime("%H%M%S"))
                        raise KeyboardInterrupt()
                    if state == "done":
                        st.finished.append("%s (shared GPU, %s)" % (mk, time.strftime("%H:%M")))
                        break
                    tag = "contaminated" if state in ("yielded", "busy") else state
                    os.rename(run, "%s_shared_%s_%s" % (run, tag, time.strftime("%H%M%S")))
                    st.attempts.append("%s shared %s at %s" % (mk, state, time.strftime("%H:%M")))
                    if state not in ("yielded", "busy"):
                        log("%s: shared mode ended in %s; the mk goes to the normal queue" % (mk, state))
                        break
                    # the other program showed activity: wait until it has been idle for 10 minutes again
                    while True:
                        st.set("yielded to another user (shared mode)",
                               "%s interrupted; waiting until the other process has been idle for %d min "
                               "(sampled every second)" % (mk, SHARED_REIDLE_S // 60))
                        ok, _, rec = idle_check(SHARED_REIDLE_S, abort=stopped)
                        if stopped():
                            raise KeyboardInterrupt()
                        if ok:
                            break
                        log("other process not idle yet: %s" % "; ".join(rec["reasons"]))
                if latest_done(mk):
                    continue
            need = model(mk)["gpus"]
            k = 0
            while True:
                if stopped():
                    raise KeyboardInterrupt()
                if mk != "judge_b" and not snapshot_path(mk):
                    if not make_room(mk):
                        st.set("stopped: %s does not fit the disk budget" % mk, "")
                        return 2
                    fetch(mk)
                until = st.deadline if need == 1 else min(st.deadline, time.time() + TWO_GPU_WAIT_S)
                if not wait_free(need, 1 if clean_last else FREE_POLLS, until, mk):
                    if stopped():
                        raise KeyboardInterrupt()
                    if mk == "judge_b":
                        log("judge_b: no two-GPU window; dropped as the design allows")
                        st.attempts.append("judge_b dropped (no two-GPU window)")
                        break
                    raise KeyboardInterrupt()
                if latest_done(mk):
                    break
                if mk == "judge_b" and not snapshot_path(mk):
                    if not make_room(mk):
                        log("judge_b: does not fit the disk budget; dropped")
                        st.attempts.append("judge_b dropped (disk budget)")
                        break
                    fetch(mk)
                    clean_last = False
                    continue
                if mk in JUDGES:
                    build_pool()
                log("%s: starting%s" % (mk, (" with " + " ".join(CONTINGENCIES[k])) if k else ""))
                state, run = run_child(mk, CONTINGENCIES[k])
                log("%s: %s" % (mk, state))
                if state != "done" and stopped():
                    os.rename(run, run + "_stopped_" + time.strftime("%H%M%S"))
                    st.attempts.append("%s stopped at %s" % (mk, time.strftime("%H:%M")))
                    raise KeyboardInterrupt()
                clean_last = state == "done"
                if state == "done":
                    st.finished.append("%s (%s)" % (mk, time.strftime("%H:%M")))
                    break
                os.rename(run, "%s_%s_%s" % (run, "contaminated" if state in ("yielded", "busy") else state,
                                             time.strftime("%H%M%S")))
                st.attempts.append("%s %s at %s" % (mk, state, time.strftime("%H:%M")))
                if state in ("yielded", "busy"):
                    st.set("yielded to another user", "%s interrupted; back to waiting" % mk)
                    continue
                if state == "failed" and k + 1 < len(CONTINGENCIES):
                    k += 1
                    continue
                if state == "failed":
                    log("%s: failed after the listed contingencies; dropped" % mk)
                    break
                st.set("stopped: %s ended in state %s; needs inspection" % (mk, state), "")
                return 5
        st.set("done: every session finished", "")
        log("queue: all sessions finished")
    except KeyboardInterrupt:
        why = "stop requested" if os.path.exists(STOP_FILE) else (
            ("%s reached" % st.limit if deadline is None else "give-up time reached")
            if time.time() > st.deadline else "terminated")
        st.set("stopped: " + why, "")
        log("queue: " + why)
        code = 3
    finally:
        if child["proc"] is not None and child["proc"].poll() is None:
            os.killpg(child["proc"].pid, signal.SIGTERM)
            child["proc"].wait(timeout=180)
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()
    return code


def cmd_serve(args):
    """vLLM's command-line program with its arguments read from a file (see Server)."""
    with open(args.args_file) as f:
        argv = json.load(f)
    sys.argv = ["vllm"] + argv
    from vllm.entrypoints.cli.main import main as vllm_main
    return vllm_main()


# ----------------------------------------------------------------------------- pool and Slurm job

def build_pool_file(out, run_dirs, names_file=None, site=None):
    """Write the blind judging pool for the given primary contestant runs: run folders on this machine
    (`run_dirs`, each must have finished) and runs elsewhere whose names come in `names_file`
    ({"runs": {run id: {"model": alias, "names": {item: [names]}}}}). Returns the pool record."""
    with open(os.path.join(DATA, "items.json")) as f:
        man = json.load(f)
    with open(os.path.join(DATA, "pool_inputs.json")) as f:
        extra = json.load(f)["extra_names"]
    site = site or os.environ.get("FB_SITE") or "local"
    runs, models, dropped = {}, {}, []
    if names_file:
        with open(names_file) as f:
            for rid, rec in json.load(f)["runs"].items():
                runs[rid] = rec["names"]
                models[rid] = vc.canonical_key(rec.get("model"))
    for d in run_dirs:
        d = d.rstrip("/")
        if os.path.exists(d + ".dropped"):
            dropped.append(os.path.basename(d))
            log("pool: %s was dropped (did not start with any listed setting); left out" % os.path.basename(d))
            continue
        if run_state(d) != "done":
            raise SystemExit("pool: %s is not a finished run" % d)
        with open(os.path.join(d, "status.json")) as f:
            mk = vc.canonical_key(json.load(f).get("model"))
        rid = "%s/%s" % (site, os.path.basename(d))
        runs[rid] = seq_names(d)
        models[rid] = mk
    per_model = {}
    for rid, mk in models.items():
        per_model.setdefault(mk, []).append(rid)
    twice = {mk: r for mk, r in per_model.items() if len(r) > 1}
    if twice:
        raise SystemExit("pool: more than one run of a model (only the primary run may enter the pool): %s" % twice)
    pool = vc.judge_pool(man, pool_names(extra, runs))
    rec = {"version": 2, "runs": sorted(runs), "models": {rid: models[rid] for rid in sorted(runs)},
           "dropped": dropped, "items": pool}
    write_json(out, rec)
    log("judge pool: %d items, %d names, from %s" % (len(pool), sum(len(p["names"]) for p in pool), sorted(runs)))
    return rec


def cmd_pool(args):
    build_pool_file(args.out, [r if os.path.isabs(r) else os.path.join(RUNS_DIR, r) for r in args.run],
                    args.names_file)


def node_record():
    """One line about the machine of this job: node name, CPU model, CPUs of the job and of the node, and how
    many jobs of other accounts were running on the same node when the job started (a count, no names), since
    they share the node's CPUs, memory bandwidth and PCIe with our session."""
    node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename.split(".")[0]
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    others = "unknown"
    if os.environ.get("SLURM_JOB_ID"):
        try:
            out = subprocess.run(["squeue", "-h", "-w", node, "-t", "R", "-o", "%i %u"], capture_output=True,
                                 text=True, timeout=60).stdout
            me = os.environ.get("USER") or ""
            others = str(sum(1 for line in out.splitlines() if line.split()[1:] and line.split()[1] != me))
        except Exception:  # noqa: BLE001
            pass
    return "node %s; cpu %s; job cpus %s of %d on the node; other accounts' running jobs on the node at start: %s\n" % (
        node, cpu, os.environ.get("SLURM_CPUS_ON_NODE", "?"), os.cpu_count() or 0, others)


def env_record(run, python=None):
    """Python, node, GPU and package versions of this run (environment.txt in the run folder)."""
    python = python or sys.executable
    with open(os.path.join(run, "environment.txt"), "w") as f:
        f.write("python %s; environment %s\n" % (sys.version.split()[0], "staged on node-local storage"
                                                  if python != sys.executable else "on the shared folder"))
        f.write(node_record())
        for cmd in (["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv"],
                    [python, "-m", "pip", "freeze"]):
            try:
                f.write(subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout)
            except Exception as e:  # noqa: BLE001
                f.write("%s: %s\n" % (cmd[0], e))


ENV_FAULTS = ("Could not find nvcc", "ModuleNotFoundError", "ImportError", "CUDA driver version is insufficient",
              "No such file or directory: 'nvcc'", "libcuda.so", "ninja: command not found",
              "Failed to load dynamic shared library", "GLIBCXX_", "libnvrtc", "NVRTCCompiler failed")


def environment_fault(server_log):
    """The first signature of a broken software environment in a server log, or None (a model that does not
    fit or does not load raises other errors, which the listed settings are for)."""
    try:
        with open(server_log, errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    for sig in ENV_FAULTS:
        if sig in text:
            return sig
    return None


def parse_slurm_time(t):
    """Seconds of a Slurm time ([D-]HH:MM:SS, MM:SS); None if not a time."""
    try:
        d, _, hms = t.strip().rpartition("-")
        parts = [int(x) for x in hms.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return int(d or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError:
        return None


def slurm_time_left():
    """Seconds left in this Slurm job (squeue %L), or None outside a job."""
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"], capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return None
    return parse_slurm_time(out) if out.strip() else None


def stage_env():
    """Copy the Python environment to node-local storage when an archive of it exists (scratch/env.tar):
    Python importing thousands of modules from a network file system takes minutes, one archive read takes
    seconds. Returns (python, local folder) or (None, None) to use the shared environment."""
    tar = os.path.join(FBV, "scratch", "env.tar")
    base = os.environ.get("FB_LOCAL_TMP", "/tmp")
    if not os.path.exists(tar) or os.environ.get("FB_STAGE_ENV") == "0" or not os.path.isdir(base):
        return None, None
    local = os.path.join(base, "fbvlm_%s" % (os.environ.get("SLURM_JOB_ID") or os.getpid()))
    t0 = time.time()
    try:
        os.makedirs(local, mode=0o700, exist_ok=True)
        subprocess.run(["tar", "-C", local, "-xf", tar], check=True, timeout=1200)
        py = os.path.join(local, "env", "bin", "python")
        subprocess.run([py, "-c", "import torch, vllm"], check=True, timeout=900, capture_output=True)
    except Exception as e:  # noqa: BLE001
        log("staging the environment failed (%s); using the shared one" % e)
        shutil.rmtree(local, ignore_errors=True)
        return None, None
    log("environment staged on node-local storage in %.0f s" % (time.time() - t0))
    return py, local


def restarted():
    """True when Slurm restarted this job (requeued after preemption or a node failure)."""
    try:
        return int(os.environ.get("SLURM_RESTART_COUNT") or 0) > 0
    except ValueError:
        return True


def cmd_job(args):
    """Inside one Slurm job: one naming session (or judging session) on the GPU(s) Slurm allocated, retried
    with the design's listed settings when the server does not start. Exit 0 when the session finished or
    the model was dropped after every listed setting (as the design allows; a <name>.dropped file says so)."""
    key = vc.canonical_key(args.model)
    kind = "judge" if key in JUDGES else "session"
    if restarted():
        # a preempted job that Slurm put back in the queue: never resumed (its folder is set aside from the Mac)
        log("this job was restarted by Slurm (count %s); not run" % os.environ.get("SLURM_RESTART_COUNT"))
        return 3
    os.makedirs(RUNS_DIR, exist_ok=True)
    base = os.path.join(RUNS_DIR, args.name)
    left = slurm_time_left()
    if left:
        args.time_limit_s = left                             # the job's real limit (it may have been raised)
    deadline = time.time() + args.time_limit_s - 150          # leave time to stop the server before Slurm's limit
    # Slurm sends SIGTERM to every process at its time limit: the session stops its server; this
    # process only waits for it and then sets the folder aside.
    signal.signal(signal.SIGTERM, lambda s, f: log("SIGTERM: waiting for the session to stop its server"))
    py, local = stage_env()
    try:
        return job_attempts(args, key, kind, base, deadline, py or sys.executable)
    finally:
        if local:
            shutil.rmtree(local, ignore_errors=True)


def job_attempts(args, key, kind, base, deadline, python):
    """The attempts of one job: the design's settings, then the listed contingencies."""
    for k, extra in enumerate(CONTINGENCIES):
        if os.path.isdir(base):
            os.rename(base, base + "_attempt_" + time.strftime("%H%M%S"))
        os.makedirs(base)
        env_record(base, python)
        cmd = [python, os.path.abspath(__file__), kind, key, "--run", base, "--no-hold",
               "--max-seconds", str(int(max(300, deadline - time.time())))]
        if args.gpus:
            cmd += ["--gpus", str(args.gpus)]
        if kind == "judge":
            pool = os.path.join(base, "judge_pool.json")
            build_pool_file(pool, [r if os.path.isabs(r) else os.path.join(RUNS_DIR, r) for r in args.pool_run],
                            args.names_file)
            cmd += ["--pool", pool]
        if extra:
            cmd += ["--extra"] + list(extra)
        log("%s: %s%s" % (key, kind, (" with " + " ".join(extra)) if extra else ""))
        with open(os.path.join(base, "session.out"), "w") as out:
            subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)
        state = run_state(base) or "failed"
        log("%s: %s" % (key, state))
        if state == "done":
            return 0
        env_fault = state == "failed" and environment_fault(os.path.join(base, "server.log"))
        if env_fault:
            # a broken environment is not the model's failure: no further listed setting, no 'dropped' marker
            log("%s: the server failed for a reason of the environment (%s); not retried" % (key, env_fault))
            os.rename(base, "%s_envfault_%s" % (base, time.strftime("%H%M%S")))
            return 7
        os.rename(base, "%s_%s_%s" % (base, "contaminated" if state in ("yielded", "busy") else state,
                                      time.strftime("%H%M%S")))
        if state != "failed":
            return 5
        if k + 1 == len(CONTINGENCIES):
            with open(base + ".dropped", "w") as f:
                f.write("%s did not start with any listed setting (%s)\n" % (key, time.strftime("%Y-%m-%d %H:%M")))
            log("%s: failed after the listed contingencies; dropped" % key)
            return 0
        if time.time() > deadline - 300:
            log("%s: no time left for the next listed setting" % key)
            return 6
    return 6


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus")
    q = sub.add_parser("queue")
    q.add_argument("--shared", action="store_true", help="run the allowed models in shared mode first")
    q.add_argument("--models", nargs="+", choices=CONTESTANTS,
                   help="only these contestants, in the queue's order (additional runs): no judge, no judging pool")
    q.add_argument("--until", help="give up at this local time (YYYY-MM-DDTHH:MM) instead of after %d h" % QUEUE_HOURS)
    p = sub.add_parser("fetch")
    p.add_argument("model", choices=model_choices())
    p = sub.add_parser("serve")
    p.add_argument("args_file", help="JSON list of vLLM arguments (written by Server)")
    p = sub.add_parser("pool")
    p.add_argument("--out", required=True)
    p.add_argument("--run", action="append", default=[], help="finished contestant run folder (or name in runs/)")
    p.add_argument("--names-file", help="names of primary runs made elsewhere (JSON)")
    p = sub.add_parser("job")
    p.add_argument("model", choices=model_choices())
    p.add_argument("--name", required=True, help="run folder name in runs/")
    p.add_argument("--time-limit-s", type=float, default=3600, help="the Slurm job's time limit")
    p.add_argument("--gpus", type=int, help="GPUs to use instead of the design's number (recorded)")
    p.add_argument("--pool-run", action="append", default=[], help="judges: a primary contestant run in runs/")
    p.add_argument("--names-file", help="judges: names of primary runs made elsewhere (JSON)")
    for name in ("session", "judge", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("model", choices=model_choices())
        p.add_argument("--run", required=True)
        p.add_argument("--gpus", type=int, help="GPUs to use instead of the design's number (recorded)")
        p.add_argument("--pool", help="judge: the pool file (default data/judge_pool.json)")
        p.add_argument("--port", type=int)
        p.add_argument("--max-seconds", type=float, default=3 * 3600)
        p.add_argument("--hold-max", type=float, default=1800)
        p.add_argument("--only-roundtrip", action="store_true")
        p.add_argument("--no-hold", action="store_true", help="do not wait for the Mac's round trip")
        p.add_argument("--shared", action="store_true", help="share one GPU with another account's idle process "
                       "(allowed models only, memory cap, watchdog)")
        p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                       help="extra vLLM arguments (the listed contingencies only)")
    args = ap.parse_args(argv)
    if getattr(args, "model", None):
        args.model = vc.canonical_key(args.model)
    fn = {"gpus": cmd_gpus, "fetch": cmd_fetch, "session": cmd_session, "judge": cmd_judge, "smoke": cmd_smoke,
          "queue": cmd_queue, "pool": cmd_pool, "job": cmd_job, "serve": cmd_serve}
    sys.exit(fn[args.cmd](args) or 0)


if __name__ == "__main__":
    main()
