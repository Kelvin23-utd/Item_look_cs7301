#!/usr/bin/env python3
"""Tables, plot and judge pool of the VLM naming experiment, on the Mac.

    analysis/.venv-bench/bin/python analysis/vlm_report.py pool
    analysis/.venv-bench/bin/python analysis/vlm_report.py report --data backups/phone_files_2026-11-17_0008 \
        --md analysis/notes/2026-11-17_vlm_naming_results.md \
        --changes-file analysis/notes/2026-11-17_vlm_naming_changes.md

`pool` collects every name proposed for each R and S item (the primary run of every contestant, every
variant and repeat, the baselines, and the typed name of R-human items), dedupes them after
normalisation and writes the blind judging pool to analysis/out/vlm/package/judge_pool.json (names
only; no method, no hint which name was typed).

`report` reads analysis/out/vlm/ (items_local, baselines, the mirrors ws_runs/ of the lab workstation,
cluster_runs/ of a university cluster and cluster2_runs/ of a second one, whose runs are additional only)
and the phone's log (--data), and writes a number-only
Markdown note and analysis/out/vlm/quality_cost.png. A model's primary run is its first complete run in
time at any site; it gives the names. Every run gives cost rows for its own site and GPU. A judge run
counts when the pool it judged is the pool of the primary runs. Models are named by their aliases.
Definitions: analysis/notes/2026-11-17_vlm_naming_design.md.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlm_common as vc  # noqa: E402

OUT = os.path.join(HERE, "out", "vlm")
CONTESTANTS = list(vc.CONTESTANTS)
JUDGES = list(vc.JUDGES)
PRIMARY_JUDGE = vc.PRIMARY_JUDGE
# Judges that share a model family with contestants (the design's judge table); stated beside their scores.
SAME_FAMILY = {"judge_b": ["model_a", "model_c"]}
LABEL = dict({k: vc.alias_label(k) for k in vc.PUBLIC_MODELS}, **{
    "phone": "Phone detector (EfficientDet-Lite0)", "clip": "MobileCLIP2-S2 zero-shot", "yoloe11s": "YOLOE-11S (Track B)",
    "owlv2": "OWLv2 (Track B)"})
BASELINES = ["phone", "clip", "yoloe11s", "owlv2"]
STAGES = [("queue", "vllm:request_queue_time_seconds_sum"), ("prefill", "vllm:request_prefill_time_seconds_sum"),
          ("decode", "vllm:request_decode_time_seconds_sum"), ("engine", "vllm:e2e_request_latency_seconds_sum")]
# Mirrors of the GPU machines' runs/ folders: (folder under analysis/out/vlm, site)
SITES = [("ws_runs", "workstation"), ("cluster_runs", "cluster"), ("cluster2_runs", "cluster2")]
SITE_TEXT = {"workstation": "lab workstation", "cluster": "university cluster", "cluster2": "second university cluster"}
# Sites whose runs are additional only (cost on their GPU and name agreement; changes note, item 21): a run there
# is never a primary run, so it never enters naming quality, the label-free signals or the judging pool.
ADDITIONAL_SITES = {"cluster2"}
# Order of the sites in the plot's number table (the sites added later come last)
SITE_ORDER = {"cluster": 0, "workstation": 1, "cluster2": 2}
WS_GPU = "RTX 4090, 24 GB"        # the workstation's GPUs (design note); older runs did not record the name


# ----------------------------------------------------------------------------- loading

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def load_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except OSError:
        pass
    return rows


def runs_of(out, key, suffix=""):
    """Every finished run of a model in the mirrors (out/ws_runs, out/cluster_runs, out/cluster2_runs), first
    completed first: dicts with dir, site, id ("site/folder"), t_end (the status time when the run ended) and
    status. A folder counts only when its name is exactly <date>_<key><suffix> (the alias or the former
    key), so set-aside attempts are never taken. The id uses the alias even for a former-key folder."""
    key = vc.canonical_key(key)
    res = []
    for sub, site in SITES:
        for k in vc.run_keys(key):
            pat = re.compile(r"^\d{4}-\d{2}-\d{2}_%s%s$" % (re.escape(k), re.escape(suffix)))
            for d in glob.glob(os.path.join(out, sub, "*_" + k + suffix)):
                if not pat.match(os.path.basename(d)):
                    continue
                st = load_json(os.path.join(d, "status.json"), {}) or {}
                if st.get("state") == "done":
                    res.append({"dir": d, "site": site, "id": "%s/%s" % (site, vc.neutral_name(os.path.basename(d))),
                                "t_end": st.get("t") or 0.0, "status": st})
    res.sort(key=lambda r: (r["t_end"], r["id"]))
    return res


def primary_run(out, key):
    """The primary run of a model: its first complete run in time, at any site (rule fixed before the
    cluster runs) except the additional-only sites. Later runs give cost on their hardware and a
    name-agreement check only."""
    r = [x for x in runs_of(out, key) if x["site"] not in ADDITIONAL_SITES]
    return r[0] if r else None


def hardware(run):
    """GPU of a run: the NVML name recorded at the start, else the workstation's known GPU."""
    st = run["status"]
    used = st.get("gpus") or [0]
    gs = [g for g in (st.get("gpus_before") or []) if g.get("index") in used]
    if gs and gs[0].get("name"):
        name = "%s, %.0f GiB" % (gs[0]["name"].replace("NVIDIA ", "").replace("GeForce ", ""),
                                 gs[0]["mem_total_mib"] / 1024.0)
    elif run["site"] == "workstation":
        name = WS_GPU
    else:
        name = "GPU not recorded"
    return ("%d × " % len(used) if len(used) > 1 else "") + name


def site_label(run):
    return "%s (%s%s)" % (SITE_TEXT.get(run["site"], run["site"]), hardware(run),
                          "; shared GPU" if run["status"].get("shared") else "")


def load_run(out, run, key):
    d = run["dir"]
    rts = [r for r in runs_of(out, key, "_roundtrip") if r["site"] == run["site"]]
    rt_dir = rts[-1]["dir"] if rts else d
    return dict(run, **{
        "key": key, "label": site_label(run), "hw": hardware(run),
        "rows": load_jsonl(os.path.join(d, "requests.jsonl")),
        "judg": load_jsonl(os.path.join(d, "judgments.jsonl")),
        "conc": load_json(os.path.join(d, "concurrency.json"), []),
        "pings": load_json(os.path.join(rt_dir, "pings_local.json"), []),
        "rt": load_jsonl(os.path.join(rt_dir, "roundtrip_mac.jsonl")),
        "log": open(os.path.join(d, "server.log"), errors="replace").read()
        if os.path.exists(os.path.join(d, "server.log")) else "",
        "power": os.path.join(d, "power.csv"),
        "pool": load_json(os.path.join(d, "judge_pool.json")),
    })


def seq_names(rows):
    out = {}
    for r in rows:
        if r.get("phase") == "seq" and r.get("name"):
            out.setdefault(r["item"], []).append(r["name"])
    return out


class Results:
    def __init__(self, out=OUT):
        self.out = out
        self.local = load_json(os.path.join(out, "local", "items_local.json"), {"items": {}})
        self.items = self.local["items"]
        self.manifest = load_json(os.path.join(out, "package", "items.json"), {"items": []})
        self.base = load_json(os.path.join(out, "baselines.json"), {"names": {}, "cost": {}, "notes": []})
        # contestants: the primary run gives the names (quality, label-free signals); every finished run,
        # primary or additional, gives cost on its own hardware
        self.runs, self.extra_runs, self.run_dirs = {}, {}, {}
        for m in CONTESTANTS:
            p = primary_run(out, m)
            if p:
                self.runs[m] = load_run(out, p, m)
                self.run_dirs[m] = p["dir"]
                self.extra_runs[m] = [load_run(out, r, m) for r in runs_of(out, m) if r["dir"] != p["dir"]]
        self.models = [m for m in CONTESTANTS if self.runs.get(m, {}).get("rows")]
        # judges: the first finished run whose pool is the pool of the primary runs
        self.expected_pool = vc.judge_pool(self.manifest, vc.pool_names(
            extra_names(self), {self.runs[m]["id"]: seq_names(self.runs[m]["rows"]) for m in self.models}))
        self.pool_frame = {p["item"]: p["with_frame"] for p in self.expected_pool}
        self.judge_notes = {}
        for j in JUDGES:
            cands = [r for r in runs_of(out, j) if r["site"] not in ADDITIONAL_SITES]
            for r in cands:
                run = load_run(out, r, j)
                if run["pool"] and run["pool"].get("items") == self.expected_pool and run["judg"]:
                    self.runs[j] = run
                    self.run_dirs[j] = r["dir"]
                    break
            if cands and j not in self.runs:
                self.judge_notes[j] = ("%d finished run(s) found, but none judged the pool of the primary runs; "
                                       "not used" % len(cands))
        self.judges = [j for j in JUDGES if self.runs.get(j, {}).get("judg")]

    def ids(self, which):
        """Item ids of a set: R, Rh (typed reference), S, Sf (S with a saved keyframe), RS."""
        out = []
        for iid, v in sorted(self.items.items()):
            s = v["set"]
            if which == "R" and s == "R" or which == "S" and s == "S" or which == "RS" and s in ("R", "S"):
                out.append(iid)
            elif which == "Rh" and s == "R" and v.get("reference"):
                out.append(iid)
            elif which == "Sf" and s == "S" and v.get("frame"):
                out.append(iid)
        return out

    def vlm_names(self, model, variant, repeat=1, phase="seq", run=None):
        out = {}
        for r in (run or self.runs.get(model, {})).get("rows", []):
            if r.get("phase") == phase and r.get("variant") == variant and r.get("repeat") == repeat:
                out[r["item"]] = r.get("name")
        return out

    def names(self, method, variant):
        """item -> name for a method and variant; baselines exist for their own input only."""
        if method in CONTESTANTS:
            return self.vlm_names(method, variant)
        if method in ("phone", "clip") and variant == "crop-full":
            return self.base["names"].get(method, {})
        if method in ("yoloe11s", "owlv2") and variant == "keyframes":
            return self.base["names"].get(method, {})
        return {}

    def verdicts(self, judge, order="A"):
        """item -> {normalised name: verdict}."""
        out = {}
        for r in self.runs.get(judge, {}).get("judg", []):
            if r.get("order") == order:
                out[r["item"]] = {vc.normalize(n): v for n, v in (r.get("verdicts") or {}).items()}
        return out

    def cost_runs(self):
        """(model, run) for every finished contestant run, primary first, for the cost tables."""
        return [(m, r) for m in self.models for r in [self.runs[m]] + self.extra_runs.get(m, [])]


def run_agreement(R, m, run, variants=None):
    """Names of an additional run against the primary run (sequential phase, repeat 1): (identical
    normalised name, same name by the label-free rule, n) over every (item, input)."""
    same = ident = n = 0
    for v in variants or vc.VARIANTS:
        a, b = R.vlm_names(m, v), R.vlm_names(m, v, run=run)
        for i in sorted(set(a) & set(b)):
            n += 1
            ident += bool(a[i]) and bool(b[i]) and vc.normalize(a[i]) == vc.normalize(b[i])
            same += vc.same_name(a[i], b[i])
    return ident, same, n


# ----------------------------------------------------------------------------- pool

def extra_names(R):
    """item -> names that are not from a contestant: every baseline's name and, for R-human, the typed
    name (unlabelled: the file does not say which is which)."""
    out = {}
    for iid in R.ids("RS"):
        names = [R.base["names"].get(b, {}).get(iid) for b in BASELINES]
        names.append(R.items[iid].get("reference"))
        out[iid] = sorted({n for n in names if n})
    return out


def cmd_pool_inputs(args):
    """package/pool_inputs.json: the non-contestant names, so the GPU machine can build the same pool."""
    R = Results(args.out)
    with open(os.path.join(args.out, "package", "pool_inputs.json"), "w") as f:
        json.dump({"version": 1, "extra_names": extra_names(R)}, f, indent=1)


def cmd_pool(args):
    """The blind judging pool from the primary runs (the same function the GPU machine uses)."""
    R = Results(args.out)
    if not R.models:
        sys.exit("no contestant results under %s" % args.out)
    path = os.path.join(args.out, "package", "judge_pool.json")
    with open(path, "w") as f:
        json.dump({"version": 2, "runs": sorted(R.runs[m]["id"] for m in R.models), "items": R.expected_pool}, f,
                  indent=1)
    print("judge pool: %d items, %d (item, name) pairs, from %s" % (
        len(R.expected_pool), sum(len(p["names"]) for p in R.expected_pool), [R.runs[m]["id"] for m in R.models]))

# ----------------------------------------------------------------------------- helpers

def pct(k, n):
    if not n:
        return "–"
    lo, hi = vc.wilson(k, n)
    return "%d/%d = %.0f%% [%.0f, %.0f]" % (k, n, 100 * k / n, 100 * lo, 100 * hi)


def f1(x, nd=2):
    return "–" if x is None or x != x else ("%.*f" % (nd, x))


def med_p90(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    return "%s [%s]" % (f1(vc.median(xs), nd), f1(vc.quantile(xs, 0.9), nd))


def judged(R, judge, names, ids, order="A"):
    """(right, lenient, unclear, missing, n) over the ids the judge judged (an item whose request failed has no
    verdict at all and is left out: missing data, not a wrong name). A name without its own verdict in a judged
    item counts as not right."""
    V = R.verdicts(judge, order)
    ids = [i for i in ids if V.get(i)]
    right = len_ = unc = miss = 0
    for i in ids:
        n = names.get(i)
        v = V.get(i, {}).get(vc.normalize(n)) if n else None
        if n and v is None:
            miss += 1
        right += v == "right"
        len_ += v in ("right", "close")
        unc += v == "unclear"
    return right, len_, unc, miss, len(ids)


def correct_vector(R, judge, names, ids, lenient=False):
    """Per judged item (see judged), whether the name is judged right (or close when lenient)."""
    V = R.verdicts(judge)
    ok = ("right", "close") if lenient else ("right",)
    return [bool(names.get(i)) and V.get(i, {}).get(vc.normalize(names[i])) in ok for i in ids if V.get(i)]


def judged_ids(R, judge, ids):
    V = R.verdicts(judge)
    return [i for i in ids if V.get(i)]


def mcnemar_vs(a, b):
    """p of a versus b (lists of booleans on the same items)."""
    x = sum(1 for p, q in zip(a, b) if p and not q)
    y = sum(1 for p, q in zip(a, b) if q and not p)
    return vc.mcnemar_exact(x, y), x, y


def power_rows(path):
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append((float(r["t"]), int(r["gpu"]), float(r["power_w"])))
    except OSError:
        pass
    return rows


def mean_power(rows, t0, t1):
    """Mean total power (sum over GPUs) between t0 and t1."""
    gpus = sorted({g for _, g, _ in rows})
    tot = 0.0
    for g in gpus:
        pts = [(t, w) for t, gg, w in rows if gg == g and t0 <= t <= t1]
        if pts:
            tot += sum(w for _, w in pts) / len(pts)
    return tot


def parse_log(text):
    out = {}
    m = re.search(r"Model loading took ([0-9.]+) GiB(?: memory)? and ([0-9.]+) seconds", text)
    if m:
        out["weights_gib"], out["load_s"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Loading weights took ([0-9.]+) seconds", text)
    if m:
        out["weights_s"] = float(m.group(1))
    m = re.search(r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) seconds", text)
    if m:
        out["init_s"] = float(m.group(1))
    m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", text)
    if m:
        out["kv_tokens"] = int(m.group(1).replace(",", ""))
    return out


def phone_times(data):
    """Medians of the phone's own latencies from the log (non-excluded sessions)."""
    import bench_common as bc
    bd = bc.BenchData(data)
    kf = {k: [] for k in ("detectMs", "convertMs", "embedMs", "totalMs")}
    reg = {k: [] for k in ("convertMs", "detectMs", "embedMs")}
    ver = []
    scans = []
    for e in bd.events:
        lat = e.get("latency") or {}
        if e.get("type") == "scan_keyframe":
            for k in kf:
                if isinstance(lat.get(k), (int, float)):
                    kf[k].append(lat[k])
        elif e.get("type") == "register":
            for k in reg:
                if isinstance(lat.get(k), (int, float)):
                    reg[k].append(lat[k])
        elif e.get("type") == "verify" and isinstance(lat.get("totalMs"), (int, float)):
            ver.append(lat["totalMs"])
        elif e.get("type") == "scan_end":
            scans.append(e)
    return {"keyframe": kf, "register": reg, "verify": ver, "scans": scans}


# ----------------------------------------------------------------------------- report

def shared_text(st):
    sh = st.get("shared")
    if not sh:
        return ""
    base = sh.get("baseline") or sh.get("other_before") or {}
    other = sum(v for k, v in base.items() if k.startswith("%d:" % sh["gpu"]))
    wd = sh.get("watchdog") or {}
    return ("; GPU shared with an idle process of another account holding %d MiB; our cap %d MiB; watchdog: %d "
            "samples, %s, lowest free %s MiB" % (other, sh["cap_mib"], wd.get("samples", 0),
                                                "fired" if wd.get("fired") else "no activity seen",
                                                wd.get("min_free_mib")))


def run_text(run):
    st = run["status"]
    g = len(st.get("gpus") or [0])
    ov = st.get("gpus_override")
    return "%s, %d GPU%s%s%s" % (SITE_TEXT.get(run["site"], run["site"]) + " (" + run["hw"] + ")", g,
                                 "" if g == 1 else "s", " (design: %d)" % ov["design"] if ov else "", shared_text(st))


def report(R, args):
    L = []
    w = L.append
    today = datetime.date.today().isoformat()
    w("# VLM naming: results")
    w("")
    w("Generated by `analysis/vlm_report.py` on %s from %s (sessions left out as in Tracks A and B). Design, "
      "prompts and metrics: `analysis/notes/2026-11-17_vlm_naming_design.md`." % (
          today, ", ".join("`%s`" % d for d in R.local.get("data", []))))
    w("")
    w("**Small samples.** One home, one scan of one room and four sessions of tapped items. The human "
      "reference has 8 items; the model-judged sets have 18 to 44. Intervals are Wilson 95% intervals and are "
      "wide; differences inside them are not shown. Every number in a column marked *model-judged* comes from "
      "a judging model, not from a person.")
    w("")
    later = sorted({(SITE_ORDER.get(x["site"], 9), x["site"]) for m in R.models for x in R.extra_runs.get(m, [])
                    if x["site"] in ADDITIONAL_SITES})
    later_hw = {s: sorted({x["hw"] for m in R.models for x in R.extra_runs.get(m, []) if x["site"] == s})
                for _, s in later}
    if not later:
        sites = ("Sessions ran at two sites: the lab workstation of the design (RTX 4090, 24 GB) and a university "
                 "cluster (one GPU type for every cluster session).")
    else:
        sites = ("Sessions ran at %s sites: the lab workstation of the design (RTX 4090, 24 GB), a university "
                 "cluster (one GPU type for every cluster session) and %s." % (
                     {3: "three", 4: "four"}.get(2 + len(later), str(2 + len(later))), " and ".join(
                     "a %s (%s; additional runs only, for cost on that GPU and name agreement)" % (
                         SITE_TEXT[s], " / ".join(later_hw[s])) for _, s in later)))
    w("**Sites and runs.** %s Naming quality and the label-free signals come "
      "from each model's *primary run*, its first complete run in time (rule fixed before the cluster runs); the "
      "judging pool was built from the primary runs. A later run of a model is used only for cost on its own "
      "hardware and for the agreement of its names with the primary run. Every cost row names its site and GPU; "
      "numbers of different sites are never combined. The design question (phone plus a local workstation) is "
      "answered by the RTX 4090 rows where they exist; the cluster rows compare all four contestants on one GPU "
      "type.%s" % (sites, "".join(
          " The %s's rows give the cost of the contestants on its GPU (%s) with the design's serving settings; they "
          "are never merged with the workstation's or the first cluster's rows." % (SITE_TEXT[s], " / ".join(later_hw[s]))
          for _, s in later)))
    w("")
    nR, nRh, nS, nSf = (len(R.ids(x)) for x in ("R", "Rh", "S", "Sf"))
    nV = sum(1 for v in R.items.values() if v["set"] == "V")
    w("Items: R = %d tapped records (R-human = %d with a typed name), S = %d scan entries (S-frame = %d with a "
      "saved keyframe), V = %d other sightings of entries." % (nR, nRh, nS, nSf, nV))
    w("")

    # ---- attempts not used
    att = []
    for sub, site in SITES:
        for a in sorted(glob.glob(os.path.join(R.out, sub, "*_*_*"))):
            if re.search(r"_(contaminated|failed|error|stopped|attempt|busy|nofit|envfault|cancelled|preempted|"
                         r"interrupted)(_\d+)?$", os.path.basename(a)):
                att.append((site, a))
    if att:
        w("Sessions not used (kept on disk, listed for the record; they did not finish, or their timing was taken "
          "while another process could share the GPU):")
        w("")
        for site, a in att:
            st = load_json(os.path.join(a, "status.json"), {}) or {}
            w("- %s, `%s`: %s%s" % (SITE_TEXT[site], vc.neutral_name(os.path.basename(a)), st.get("state", "?"),
                                    ("; " + st["notes"][-2]) if len(st.get("notes") or []) >= 2 else ""))
        w("")

    # ---- models
    w("## Models")
    w("")
    w("The models are named by aliases; their exact identifiers (repository and pinned revision) are kept in a "
      "private file outside the repository. All were served by vLLM 0.33.0 with the settings of the design note.")
    w("")
    w("| Model | Size | Licence | Weights | GPUs (design) | Runs used |")
    w("| --- | --- | --- | --- | --- | --- |")
    for m in CONTESTANTS + JUDGES:
        pm = vc.PUBLIC_MODELS[m]
        if m in R.models:
            ran = "primary: " + run_text(R.runs[m])
            for x in R.extra_runs.get(m, []):
                ran += "; additional (cost and name agreement only): " + run_text(x)
        elif m in R.judges:
            ran = run_text(R.runs[m])
        else:
            ran = "not run" + ((" (" + R.judge_notes[m] + ")") if m in R.judge_notes else "")
        w("| %s%s | %s | %s | %s | %d | %s |" % (LABEL[m], " (judge)" if m in JUDGES else "", pm["size"],
                                                pm["licence"], pm["precision"], pm["gpus"], ran))
    w("")

    # ---- human reference
    w("## Naming quality")
    w("")
    w("### Human reference: names the user typed (R-human, n = %d)" % nRh)
    w("")
    w("Matches = the typed name (or a listed synonym) appears in the method's name (rule in the design note).")
    w("")
    w("| Method | crop-full | crop-320 |")
    w("| --- | --- | --- |")
    rh = R.ids("Rh")
    for meth in ["phone", "clip"] + R.models:
        cells = []
        for v in ("crop-full", "crop-320"):
            nm = R.names(meth, v)
            if not nm:
                cells.append("–")
                continue
            k = sum(vc.human_match(nm.get(i), R.items[i]["reference"]) for i in rh)
            cells.append(pct(k, len(rh)))
        w("| %s | %s | %s |" % (LABEL[meth], cells[0], cells[1]))
    w("")

    # ---- model-judged
    if not R.judges:
        w("### Model-judged numbers: pending")
        w("")
        w("No judging model has run yet (the judges need a GPU no other process uses). Until then this note gives "
          "only the human-reference numbers, the agreement between methods and the label-free signals.")
        w("")
    if R.judges:
        jl = R.judges
        w("### Model-judged: share of items whose name a judge rated right (strict) or right or close (lenient)")
        w("")
        w("Primary judge: %s. %s" % (LABEL[PRIMARY_JUDGE] if PRIMARY_JUDGE in jl else "–",
                                     "Second judge: %s." % LABEL["judge_b"] if "judge_b" in jl
                                     else "**Single-judge results**: the second judge was not run, so the design's "
                                          "rule (no winner without agreement of both judges) cannot be checked; any "
                                          "ordering below rests on one judging model."))
        w("")
        pairs = sum(len(p["names"]) for p in R.expected_pool)
        w("Judging pool: %d items, %d (item, name) pairs from the primary runs of %s, the baselines and the typed "
          "names; each judge saw them blind, in a shuffled order and again reversed. %s" % (
              len(R.expected_pool), pairs, ", ".join(LABEL[m] for m in R.models),
              " ".join("%s gave no valid verdict for %d of %d pairs (first order)." % (
                  LABEL[j], pairs - sum(len(v) for v in R.verdicts(j).values()), pairs) for j in jl)))
        w("")
        for j in jl:
            failed = [r for r in R.runs[j]["judg"] if r.get("error")]
            if failed:
                kinds = sorted({"two images" if R.pool_frame.get(r["item"]) else "one image" for r in failed})
                w("**%s judged %d of %d items.** %d of its %d requests were rejected by the server before the model "
                  "saw them (%s per request); those items have no verdict from %s and are left out of its cells, so "
                  "each cell gives its own n. They are missing data, not wrong names." % (
                      LABEL[j], sum(1 for v in R.verdicts(j).values() if v), len(R.expected_pool), len(failed),
                      len(R.runs[j]["judg"]),
                      " and ".join(kinds), LABEL[j]))
                w("")
        for j in jl:
            kin = [m for m in SAME_FAMILY.get(j, []) if m in R.models]
            if kin:
                w("**%s is not independent of %s.** It is a larger model of the same family (design note), so its "
                  "verdicts may favour their names. The judge from an outside family was meant to check this; read "
                  "its coverage and its agreement with the typed names below before relying on either judge." % (
                      LABEL[j], " and ".join(LABEL[m] for m in kin)))
                w("")
        sets = [("S", "Scan entries, all (S)", ["crop-full", "crop-320"]),
                ("Sf", "Scan entries with a saved keyframe (S-frame): every method and input", vc.VARIANTS + ["keyframes"]),
                ("R", "Tapped records (R)", ["crop-full", "crop-320"])]
        for sid, title, variants in sets:
            ids = R.ids(sid)
            w("#### %s, n = %d" % (title, len(ids)))
            w("")
            hdr = "| Method | Input |"
            for j in jl:
                hdr += " Strict, %s (model-judged) | Lenient, %s (model-judged) |" % (LABEL[j], LABEL[j])
            for j in jl:
                hdr += " vs phone, strict, %s: p (better / worse) |" % LABEL[j]
            w(hdr)
            w("|" + " --- |" * (2 + 3 * len(jl)))
            phone_vec = {j: correct_vector(R, j, R.names("phone", "crop-full"), ids) for j in jl}   # judged items
            for meth in ["phone", "clip", "yoloe11s", "owlv2"] + R.models:
                for v in variants:
                    nm = R.names(meth, v)
                    if not nm or not any(i in nm for i in ids):
                        continue
                    row = "| %s | %s |" % (LABEL[meth], v)
                    for j in jl:
                        r_, l_, u_, mi, n = judged(R, j, nm, ids)
                        row += " %s | %s |" % (pct(r_, n), pct(l_, n))
                    for j in jl:
                        if meth == "phone" or not phone_vec[j]:
                            row += " – |"
                        else:
                            p, b, c = mcnemar_vs(correct_vector(R, j, nm, ids), phone_vec[j])
                            row += " %.3f (%d / %d) |" % (p, b, c)
                    w(row)
            w("")

        # ---- judges vs human
        w("### Judges against the human reference (R-human, n = %d items)" % nRh)
        w("")
        w("| Judge | Typed names judged right | Pairs | Match & right | Match & not right | No match & right | "
          "No match & not right | Agreement, strict | Kappa, strict | Agreement, lenient | Kappa, lenient |")
        w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for j in jl:
            V = R.verdicts(j)
            typed_right = sum(V.get(i, {}).get(vc.normalize(R.items[i]["reference"])) == "right" for i in rh)
            hm, js, jlen = [], [], []
            for i in rh:
                for nn, v in V.get(i, {}).items():
                    hm.append(vc.human_match(nn, R.items[i]["reference"]))
                    js.append(v == "right")
                    jlen.append(v in ("right", "close"))
            n = len(hm)
            tab = [sum(1 for a, b in zip(hm, js) if a and b), sum(1 for a, b in zip(hm, js) if a and not b),
                   sum(1 for a, b in zip(hm, js) if not a and b), sum(1 for a, b in zip(hm, js) if not a and not b)]
            ag_s = sum(a == b for a, b in zip(hm, js))
            ag_l = sum(a == b for a, b in zip(hm, jlen))
            w("| %s | %d/%d | %d | %d | %d | %d | %d | %s | %s | %s | %s |" % (
                LABEL[j], typed_right, len(rh), n, *tab, pct(ag_s, n), f1(vc.cohen_kappa(hm, js)),
                pct(ag_l, n), f1(vc.cohen_kappa(hm, jlen))))
        w("")
        # inter-judge and order
        rows = []
        for j in jl:
            A, B = R.verdicts(j, "A"), R.verdicts(j, "B")
            a, b = [], []
            for i in A:
                for nn, v in A[i].items():
                    if nn in B.get(i, {}):
                        a.append(v)
                        b.append(B[i][nn])
            unc = sum(1 for i in A for v in A[i].values() if v == "unclear")
            tot = sum(len(A[i]) for i in A)
            rows.append("| %s | %d | %d | %s | %s |" % (LABEL[j], tot, unc, pct(sum(x == y for x, y in zip(a, b)), len(a)),
                                                       f1(vc.cohen_kappa(a, b))))
        w("Each judge's consistency between the two name orders, over all pooled (item, name) pairs:")
        w("")
        w("| Judge | Pairs judged | Unclear | Same verdict in both orders | Kappa |")
        w("| --- | --- | --- | --- | --- |")
        L.extend(rows)
        w("")
        if len(jl) == 2:
            A, B = R.verdicts(jl[0]), R.verdicts(jl[1])
            a, b = [], []
            for i in A:
                for nn, v in A[i].items():
                    if nn in B.get(i, {}):
                        a.append(v)
                        b.append(B[i][nn])
            sa, sb = [x == "right" for x in a], [x == "right" for x in b]
            w("Between the two judges: %s same verdict (kappa %s); right versus not right: %s (kappa %s)." % (
                pct(sum(x == y for x, y in zip(a, b)), len(a)), f1(vc.cohen_kappa(a, b)),
                pct(sum(x == y for x, y in zip(sa, sb)), len(sa)), f1(vc.cohen_kappa(sa, sb))))
            w("")
            # ranking check, on the S items both judges judged
            ids = [i for i in R.ids("S") if all(R.verdicts(j).get(i) for j in jl)]
            ranks, texts = [], []
            for j in jl:
                sc = {m: judged(R, j, R.vlm_names(m, "crop-full"), ids)[0] for m in R.models}
                levels = sorted(set(sc.values()), reverse=True)
                ranks.append([sorted(m for m in sc if sc[m] == lv) for lv in levels])
                texts.append("%s: %s" % (LABEL[j], " > ".join(" = ".join("%s (%d)" % (LABEL[m], sc[m]) for m in grp)
                                                              for grp in ranks[-1])))
            w("Contestant order on the S items both judges judged (n = %d of %d), crop-full, strict (items judged "
              "right): %s. %s" % (len(ids), len(R.ids("S")), "; ".join(texts),
                                  "Same order." if ranks[0] == ranks[1] else
                                  "The orders differ, so no winner is claimed (design rule)."))
            w("")

    # ---- label-free
    w("### Label-free signals")
    w("")
    rs = R.ids("RS")
    meths = ["phone", "clip"] + R.models
    w("Agreement between methods: share of the %d R and S items with the same name (equal normalised name or "
      "head word), crop-full." % len(rs))
    w("")
    w("| | " + " | ".join(LABEL[m] for m in meths) + " |")
    w("|" + " --- |" * (len(meths) + 1))
    for a in meths:
        na = R.names(a, "crop-full")
        cells = []
        for b in meths:
            nb = R.names(b, "crop-full")
            k = sum(vc.same_name(na.get(i), nb.get(i)) for i in rs)
            cells.append("–" if a == b else "%.0f%%" % (100 * k / len(rs)))
        w("| %s | %s |" % (LABEL[a], " | ".join(cells)))
    w("")
    w("Stability of each primary run's names:")
    w("")
    w("| Model | Primary run | Same name in 3 repeats (all inputs) | Batch 4 vs 1 | Batch 16 vs 1 | Cross-view (V) | "
      "crop-320 vs crop-full | ctx-full vs crop-full | ctx-320 vs ctx-full | Says 'one item' (crop-full) | "
      "Unparsed or truncated |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in R.models:
        rows = R.runs[m]["rows"]
        seq = [r for r in rows if r.get("phase") == "seq"]
        by = {}
        for r in seq:
            by.setdefault((r["item"], r["variant"]), {})[r["repeat"]] = vc.normalize(r.get("name") or "")
        stable = sum(1 for d in by.values() if len(d) == 3 and len(set(d.values())) == 1)
        base1 = R.vlm_names(m, "crop-full")
        conc = {}
        for r in rows:
            if r.get("phase") == "concurrency":
                conc.setdefault(r["concurrency"], []).append((r["item"], r.get("name")))
        bs = []
        for c in (4, 16):
            pairs = conc.get(c, [])
            bs.append(pct(sum(1 for i, n in pairs if n == base1.get(i)), len(pairs)))
        cv = [vc.same_name(r.get("name"), base1.get("inv:" + R.items[r["item"]]["entry"]))
              for r in rows if r.get("phase") == "crossview"]
        c320 = R.vlm_names(m, "crop-320")
        cf = R.vlm_names(m, "ctx-full")
        c3 = R.vlm_names(m, "ctx-320")
        sf = R.ids("Sf")
        ones = [((r.get("answer") or {}).get("kind") == "one item") for r in seq
                if r["variant"] == "crop-full" and r["repeat"] == 1]
        bad = sum(1 for r in rows if r.get("phase") in ("seq", "crossview") and not r.get("name"))
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d of %d |" % (
            LABEL[m], R.runs[m]["label"], pct(stable, len(by)), bs[0], bs[1], pct(sum(cv), len(cv)),
            pct(sum(vc.same_name(c320.get(i), base1.get(i)) for i in rs), len(rs)),
            pct(sum(vc.same_name(cf.get(i), base1.get(i)) for i in sf), len(sf)),
            pct(sum(vc.same_name(c3.get(i), cf.get(i)) for i in sf), len(sf)),
            pct(sum(ones), len(ones)), bad, sum(1 for r in rows if r.get("phase") in ("seq", "crossview"))))
    w("")
    extra = [(m, x) for m in R.models for x in R.extra_runs.get(m, [])]
    if extra:
        w("Additional runs against the primary run of the same model (sequential phase, repeat 1, every item and "
          "input; an additional run never enters the quality numbers or the judging pool):")
        w("")
        w("| Model | Primary run | Additional run | Identical normalised name | Same name (label-free rule) |")
        w("| --- | --- | --- | --- | --- |")
        for m, x in extra:
            ident, same, n = run_agreement(R, m, x)
            w("| %s | %s | %s | %s | %s |" % (LABEL[m], R.runs[m]["label"], x["label"], pct(ident, n), pct(same, n)))
        w("")

    # ---- cost
    w("## Cost")
    w("")
    w("Each row is one run at one site, on the GPU named in the row; rows of different sites are not comparable as "
      "the same machine. A model's primary run and its additional runs each have their own rows.")
    w("")
    w("### Per item, one request at a time (client on the same machine as the server; all repeats)")
    w("")
    w("Median [90th percentile]. Latency: request sent to last byte. First token: includes the server's image "
      "preprocessing, vision encoder and prefill. Energy: GPU energy counter over the request (GPU only, not the "
      "CPU or the rest of the machine); above idle subtracts the idle power measured with the model loaded.")
    w("")
    w("| Model | Site (GPU) | Input | n | Latency s | First token s | Decode s | Prompt tokens | Output tokens | "
      "Decode tokens/s | GPU energy J | Above idle J | JPEG bytes | Request bytes |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    cost = {}
    for m, run in R.cost_runs():
        st = run["status"]
        prow = power_rows(run["power"])
        iw = st.get("idle_window")
        idle_w = mean_power(prow, iw[0], iw[1]) if iw and prow else None
        c = cost[(m, run["id"])] = {"idle_w": idle_w, "run": run}
        for v in vc.VARIANTS:
            rows = [r for r in run["rows"] if r.get("phase") == "seq" and r["variant"] == v and not r.get("error")]
            if not rows:
                continue
            lat = [r["latency_s"] for r in rows]
            tps = [(r["completion_tokens"] - 1) / r["decode_s"] for r in rows
                   if r.get("decode_s") and r.get("completion_tokens") and r["completion_tokens"] > 1]
            above = [r["energy_j"] - idle_w * r["latency_s"] for r in rows if idle_w is not None]
            c[v] = {"lat": vc.median(lat), "energy": vc.median([r["energy_j"] for r in rows]),
                    "above": vc.median(above) if above else None, "n": len(rows)}
            w("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                LABEL[m], run["label"], v, len(rows), med_p90(lat), med_p90([r.get("ttft_s") for r in rows]),
                med_p90([r.get("decode_s") for r in rows]), med_p90([r.get("prompt_tokens") for r in rows], 0),
                med_p90([r.get("completion_tokens") for r in rows], 0), med_p90(tps, 0),
                med_p90([r["energy_j"] for r in rows], 1), med_p90(above, 1),
                med_p90([r.get("jpeg_bytes") for r in rows], 0), med_p90([r.get("request_bytes") for r in rows], 0)))
    w("")
    w("### Where the time goes (vLLM server metrics per request, crop-full, medians in seconds)")
    w("")
    w("| Model | Site (GPU) | n | Queue | Prefill (vision encoder + prompt) | Decode | Server end-to-end | "
      "Client latency | Client − server (HTTP, JSON, base64 and image preprocessing) | Sender's base64 encoding ms |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m, run in R.cost_runs():
        rows = [r for r in run["rows"] if r.get("phase") == "seq" and r["variant"] == "crop-full"
                and not r.get("error") and r.get("vllm")]
        if not rows:
            continue
        st = {k: [r["vllm"].get(key) for r in rows if r["vllm"].get(key) is not None] for k, key in STAGES}
        gap = [r["latency_s"] - r["vllm"][STAGES[3][1]] for r in rows if r["vllm"].get(STAGES[3][1]) is not None]
        w("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            LABEL[m], run["label"], len(rows), *(f1(vc.median(st[k]), 3) if st[k] else "–" for k, _ in STAGES),
            f1(vc.median([r["latency_s"] for r in rows]), 3), f1(vc.median(gap), 3) if gap else "–",
            f1(1000 * vc.median([r.get("b64_s") or 0 for r in rows]), 2)))
    w("")
    w("### Start-up, memory and power")
    w("")
    w("| Model | Site (GPU) | Server start to ready s | Weight loading s | Weights GiB | GPU memory held MiB "
      "(peak − before) | KV cache tokens | GPU W before loading | Idle W, model loaded | Mean W during sequential runs |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m, run in R.cost_runs() + [(j, R.runs[j]) for j in R.judges]:
        st = run["status"]
        lg = parse_log(run["log"])
        before = st.get("mem_before") or {}
        peak = st.get("peak_mem_mib") or {}
        held = sum(peak[g] - before.get(str(g), before.get(g, 0)) for g in peak) if peak else None
        prow = power_rows(run["power"])
        iw = st.get("idle_window")
        idle_w = mean_power(prow, iw[0], iw[1]) if iw and prow else None
        rows = [r for r in run["rows"] if r.get("phase") == "seq"]
        busy = mean_power(prow, min(r["t_send"] for r in rows), max(r["t_last"] for r in rows)) if rows and prow else None
        pre = (st.get("shared") or {}).get("preload_idle_window")
        pre_w = mean_power(prow, pre[0], pre[1]) if pre and prow else None
        w("| %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            LABEL[m], " (judge)" if m in JUDGES else "", run["label"], f1(st.get("startup_s"), 0),
            f1(lg.get("load_s") or lg.get("weights_s"), 1), f1(lg.get("weights_gib"), 1),
            "–" if held is None else "%d" % held, lg.get("kv_tokens", "–"), f1(pre_w, 0), f1(idle_w, 0), f1(busy, 0)))
    w("")
    w("On a GPU of its own, vLLM reserves 90% of the GPU's memory at start (KV cache), so the memory held reflects "
      "that setting and the GPU's size; the weights column is what the model itself needs. A shared-GPU session "
      "instead runs with a fixed 1 GiB KV cache and a memory cap that leaves at least 5 GiB unreserved for the other "
      "process; its energy is measured on a GPU that also holds the other (idle) process, against the power measured "
      "right before loading. On the cluster, the Python environment and the weights are read from its network file "
      "system, which makes start-up there much slower than on the workstation; the start-up columns of the two sites "
      "are not comparable." + "".join(
          " The %s reads the Python environment and the weights from its own network file system, so its start-up "
          "column is comparable with neither of the other sites." % SITE_TEXT[s] for _, s in later))
    w("")
    w("### Batched throughput (crop-full, 88 requests = two passes over R and S)")
    w("")
    w("| Model | Site (GPU) | Concurrency | Items per second | GPU J per item | Median latency s |")
    w("| --- | --- | --- | --- | --- | --- |")
    for m, run in R.cost_runs():
        rows = run["rows"]
        seq = [r for r in rows if r.get("phase") == "seq" and r["variant"] == "crop-full" and r["repeat"] in (1, 2)]
        if seq:
            wall = sum(r["latency_s"] for r in seq)
            w("| %s | %s | 1 (sequential, summed) | %s | %s | %s |" % (
                LABEL[m], run["label"], f1(len(seq) / wall), f1(sum(r["energy_j"] for r in seq) / len(seq), 1),
                f1(vc.median([r["latency_s"] for r in seq]))))
        for c in run["conc"]:
            lat = [r["latency_s"] for r in rows if r.get("phase") == "concurrency" and r["concurrency"] == c["concurrency"]]
            w("| %s | %s | %d | %s | %s | %s |" % (LABEL[m], run["label"], c["concurrency"], f1(c["requests"] / c["wall_s"]),
                                                 f1(c["energy_j"] / c["requests"], 1), f1(vc.median(lat))))
    w("")
    w("### Round trip from the Mac through an SSH tunnel (lab workstation only; crop-full, one request at a time)")
    w("")
    w("The cluster's compute nodes are not reachable from the Mac the way the workstation is, so no round trip was "
      "measured there; the design's question about the network is answered by the workstation rows.")
    w("")
    w("| Model | Site (GPU) | Pings from the Mac s | Pings on the workstation s | Items from the Mac s | Same items on "
      "the workstation s | Difference per item s | Sender preparation ms | Request bytes | Response bytes |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m, run in R.cost_runs():
        rt = run["rt"]
        if not rt:
            continue
        pm = [r["latency_s"] for r in rt if r["phase"] == "ping" and not r.get("error")]
        pl = [r["latency_s"] for r in run["pings"] if not r.get("error")]
        it = [r for r in rt if r["phase"] == "item" and not r.get("error")]
        loc = {r["item"]: r["latency_s"] for r in run["rows"]
               if r.get("phase") == "seq" and r["variant"] == "crop-full" and r["repeat"] == 1}
        diff = [r["latency_s"] - loc[r["item"]] for r in it if r["item"] in loc]
        w("| %s | %s | %s (n = %d) | %s (n = %d) | %s (n = %d) | %s | %s | %s | %s | %s |" % (
            LABEL[m], run["label"], med_p90(pm, 3), len(pm), med_p90(pl, 3), len(pl),
            med_p90([r["latency_s"] for r in it]), len(it),
            med_p90([loc[r["item"]] for r in it if r["item"] in loc]), med_p90(diff, 3),
            f1(1000 * vc.median([r["prep_s"] for r in it]), 2) if it else "–",
            med_p90([r["request_bytes"] for r in it], 0), med_p90([r["response_bytes"] for r in it], 0)))
    w("")

    # ---- phone
    if args.data:
        ph = phone_times(args.data)
        w("### The phone's own times, from the log (Pixel 8, same pull and sessions)")
        w("")
        w("| Step | n | Median ms | 90th percentile ms |")
        w("| --- | --- | --- | --- |")
        for k, v in ph["keyframe"].items():
            if v:
                w("| Scan keyframe, %s | %d | %s | %s |" % (k, len(v), f1(vc.median(v), 0), f1(vc.quantile(v, 0.9), 0)))
        for k, v in ph["register"].items():
            if v:
                w("| Tap registration, %s | %d | %s | %s |" % (k, len(v), f1(vc.median(v), 0), f1(vc.quantile(v, 0.9), 0)))
        if ph["verify"]:
            w("| Verify capture, totalMs | %d | %s | %s |" % (len(ph["verify"]), f1(vc.median(ph["verify"]), 0),
                                                           f1(vc.quantile(ph["verify"], 0.9), 0)))
        w("")
        for s in ph["scans"]:
            if (s.get("entries") or {}).get("total"):
                w("Scan `%s`: %d keyframes in %.1f s, %d entries." % (s.get("scan"), s.get("keyframes", 0),
                                                                    (s.get("durationMs") or 0) / 1000,
                                                                    s["entries"]["total"]))
        w("")
    return L, cost


def plot_judge(R):
    """The judge of the plot: the primary judge when it judged every S-frame item, else the first other judge
    that did (named in the plot and the note), else None."""
    sf = R.ids("Sf")
    for j in [PRIMARY_JUDGE] + [x for x in JUDGES if x != PRIMARY_JUDGE]:
        if j in R.judges and len(judged_ids(R, j, sf)) == len(sf):
            return j
    return None


def plot_rows(R, cost):
    """Rows of the quality-against-cost plot: (model, input, site label, run id, strict k, n, seconds, joules).
    Quality: the plot judge's strict score on S-frame for the primary run's names; cost: every run of the
    model, one row per site (never combined)."""
    J = plot_judge(R)
    if not J:
        return []
    ids = R.ids("Sf")
    rows = []
    for (m, rid), c in cost.items():
        for v in vc.VARIANTS:
            if v not in c:
                continue
            k, _, _, _, n = judged(R, J, R.vlm_names(m, v), ids)
            rows.append((m, v, c["run"]["label"], rid, k, n, c[v]["lat"], c[v]["energy"]))
    order = {m: i for i, m in enumerate(CONTESTANTS)}
    rows.sort(key=lambda r: (SITE_ORDER.get(r[3].split("/")[0], 9), order[r[0]], vc.VARIANTS.index(r[1])))
    return rows


# Marker style of each site in the plot: filled (the first cluster: all four contestants on one GPU type),
# open (the lab workstation), half-filled (the second cluster)
SITE_MARKER = {"cluster": "filled", "workstation": "open", "cluster2": "half-filled"}


def plot(R, cost, path):
    """Primary-judge strict score on S-frame against seconds and joules per item. Filled markers: the
    cluster runs (all four contestants on one GPU type); open markers: the lab workstation's runs;
    half-filled markers: the second cluster's runs."""
    rows = plot_rows(R, cost)
    if not rows:
        return False
    J = plot_judge(R)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = R.ids("Sf")
    colors = {"model_a": "#2a78d6", "model_b": "#eb6834", "model_c": "#1baf7a", "model_d": "#eda100"}
    markers = {"crop-full": "o", "crop-320": "s", "ctx-full": "^", "ctx-320": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    hw = {}
    for ax, idx, xl in ((axes[0], 6, "Seconds per item (median, one request at a time)"),
                        (axes[1], 7, "GPU joules per item (median, one request at a time)")):
        for b, ls in (("phone", "--"), ("clip", ":"), ("yoloe11s", "-."), ("owlv2", (0, (1, 3)))):
            nm = R.names(b, "keyframes" if b in ("yoloe11s", "owlv2") else "crop-full")
            if nm:
                k = judged(R, J, nm, ids)[0]
                ax.axhline(100 * k / len(ids), color="#888888", lw=1, ls=ls)
                if ax is axes[1]:
                    ax.text(1.01, 100 * k / len(ids), LABEL[b], transform=ax.get_yaxis_transform(), fontsize=7,
                            va="center", color="#555555")
        for m, v, label, rid, k, n, lat, en in rows:
            site = rid.split("/")[0]
            hw.setdefault(site, label)
            lo, hi = vc.wilson(k, n)
            y = 100 * k / n
            style = SITE_MARKER.get(site, "open")
            filled = style == "filled"
            kw = {"fillstyle": "bottom", "markerfacecoloralt": "white"} if style == "half-filled" else {}
            ax.errorbar(lat if idx == 6 else en, y, yerr=[[y - 100 * lo], [100 * hi - y]], fmt=markers[v],
                        color=colors[m], mfc=colors[m] if style != "open" else "white", ms=8,
                        mec="white" if filled else colors[m], mew=1.2, elinewidth=1, alpha=0.9,
                        label="%s, %s" % (LABEL[m], v) if ax is axes[0] and filled else None, **kw)
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter
        for axis_fmt in (ax.xaxis.set_major_formatter, ax.xaxis.set_minor_formatter):
            axis_fmt(FuncFormatter(lambda x, _: "%g" % x))
        ax.set_xlabel(xl, fontsize=9)
        ax.grid(True, color="#e5e5e5", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Named right, %% of S-frame (n = %d), model-judged" % len(ids), fontsize=9)
    axes[0].set_ylim(0, 100)
    fig.legend(loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.14))
    sub = "; ".join("%s: %s markers" % (hw[s], SITE_MARKER.get(s, "open")) for s in sorted(hw))
    fig.suptitle("Naming quality (model-judged by %s%s) against cost per site; grey lines: baselines\n%s" % (
        LABEL[J], "" if J == PRIMARY_JUDGE else ", not the primary judge, which could not judge these items", sub),
        fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 0.93, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def cmd_report(args):
    R = Results(args.out)
    L, cost = report(R, args)
    ok = plot(R, cost, os.path.join(args.out, "quality_cost.png"))
    L.append("## Plot")
    L.append("")
    if ok:
        J = plot_judge(R)
        L.append("`analysis/out/vlm/quality_cost.png` (git-ignored; regenerate with this script): the strict score "
                 "given by %s on S-frame for each contestant and input (names from the primary run) against the "
                 "median seconds and GPU joules per item of every run of that model, one point per site.%s The "
                 "numbers behind it (model-judged quality; cost per site, never combined):" % (
                     LABEL[J], "" if J == PRIMARY_JUDGE else (
                         " The design's plot uses the primary judge (%s), which has no verdict on any S-frame item "
                         "(see above), so this plot uses %s; recorded in the changes." % (LABEL[PRIMARY_JUDGE], LABEL[J]))))
        L.append("")
        L.append("| Model | Input | Cost measured on | Named right, S-frame (model-judged) | Seconds per item | "
                 "GPU J per item |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for m, v, label, rid, k, n, lat, en in plot_rows(R, cost):
            L.append("| %s | %s | %s | %s | %s | %s |" % (LABEL[m], v, label, pct(k, n), f1(lat), f1(en, 1)))
        L.append("")
        L.append("Baselines on S-frame (model-judged by %s): " % LABEL[J] + "; ".join(
            "%s %s" % (LABEL[b], pct(judged(R, J, R.names(b, "keyframes" if b in ("yoloe11s", "owlv2")
                                                          else "crop-full"), R.ids("Sf"))[0], len(R.ids("Sf"))))
            for b in BASELINES if R.names(b, "keyframes" if b in ("yoloe11s", "owlv2") else "crop-full")) + ".")
    else:
        L.append("Not drawn (needs a judge with a verdict on every S-frame item and at least one contestant).")
    L.append("")
    L.append("## Changes after the design was fixed")
    L.append("")
    L.append(args.changes)
    L.append("")
    md = "\n".join(L)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--out", default=OUT)
    p = sub.add_parser("pool-inputs")
    p.add_argument("--out", default=OUT)
    p = sub.add_parser("report")
    p.add_argument("--out", default=OUT)
    p.add_argument("--data", action="append", help="pulled data folder, for the phone's own times")
    p.add_argument("--md")
    p.add_argument("--changes", default="None.")
    p.add_argument("--changes-file", help="read the 'changes' section from this file")
    args = ap.parse_args(argv)
    if getattr(args, "changes_file", None):
        with open(args.changes_file) as f:
            args.changes = f.read().strip()
    {"pool": cmd_pool, "pool-inputs": cmd_pool_inputs, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
