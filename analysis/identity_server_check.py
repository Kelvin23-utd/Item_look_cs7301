#!/usr/bin/env python3
"""Would a stronger image encoder off the phone decide the audited same-object cases, and do several
real views help over one? A first, cheap server-side comparison (encoders only; no vision-language model).

    analysis/.venv-bench/bin/python -I analysis/identity_server_check.py backups/phone_files_2026-11-19_run3_1958
    python3 -I analysis/identity_server_check.py <pull> --encoders none     # the phone's embedding only
    ... --cases <cases.json> --out <folder>

Input: one pulled data folder of the v0.8 study build (FindBack/docs/PLAN_v0.8_study_logging.md): the
observation logs `obs/<scan>.jsonl` (per kept detection: the phone's embedding `emb`, `quality` = box
area x score, `mapPoint`, the matcher's `decision.entry`) and the retained full-resolution view crops
`views/<scan>/<obs>.jpg`. Observations are read with analysis/identity_replay.py's loader.

Definitions (fixed before any result was seen; nothing is tuned on the cases):

* group: the observations the app assigned (`decision.entry`) to one entry within one scan, tentative
  entries dropped at the scan's end included. A group's views are its observations whose view is
  retained at the scan's end (some observations have none: person in frame, not complementary).
* case: a query (the later sighting: an entry's group in the test scan, or given observations) and the
  right group(s) from the audited facts (CASES below). Candidates: every group of every scan that started
  before the test scan.
* encoders: the phone's logged embedding over every observation (`phone_all`) and over the observations
  with a retained view only (`phone`, the same crops the other encoders see), and each stronger encoder of
  analysis/bench_match.py whose weights are in analysis/.bench-cache (`mnv3l`, `dinov2s`,
  `mobileclip2s2`), run on the view JPEGs. Every score is a cosine of unit vectors.
* modes: `one` = the query's best view (largest quality) against the candidate's best view; `max` and
  `median` = of the cosines of every query view against every candidate view. Ranking uses `max` (the
  rule of identity_replay.score_matrix); ranks by `one` and by `median` are reported as well.
* rank: 1 + the number of wrong candidates scoring at least as high as the best right group (ties count
  against the right answer); best other = the highest-scoring wrong candidate; margin = right - best other.
* clear negatives (calibration, no moved case used): pairs of groups of different entries within one scan
  in which every observation of one lies more than FAR_M (identity_replay, 1.0 m) from every observation of
  the other, the moved cases' query groups left out. Pairs share groups, so they are not independent.

Output (entry ids, no positions or images): analysis/out/identity_server/ (git-ignored): `report.txt`,
`tables.md`, `result.json`. The ranking code is standard library only (analysis/test_identity_server_check.py);
the encoders need analysis/.venv-bench. Times are Mac CPU times, not server measurements.
"""
import argparse
import collections
import json
import math
import operator
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import identity_replay as ir  # noqa: E402
import study_check as sc  # noqa: E402

FAR_M = ir.DEFAULT.FAR_M
RANK_MODE = "max"
MODES = ("one", "max", "median")
MODE_LABEL = {"one": "one view against one view", "max": "several views, max (ranking)",
              "median": "several views, median"}
PHONE_KEYS = ("phone_all", "phone")
DEFAULT_ENCODERS = ("mnv3l", "dinov2s", "mobileclip2s2")
LABEL = {
    "phone_all": "Phone (MobileNetV3-small, logged), every observation",
    "phone": "Phone (MobileNetV3-small, logged), observations with a view",
    "mnv3l": "MobileNetV3-large",
    "dinov2s": "DINOv2 ViT-S/14",
    "mobileclip2s2": "MobileCLIP2-S2 (image encoder)",
    "ours": "MobileNetV3-small (app model, offline from the view JPEG)",
}
SHORT = {"phone_all": "Phone, all obs", "phone": "Phone, views", "mnv3l": "MobileNetV3-L",
         "dinov2s": "DINOv2-S/14", "mobileclip2s2": "MobileCLIP2-S2"}
WARMUP = 3

# The audited same-object facts (visual audit of the retained views; notes 2026-11-19_smoke_audit.md,
# 2026-11-19_second_two_pass_run.md, 2026-11-19_three_scan_run.md). Everything else is unknown.
CASES = [
    {"name": "moved 1", "kind": "moved", "test": "03f9e780", "query_obs": ["03f9e780-48-1"],
     "right": [["e4144f49", "e4af587b"]]},
    {"name": "moved 2", "kind": "moved", "test": "b6221dcb", "query_entry": "f9a56c12",
     "right": [["0c3c261f", "e98ae991"]]},
    {"name": "moved 3", "kind": "moved", "test": "fb65023c", "query_entry": "8c1233db",
     "right": [["0dfd8938", "a1be09aa"], ["fb0bf2d9", "30452d65"]]},
    {"name": "unmoved 1", "kind": "unmoved", "test": "fb0bf2d9", "query_entry": "30452d65",
     "right": [["0dfd8938", "a1be09aa"]]},
    {"name": "unmoved 2", "kind": "unmoved", "test": "fb0bf2d9", "query_entry": "fe39d6b3",
     "right": [["0dfd8938", "8d3d92de"]]},
    {"name": "unmoved 3", "kind": "unmoved", "test": "fb0bf2d9", "query_entry": "9983a5b5",
     "right": [["0dfd8938", "30452d65"]]},
    {"name": "unmoved 4", "kind": "unmoved", "test": "fb0bf2d9", "query_entry": "dd8e68a0",
     "right": [["0dfd8938", "9d921407"]]},
]


# ============================================================================ pure part (standard library)

def dot(a, b):
    return sum(map(operator.mul, a, b))


class Sim:
    """Memoised cosine between unit vectors stored by key."""

    def __init__(self, vectors):
        self.v, self.memo = vectors, {}

    def __call__(self, a, b):
        k = (a, b) if a <= b else (b, a)
        r = self.memo.get(k)
        if r is None:
            r = self.memo[k] = dot(self.v[a], self.v[b])
        return r


def best_item(items):
    """Key of the item with the largest quality; the first one on ties. items: [(key, quality)]."""
    best = None
    for k, q in items:
        if best is None or q > best[1]:
            best = (k, q)
    return best[0] if best else None


def score_pair(q_items, c_items, sim):
    """Scores of a query side against a candidate side ([(key, quality)] each): `one` (best view against
    best view), `max` and `median` of every pairwise cosine. None when either side is empty."""
    if not q_items or not c_items:
        return None
    vals = [sim(a, b) for a, _ in q_items for b, _ in c_items]
    return {"one": sim(best_item(q_items), best_item(c_items)), "max": max(vals),
            "median": statistics.median(vals), "pairs": len(vals)}


def rank_right(scores, right):
    """Rank of the right answer. scores: {candidate: float or None (not scorable)}; right: the
    candidates that count as right. Returns rank (1 + wrong candidates scoring at least as high as the
    best right one; None when no right candidate is scorable), its score, the best other (wrong)
    candidate, its score, margin = right - other, and the rank of each right candidate."""
    scored = [(g, s) for g, s in scores.items() if s is not None]
    wrong = sorted(((s, g) for g, s in scored if g not in right), key=lambda x: (-x[0], x[1]))
    rights = sorted(((s, g) for g, s in scored if g in right), key=lambda x: (-x[0], x[1]))
    out = {"candidates": len(scored), "unscored": len(scores) - len(scored), "rank": None, "score": None,
           "right": None, "other": wrong[0][1] if wrong else None, "other_score": wrong[0][0] if wrong else None,
           "margin": None, "right_ranks": {}}
    if not rights:
        return out
    s_r, g_r = rights[0]
    out.update(rank=1 + sum(1 for s, _ in wrong if s >= s_r), score=s_r, right=g_r)
    if wrong:
        out["margin"] = s_r - wrong[0][0]
    out["right_ranks"] = {g: 1 + sum(1 for s2, _ in wrong if s2 >= s) for s, g in rights}
    return out


def percentile(xs, q):
    """Linear interpolation between order statistics (numpy's default); q in [0, 100]."""
    if not xs:
        return None
    ys = sorted(xs)
    pos = (len(ys) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def share_at_or_above(xs, x):
    return None if not xs or x is None else sum(1 for v in xs if v >= x) / float(len(xs))


def summary(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "p95": percentile(xs, 95),
            "p99": percentile(xs, 99), "max": max(xs)}


class Ob:
    """One observation as this check reads it."""

    __slots__ = ("id", "scan", "entry", "label", "quality", "map_point", "emb", "view", "view_bytes")

    def __init__(self, oid, scan, entry, label, quality, map_point, emb, view=None, view_bytes=0):
        self.id, self.scan, self.entry, self.label, self.quality = oid, scan, entry, label, quality
        self.map_point, self.emb, self.view, self.view_bytes = tuple(map_point), emb, view, view_bytes


class Group:
    """The observations the app assigned to one entry within one scan."""

    def __init__(self, scan, entry, obs, tentative=False):
        self.scan, self.entry, self.obs, self.tentative = scan, entry, list(obs), tentative
        self.key = group_key(scan, entry)

    @property
    def viewed(self):
        return [o for o in self.obs if o.view]

    @property
    def label(self):
        c = collections.Counter(o.label for o in self.obs if o.label)
        return c.most_common(1)[0][0] if c else None


def group_key(scan, entry):
    return "%s:%s" % (scan, entry)


def make_groups(obs_by_scan, tentative_by_scan=None):
    """obs_by_scan: ordered {scan: [Ob]} -> ordered {key: Group}, in scan order then first observation."""
    out = collections.OrderedDict()
    for scan, obs in obs_by_scan.items():
        per = collections.OrderedDict()
        for o in obs:
            per.setdefault(o.entry, []).append(o)
        for entry, os_ in per.items():
            g = Group(scan, entry, os_, entry in (tentative_by_scan or {}).get(scan, ()))
            out[g.key] = g
    return out


def side(obs, encoder):
    """[(key, quality)] an encoder can use: every observation for phone_all, else those with a view."""
    return [(o.id, o.quality) for o in obs if encoder == "phone_all" or o.view]


def far_pairs(groups, far_m=FAR_M, exclude=()):
    """Pairs (key, key) of groups of one scan and different entries in which every observation of one
    lies more than far_m from every observation of the other."""
    by_scan = collections.OrderedDict()
    for g in groups:
        if g.key not in exclude:
            by_scan.setdefault(g.scan, []).append(g)
    out = []
    for gs in by_scan.values():
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                a, b = gs[i], gs[j]
                if a.entry == b.entry:
                    continue
                if min(ir.dist(x.map_point, y.map_point) for x in a.obs for y in b.obs) > far_m:
                    out.append((a.key, b.key))
    return out


def resolve_case(case, groups, obs_by_id, scan_order):
    """(query observations, candidate groups, right keys) of a case; candidates are the groups of
    the scans before the test scan in scan_order."""
    test = case["test"]
    if case.get("query_obs"):
        query = [obs_by_id[i] for i in case["query_obs"]]
    else:
        query = groups[group_key(test, case["query_entry"])].obs
    earlier = set(scan_order[:scan_order.index(test)])
    cands = [g for g in groups.values() if g.scan in earlier]
    right = {group_key(s, e) for s, e in case["right"]}
    missing = right - {g.key for g in cands}
    if missing:
        raise ValueError("%s: right group(s) %s are not candidates" % (case["name"], sorted(missing)))
    return query, cands, right


def case_costs(query, cands):
    """Crops a server step would need: the query side (uploaded at test time) and the candidate side
    (kept from the history scans), one view per side and every view."""
    qv = [o for o in query if o.view]
    qb = max(qv, key=lambda o: o.quality) if qv else None
    cv = [g for g in cands if g.viewed]
    best = [max(g.viewed, key=lambda o: o.quality) for g in cv]
    allv = [o for g in cv for o in g.viewed]
    return {"query_one": {"n": 1 if qb else 0, "bytes": qb.view_bytes if qb else 0, "ids": [qb.id] if qb else []},
            "query_all": {"n": len(qv), "bytes": sum(o.view_bytes for o in qv), "ids": [o.id for o in qv]},
            "cand_one": {"n": len(best), "bytes": sum(o.view_bytes for o in best), "ids": [o.id for o in best]},
            "cand_all": {"n": len(allv), "bytes": sum(o.view_bytes for o in allv), "ids": [o.id for o in allv]}}


def evaluate(cases, groups, obs_by_id, scan_order, sims, moved_exclude=None):
    """Every case under every encoder in `sims` ({encoder: Sim}). Returns (per-case results, negatives)."""
    if moved_exclude is None:
        moved_exclude = set()
        for case in cases:
            if case["kind"] == "moved":
                q, _, _ = resolve_case(case, groups, obs_by_id, scan_order)
                moved_exclude |= {group_key(o.scan, o.entry) for o in q}
    pairs = far_pairs(groups.values(), FAR_M, moved_exclude)
    neg, neg_top = {}, {}
    for enc, sim in sims.items():
        vals = {m: [] for m in MODES}
        for a, b in pairs:
            s = score_pair(side(groups[a].obs, enc), side(groups[b].obs, enc), sim)
            if s is not None:
                for m in MODES:
                    vals[m].append((s[m], a, b))
        neg[enc] = {m: [x[0] for x in vals[m]] for m in MODES}
        neg_top[enc] = {m: [{"score": s, "a": a, "b": b, "labels": [groups[a].label, groups[b].label]}
                            for s, a, b in sorted(vals[m], key=lambda x: (-x[0], x[1], x[2]))[:5]] for m in MODES}
    neg_info = {"pairs": len(pairs), "excluded_groups": sorted(moved_exclude),
                "scans": sorted({groups[a].scan for a, _ in pairs}, key=scan_order.index),
                "groups": len({k for p in pairs for k in p}), "top": neg_top}
    results = []
    for case in cases:
        query, cands, right = resolve_case(case, groups, obs_by_id, scan_order)
        r = {"case": case, "query": [o.id for o in query], "query_views": [o.id for o in query if o.view],
             "candidates": len(cands), "candidates_with_view": sum(1 for g in cands if g.viewed),
             "right": sorted(right), "right_sizes": {k: [len(groups[k].obs), len(groups[k].viewed)] for k in sorted(right)},
             "costs": case_costs(query, cands), "by_encoder": {}}
        qkeys = {group_key(o.scan, o.entry) for o in query}
        for enc, sim in sims.items():
            qs = side(query, enc)
            sc_ = {g.key: score_pair(qs, side(g.obs, enc), sim) for g in cands}
            per = {"query_items": len(qs)}
            for m in MODES:
                rr = rank_right({k: (v[m] if v else None) for k, v in sc_.items()}, right)
                rr["share_negatives_at_or_above"] = share_at_or_above(neg[enc][m], rr["score"])
                top = sorted(((v[m], k) for k, v in sc_.items() if v), key=lambda x: (-x[0], x[1]))[:5]
                rr["top"] = [{"group": k, "score": s, "right": k in right, "label": groups[k].label,
                              "obs": len(groups[k].obs), "views": len(groups[k].viewed),
                              "tentative": groups[k].tentative,
                              "same_entry_as_query": any(groups[k].entry == group_key_entry(q) for q in qkeys),
                              "same_entry_as_right": any(groups[k].entry == group_key_entry(x) for x in right)}
                             for s, k in top]
                if rr["other"]:
                    o = groups[rr["other"]]
                    rr["other_info"] = {"label": o.label, "obs": len(o.obs), "views": len(o.viewed),
                                        "tentative": o.tentative}
                per[m] = rr
            per["scores"] = {k: v for k, v in sc_.items() if v}
            r["by_encoder"][enc] = per
        results.append(r)
    return results, neg, neg_info


def group_key_entry(key):
    return key.split(":", 1)[1]


# ============================================================================ reading a pull

def load_pull(folder):
    """(pull, scan order, {key: Group}, {obs id: Ob}, info) with identity_replay's loader."""
    pull = sc.Pull(folder)
    order, obs_by_scan, tentative, info = [], collections.OrderedDict(), {}, collections.Counter()
    models = collections.Counter()
    for scan in pull.study_scans():
        data = ir.load_scan(pull, scan)
        if not data.observations:
            continue
        lines = {o.get("obs"): o for o in pull.observations(scan)}
        obs = []
        for o in data.observations:
            line = lines[o.id]
            models[(line.get("embModel"), line.get("embDim"))] += 1
            view, nbytes = None, 0
            if o.retained:
                p = pull.path(line["view"])
                if os.path.exists(p):
                    view, nbytes = p, os.path.getsize(p)
                else:
                    info["retained view file missing"] += 1
            q = line.get("quality")
            if q is None:
                b = line.get("box") or (0, 0, 0, 0)
                q = (b[2] - b[0]) * (b[3] - b[1]) * (line.get("score") or 0.0)
            obs.append(Ob(o.id, scan, (o.decision or {}).get("entry"), o.label, float(q), o.map_point, o.emb, view, nbytes))
        order.append(scan)
        obs_by_scan[scan] = obs
        tentative[scan] = set(data.dropped)
        info["observations"] += len(obs)
        info["with view"] += sum(1 for o in obs if o.view)
    groups = make_groups(obs_by_scan, tentative)
    obs_by_id = {o.id: o for obs in obs_by_scan.values() for o in obs}
    info = dict(info)
    info["embModel"] = {"%s/%s" % k: v for k, v in models.items()}
    return pull, order, groups, obs_by_id, info


def query_sightings(pull, case, obs_by_id):
    """How identity_replay's sightings of the test scan hold the query's observations."""
    q = set(case.get("query_obs") or [])
    data = ir.load_scan(pull, case["test"])
    if not q:
        q = {o.id for o in data.observations if (o.decision or {}).get("entry") == case["query_entry"]}
    out = []
    for s in ir.build_sightings(data):
        ids = {o.id for o in s.obs}
        if ids & q:
            out.append({"sighting": s.id, "obs": len(ids), "query_obs": len(ids & q)})
    return out


# ============================================================================ encoders (bench environment)

def run_encoders(keys, views, log):
    """Features of every view crop for each encoder that loads. views: [(obs id, path)].
    Returns ({encoder: {obs id: unit vector list}}, {encoder: timing}, {encoder: reason skipped})."""
    feats, timing, skipped = {}, {}, {}
    if not keys:
        return feats, timing, skipped
    try:
        import bench_common as bc
        import bench_match as bm        # sets the Hugging Face libraries offline before any model import
    except Exception as e:  # noqa: BLE001
        return feats, timing, {k: "bench environment not available (%s)" % e.__class__.__name__ for k in keys}
    imgs, t_dec = {}, []
    for oid, path in views:
        t0 = time.perf_counter()
        imgs[oid] = bc.load_rgb(path)
        t_dec.append((time.perf_counter() - t0) * 1000.0)
    timing["_decode"] = {"n": len(t_dec), "median_ms": statistics.median(t_dec) if t_dec else None}
    for key in keys:
        t0 = time.perf_counter()
        try:
            m = bm.make_method(key)
        except Exception as e:  # noqa: BLE001
            skipped[key] = "%s: %s" % (e.__class__.__name__, str(e).splitlines()[0][:200] if str(e) else "")
            log("  %s skipped: %s" % (key, skipped[key]))
            continue
        load_s = time.perf_counter() - t0
        for oid, _ in views[:WARMUP]:
            m.features(imgs[oid])
        f, ms = {}, {}
        for oid, _ in views:
            t1 = time.perf_counter()
            v = m.features(imgs[oid])
            ms[oid] = (time.perf_counter() - t1) * 1000.0
            f[oid] = [float(x) for x in v]
        feats[key] = f
        xs = list(ms.values())
        timing[key] = {"n": len(xs), "median_ms": statistics.median(xs), "p90_ms": percentile(xs, 90),
                       "max_ms": max(xs), "load_s": load_s, "dim": len(next(iter(f.values()))) if f else None,
                       "params": getattr(m, "params", None), "file_bytes": getattr(m, "file_bytes", None),
                       "per_crop_ms": ms, "threads": getattr(bm, "TORCH_THREADS", None)}
        log("  %s: %d crops, median %.1f ms per crop (Mac CPU), loaded in %.1f s" % (key, len(xs), timing[key]["median_ms"], load_s))
    return feats, timing, skipped


# ============================================================================ report

def f3(x):
    return "-" if x is None else "%.3f" % x


def fm(x):
    return "-" if x is None else "%+.3f" % x


def kb(n):
    return "%.1f KB" % (n / 1000.0)


def cell(rr):
    if rr["rank"] is None:
        return "no view"
    return "%d/%d: %s vs %s (%s)" % (rr["rank"], rr["candidates"], f3(rr["score"]), f3(rr["other_score"]), fm(rr["margin"]))


def markdown(results, neg, neg_info, encs, timing, skipped, repro, info, args):
    L = []
    w = L.append
    w("Generated by `analysis/identity_server_check.py` from one private pull (%d study scans with observations, "
      "%d observations, %d with a retained view). Cell: rank of the right group / scorable candidate groups: right "
      "score vs best other group (margin). Ranks count ties against the right answer." % (
          len(neg_info["all_scans"]), info["observations"], info["with view"]))
    w("")
    w("### Views per case")
    w("")
    w("| Case | Test scan | Query: observations / with a view | Right group(s): observations / with a view | Candidate groups / with a view | identity_replay sightings holding the query |")
    w("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        c = r["case"]
        rs = "; ".join("`%s` in `%s`: %d / %d" % (k.split(":")[1], k.split(":")[0], v[0], v[1]) for k, v in r["right_sizes"].items())
        sg = ", ".join("%d of %d obs" % (s["query_obs"], s["obs"]) for s in r.get("sightings") or []) or "-"
        w("| %s | `%s` | %d / %d | %s | %d / %d | %s |" % (c["name"], c["test"], len(r["query"]), len(r["query_views"]), rs,
                                                         r["candidates"], r["candidates_with_view"], sg))
    for m in MODES:
        w("")
        w("### %s" % MODE_LABEL[m][0].upper() + MODE_LABEL[m][1:])
        w("")
        w("| Case | " + " | ".join(SHORT[e] for e in encs) + " |")
        w("| --- |" + " --- |" * len(encs))
        for r in results:
            w("| %s | " % r["case"]["name"] + " | ".join(cell(r["by_encoder"][e][m]) for e in encs) + " |")
    w("")
    w("### Share of clear negatives scoring at or above the right group (several views, max)")
    w("")
    w("| Case | " + " | ".join(SHORT[e] for e in encs) + " |")
    w("| --- |" + " --- |" * len(encs))
    for r in results:
        row = []
        for e in encs:
            x = r["by_encoder"][e][RANK_MODE]["share_negatives_at_or_above"]
            row.append("-" if x is None else "%.3f (n = %d)" % (x, len(neg[e][RANK_MODE])))
        w("| %s | %s |" % (r["case"]["name"], " | ".join(row)))
    w("")
    w("### Top three candidate groups of each moved case (several views, max)")
    w("")
    w("Group = entry in scan (majority detector label, observations / views; t = tentative entry dropped at the scan's "
      "end); R = an audited right group. Whether the other groups show the same object is not known.")
    w("")
    w("| Case | " + " | ".join(SHORT[e] for e in encs) + " |")
    w("| --- |" + " --- |" * len(encs))
    for r in results:
        if r["case"]["kind"] != "moved":
            continue
        row = []
        for e in encs:
            row.append("<br>".join("%s`%s` in `%s` (%s, %d/%d%s) %s" % (
                "R " if t["right"] else "", t["group"].split(":")[1], t["group"].split(":")[0], t["label"], t["obs"], t["views"],
                ", t" if t["tentative"] else "", f3(t["score"])) for t in r["by_encoder"][e][RANK_MODE]["top"][:3]))
        w("| %s | %s |" % (r["case"]["name"], " | ".join(row)))
    w("")
    w("### Reading a threshold off the clear negatives")
    w("")
    w("Right scores strictly above the clear negatives' 99th percentile and above their maximum (values in the "
      "calibration table below).")
    w("")
    nu = sum(1 for r in results if r["case"]["kind"] == "unmoved")
    nm = sum(1 for r in results if r["case"]["kind"] == "moved")
    w("| Encoder | Mode | Unmoved above p99 (of %d) | Moved above p99 (of %d) | Unmoved above max (of %d) | Moved above max (of %d) |" % (nu, nm, nu, nm))
    w("| --- | --- | --- | --- | --- | --- |")
    for e in encs:
        for m in MODES:
            s = summary(neg[e][m])
            row = []
            for thr in ("p99", "max"):
                for kind in ("unmoved", "moved"):
                    sc_ = [r["by_encoder"][e][m]["score"] for r in results if r["case"]["kind"] == kind]
                    row.append(sum(1 for x in sc_ if x is not None and s.get(thr) is not None and x > s[thr]))
            w("| %s | %s | %d | %d | %d | %d |" % (SHORT[e], m, row[0], row[1], row[2], row[3]))
    w("")
    w("### Highest clear negatives (several views, max)")
    w("")
    w("| Encoder | Top three pairs: groups (majority detector labels) score |")
    w("| --- | --- |")
    for e in encs:
        w("| %s | %s |" % (SHORT[e], "<br>".join("`%s` / `%s` (%s / %s) %s" % (
            t["a"], t["b"], t["labels"][0], t["labels"][1], f3(t["score"])) for t in neg_info["top"][e][RANK_MODE][:3])))
    w("")
    w("### Calibration: clear negatives and the unmoved positives")
    w("")
    w("Clear negatives: %d pairs of groups of different entries within one scan, every observation of one more than "
      "%.1f m from every observation of the other, from %d groups in %d scans; the moved cases' query groups (%s) left "
      "out. A pair is scored only when both groups have something the encoder can use, so n differs by encoder." % (
          neg_info["pairs"], FAR_M, neg_info["groups"], len(neg_info["scans"]),
          ", ".join("`%s`" % k for k in neg_info["excluded_groups"])))
    w("")
    w("| Encoder | Mode | Negatives n | median | 95th pct | 99th pct | max | Unmoved right scores (n = %d) | Moved right scores (n = %d) |" % (
        sum(1 for r in results if r["case"]["kind"] == "unmoved"), sum(1 for r in results if r["case"]["kind"] == "moved")))
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for e in encs:
        for m in MODES:
            s = summary(neg[e][m])
            un = [r["by_encoder"][e][m]["score"] for r in results if r["case"]["kind"] == "unmoved"]
            mo = [r["by_encoder"][e][m]["score"] for r in results if r["case"]["kind"] == "moved"]
            w("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
                SHORT[e], m, s["n"], f3(s.get("median")), f3(s.get("p95")), f3(s.get("p99")), f3(s.get("max")),
                ", ".join(f3(x) for x in un), ", ".join(f3(x) for x in mo)))
    w("")
    w("### Cost of a server step per case")
    w("")
    w("Crops are the retained view JPEGs (full resolution, quality 90). Query side: uploaded at test time. Candidate "
      "side: the history groups that have a view (a server would have received them with their own scans). Mac time: "
      "the sum of the measured encode times of those crops on this Mac (CPU), not a server measurement; JPEG decoding "
      "not included.")
    w("")
    enc_t = [e for e in encs if e not in PHONE_KEYS]
    w("| Case | Query, one view | Query, every view | Candidates, one view each | Candidates, every view | " +
      " | ".join("%s Mac ms, one / every" % SHORT[e] for e in enc_t) + " |")
    w("| --- | --- | --- | --- | --- |" + " --- |" * len(enc_t))
    for r in results:
        c = r["costs"]
        row = ["%d, %s" % (c[k]["n"], kb(c[k]["bytes"])) for k in ("query_one", "query_all", "cand_one", "cand_all")]
        for e in enc_t:
            ms = timing[e]["per_crop_ms"]
            one = sum(ms[i] for i in c["query_one"]["ids"] + c["cand_one"]["ids"])
            al = sum(ms[i] for i in c["query_all"]["ids"] + c["cand_all"]["ids"])
            row.append("%.0f / %.0f" % (one, al))
        w("| %s | %s |" % (r["case"]["name"], " | ".join(row)))
    w("")
    w("Mac encode time of the query's crops alone (the cost per query when the history crops were encoded when "
      "their scans arrived), ms, one view / every view:")
    w("")
    w("| Case | " + " | ".join(SHORT[e] for e in enc_t) + " |")
    w("| --- |" + " --- |" * len(enc_t))
    for r in results:
        c = r["costs"]
        w("| %s | %s |" % (r["case"]["name"], " | ".join("%.1f / %.1f" % (
            sum(timing[e]["per_crop_ms"][i] for i in c["query_one"]["ids"]),
            sum(timing[e]["per_crop_ms"][i] for i in c["query_all"]["ids"])) for e in enc_t)))
    w("")
    w("### Encoders on this Mac")
    w("")
    w("CPU only, %s torch threads, batch size 1, after %d warm-up crops; every retained view of the pull (n below). "
      "JPEG decode median %s ms per crop (n = %d)." % (
          next((timing[e].get("threads") for e in enc_t if timing.get(e)), "-"), WARMUP,
          "%.2f" % timing["_decode"]["median_ms"] if timing.get("_decode", {}).get("median_ms") is not None else "-",
          timing.get("_decode", {}).get("n", 0)))
    w("")
    w("| Encoder | Dimensions | Parameters | Crops n | Median ms per crop | 90th pct ms | Load s |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for e in [x for x in ["ours"] + list(enc_t) if x in timing]:
        t = timing[e]
        w("| %s | %s | %s | %d | %.1f | %.1f | %.1f |" % (LABEL[e], t["dim"], "%.2f M" % (t["params"] / 1e6) if t.get("params") else "-",
                                                      t["n"], t["median_ms"], t["p90_ms"], t["load_s"]))
    if skipped:
        w("")
        w("Not run: " + "; ".join("%s (%s)" % (k, v) for k, v in skipped.items()) + ".")
    if repro:
        w("")
        w("Check that the view JPEGs are the crops behind the logged embeddings: the app's MobileNetV3-small run offline "
          "on each view JPEG against the embedding the phone logged for that observation, cosine median %.4f, minimum "
          "%.4f (n = %d)." % (repro["median"], repro["min"], repro["n"]))
    return "\n".join(L) + "\n"


def text_report(results, neg, neg_info, encs, out):
    pr = lambda *a: print(*a, file=out)  # noqa: E731
    for r in results:
        c = r["case"]
        pr("\n== %s (%s), test scan %s: query %d obs (%d with a view), right %s, candidates %d (%d with a view)" % (
            c["name"], c["kind"], c["test"], len(r["query"]), len(r["query_views"]),
            ", ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in r["right_sizes"].items()), r["candidates"], r["candidates_with_view"]))
        for e in encs:
            pe = r["by_encoder"][e]
            for m in MODES:
                rr = pe[m]
                pr("  %-15s %-6s %s  right %s; other %s; share of negatives >= right %s" % (
                    SHORT[e], m, cell(rr), rr["right"], rr["other"],
                    "-" if rr["share_negatives_at_or_above"] is None else "%.3f" % rr["share_negatives_at_or_above"]))
            pr("    top 5 by %s: %s" % (RANK_MODE, "; ".join("%s %s%s %.3f" % (t["group"], t["label"], " RIGHT" if t["right"] else
                                                                                 (" same-entry" if t["same_entry_as_right"] or t["same_entry_as_query"] else ""),
                                                                                 t["score"]) for t in pe[RANK_MODE]["top"])))
    pr("\n== clear negatives: %d pairs from %d groups in %d scans" % (neg_info["pairs"], neg_info["groups"], len(neg_info["scans"])))
    for e in encs:
        for m in MODES:
            s = summary(neg[e][m])
            pr("  %-15s %-6s n %d, median %s, p95 %s, p99 %s, max %s" % (SHORT[e], m, s["n"], f3(s.get("median")),
                                                                         f3(s.get("p95")), f3(s.get("p99")), f3(s.get("max"))))


def git_info():
    import hashlib
    import subprocess
    path = os.path.abspath(__file__)

    def run(*a):
        try:
            return subprocess.run(["git"] + list(a), cwd=ir.REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    st = run("status", "--porcelain", "--", os.path.relpath(path, ir.REPO))
    return {"commit": run("rev-parse", "HEAD"), "scriptSha256": sha,
            "scriptUncommitted": bool(st) if st is not None else None}


class Tee:
    """Writes to a stream and keeps a copy (report.txt)."""

    def __init__(self, stream):
        self.stream, self.parts = stream, []

    def write(self, s):
        self.stream.write(s)
        self.parts.append(s)

    def flush(self):
        self.stream.flush()

    def text(self):
        return "".join(self.parts)


# ============================================================================ main

def main(argv=None, out=sys.stdout):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="pulled data folder")
    ap.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS),
                    help="stronger encoders of bench_match.py, comma-separated, or 'none' (default %(default)s)")
    ap.add_argument("--no-repro", action="store_true", help="skip the offline app-embedder check of the view JPEGs")
    ap.add_argument("--cases", default=None, help="JSON file with cases (default: the audited CASES)")
    ap.add_argument("--out", default=None, help="result folder (default analysis/out/identity_server)")
    args = ap.parse_args(argv)
    out = Tee(out)
    log = lambda *a: print(*a, file=out)  # noqa: E731
    cases = CASES
    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = json.load(f)
    out_dir = args.out or os.path.join(HERE, "out", "identity_server")
    os.makedirs(out_dir, exist_ok=True)
    gi = git_info()
    log("Server-side encoder check of %s (commit %s%s)" % (args.folder, gi.get("commit"),
                                                           ", script uncommitted" if gi.get("scriptUncommitted") else ""))
    pull, order, groups, obs_by_id, info = load_pull(args.folder)
    log("Scans with observations: %s; %d observations, %d with a retained view, %d groups; embedding model %s" % (
        ", ".join(order), info["observations"], info["with view"], len(groups), info["embModel"]))
    if info.get("retained view file missing"):
        log("  retained views whose file is missing: %d (treated as no view)" % info["retained view file missing"])
    keys = [k for k in args.encoders.split(",") if k and k != "none"]
    views = [(o.id, o.view) for o in obs_by_id.values() if o.view]
    enc_keys = (["ours"] if keys and not args.no_repro else []) + keys
    log("Encoders on %d view crops: %s" % (len(views), ", ".join(enc_keys) or "none"))
    feats, timing, skipped = run_encoders(enc_keys, views, log)
    repro = None
    if "ours" in feats:
        cs = [dot(feats["ours"][oid], obs_by_id[oid].emb) for oid, _ in views]
        repro = {"n": len(cs), "median": statistics.median(cs), "min": min(cs)}
        log("  check: offline app embedder on the view JPEGs vs the logged embedding: cosine median %.4f, min %.4f (n = %d)"
            % (repro["median"], repro["min"], repro["n"]))
    sims = collections.OrderedDict()
    phone = {o.id: o.emb for o in obs_by_id.values()}
    sims["phone_all"] = Sim(phone)
    sims["phone"] = Sim(phone)
    for k in keys:
        if k in feats:
            sims[k] = Sim(feats[k])
    encs = list(sims)
    results, neg, neg_info = evaluate(cases, groups, obs_by_id, order, sims)
    neg_info["all_scans"] = order
    for r in results:
        r["sightings"] = query_sightings(pull, r["case"], obs_by_id)
    text_report(results, neg, neg_info, encs, out)
    md = markdown(results, neg, neg_info, encs, timing, skipped, repro, info, args)
    with open(os.path.join(out_dir, "tables.md"), "w", encoding="utf-8") as f:
        f.write(md)
    result = collections.OrderedDict([
        ("folder", os.path.abspath(args.folder)), ("git", gi), ("argv", argv if argv is not None else sys.argv[1:]),
        ("definitions", {"FAR_M": FAR_M, "rankMode": RANK_MODE, "modes": MODES, "ties": "count against the right answer",
                         "candidates": "every group of every scan that started before the test scan"}),
        ("scans", order), ("info", info), ("encoders", {e: LABEL[e] for e in encs}), ("skipped", skipped),
        ("timing", {k: {kk: vv for kk, vv in v.items() if kk != "per_crop_ms"} for k, v in timing.items()}),
        ("reproduction", repro), ("negatives", {"info": neg_info, "summary": {e: {m: summary(neg[e][m]) for m in MODES} for e in encs},
                                                 "values": neg}),
        ("cases", results)])
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    log("\nWritten: %s, %s, %s" % tuple(os.path.join(out_dir, n) for n in ("report.txt", "tables.md", "result.json")))
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(out.text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
