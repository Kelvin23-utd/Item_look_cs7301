#!/usr/bin/env python3
"""Find by name, offline baseline: the app's matching rule against text-to-image retrieval, for text queries
whose target item is known.

    analysis/.venv-bench/bin/python analysis/name_retrieval.py [--models mobileclip2s2,vitl14] [--out DIR]

Methods (fixed before any result was seen):
  L     the app's rule (FindBack/docs/PLAN_v0.9_find_by_name.md, name_search.search) on the phone's own
        names and labels alone.
  N(Y)  the same rule with naming model Y's extra names and descriptors bound (extra names on).
  T(m)  text-to-image retrieval with a joint text-image encoder m: the text "a photo of " + query and every
        item's saved photo are embedded, and all items are ranked by cosine similarity (ties: id ascending).
        Secondary variant, reported separately: the raw query as the text.
Rank of the target: for L and N its 1-based position in the app's result order, or not a result; for T its
position in the full ranking. Measures: n, target is a result (L, N), found@1/3/5/10, median number of
results shown (L, N; T always lists every item).

Evaluation 1 (development set): backups/phone_files_2026-11-17_0008, its 24 scan entries and the 20 tapped
records of the naming package (a record carries only the phone's raw label as its name); queries are the
names the owner typed when saving tapped records, target = that record. The existing counts (target is a
result: L 0 of 9; N 3, 3, 4, 3) are checked before any encoder is loaded; the run stops if they differ.
Evaluation 2 (cross-describer proxy): backups/phone_files_2026-11-19_pre_v0.9_2120 (82 scan entries) with
the names files of analysis/out/name_search/<pull>/ bound by name_search.bind. For each ordered pair (X, Y),
X != Y, and each entry with a name in X's file: query = X's first name, target = that entry; methods L, N(Y),
T(m). Lenient variant: a hit is any entry to which X gave exactly the same first name.

Encoders (analysis/.bench-cache, offline): MobileCLIP2-S2 exactly as bench_match.py loads it, and OpenCLIP
ViT-L/14 trained on LAION-2B (public weights, fetched once from Hugging Face at the revision pinned below,
open_clip_pytorch_model.safetensors). Both see the whole photo squashed to their square input, as
bench_match.py does; the text goes through each model's own open_clip tokenizer.

Private outputs (per-query rows hold the query words) go to analysis/out/name_search/retrieval_2026-11-20/
(git-ignored): results.json and results.md (tables, numbers only; also printed).
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(HERE, ".bench-cache")
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
sys.path.insert(0, HERE)
import name_search as ns  # noqa: E402

vc = ns.vc
PULL1 = os.path.join(REPO, "backups", "phone_files_2026-11-17_0008")
PULL2 = os.path.join(REPO, "backups", "phone_files_2026-11-19_pre_v0.9_2120")
NAMES2 = os.path.join(ns.OUT, os.path.basename(PULL2))
OUT = os.path.join(ns.OUT, "retrieval_2026-11-20")
PROMPT = "a photo of "
VARIANTS = ("prompt", "raw")
KS = (1, 3, 5, 10)
MODELS = vc.CONTESTANTS
EXPECTED_EVAL1 = {"n": 9, "L": 0, "model_a": 3, "model_b": 3, "model_c": 4, "model_d": 3}
WARMUP = 3
ENCODERS = {
    "mobileclip2s2": {"label": "MobileCLIP2-S2", "arch": "MobileCLIP2-S2", "pretrained": "dfndr2b",
                      "repo": "timm/MobileCLIP2-S2-OpenCLIP", "file": "open_clip_model.safetensors"},
    "vitl14": {"label": "OpenCLIP ViT-L/14 LAION-2B", "arch": "ViT-L-14", "pretrained": "laion2b_s32b_b82k",
               "repo": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "revision": "1627032197142fbe2a7cfec626f4ced3ae60d07a",
               "file": "open_clip_pytorch_model.safetensors"},
}


# ============================================================================ ranks and measures

def rank_of(order, accept):
    """1-based position of the first id of `accept` in `order`, or None."""
    for i, x in enumerate(order):
        if x in accept:
            return i + 1
    return None


def rule_cell(items, query, extra, target, accept=None):
    """The app's rule for one query: results shown, the target's rank and, with `accept`, the best rank of any
    accepted id; with extra names on, whether the target is a result only via extra names."""
    ids = [it.id for it, _ in ns.search(items, query, extra)]
    cell = {"shown": len(ids), "rank": rank_of(ids, {target})}
    if accept is not None:
        cell["rankLenient"] = rank_of(ids, accept)
    if extra:
        cell["viaExtra"] = target in ns.via_extra(items, query)
    return cell


def embed_order(sims):
    """Ids by similarity descending, then id ascending; an item without a photo (None) comes last."""
    return sorted(sims, key=lambda i: (sims[i] is None, -(sims[i] or 0.0), i))


def embed_cell(sims, target, accept=None):
    order = embed_order(sims)
    cell = {"rank": rank_of(order, {target}), "sim": sims[target],
            "tie": sims[target] is not None and any(sims[i] == sims[target] for i in sims if i != target)}
    if accept is not None:
        cell["rankLenient"] = rank_of(order, accept)
    return cell


def summary(ranks, shown=None, list_size=None):
    """n, (target is a result,) found@k for each k, (median results shown | list size)."""
    out = {"n": len(ranks)}
    if shown is not None:
        out["result"] = sum(1 for r in ranks if r is not None)
    for k in KS:
        out["found@%d" % k] = sum(1 for r in ranks if r is not None and r <= k)
    if shown is not None:
        out["medianShown"] = statistics.median(shown) if shown else None
    if list_size is not None:
        out["listSize"] = list_size
    return out


def sum_rule(cells, key="rank"):
    return summary([c[key] for c in cells], [c["shown"] for c in cells])


def sum_embed(cells, n_items, key="rank"):
    return summary([c[key] for c in cells], list_size=n_items)


# ============================================================================ text-image encoders

class Encoder:
    """A joint text-image encoder: image(rgb) and text(str) give L2-normalised numpy vectors."""

    def __init__(self, key):
        import numpy as np
        import torch
        import open_clip
        import bench_match as bm
        self.np, self.torch, self.key, spec = np, torch, key, ENCODERS[key]
        if key == "mobileclip2s2":
            m = bm.MobileClip2S2()
        else:
            # bench_match's class without its loader: features() (resize, normalise, encode_image) is the same code,
            # the model and its mean, std and input size are this one's
            m = bm.MobileClip2S2.__new__(bm.MobileClip2S2)
            torch.set_num_threads(bm.TORCH_THREADS)
            m.torch = torch
            bm.HF_REVISIONS.setdefault(spec["repo"], spec["revision"])
            path = bm.hf_snapshot(spec["repo"])
            m.model = open_clip.create_model_and_transforms(spec["arch"], pretrained=os.path.join(path, spec["file"]))[0].eval()
            cfg = open_clip.get_pretrained_cfg(spec["arch"], spec["pretrained"])
            m.mean = np.array(cfg.get("mean") or open_clip.OPENAI_DATASET_MEAN, np.float32)
            m.std = np.array(cfg.get("std") or open_clip.OPENAI_DATASET_STD, np.float32)
            size = getattr(m.model.visual, "image_size", 224)
            m.size = size[0] if isinstance(size, (tuple, list)) else int(size)
            m.file_bytes = os.path.getsize(os.path.join(path, spec["file"]))
            m.revision = os.path.basename(path)
        self.m = m
        self.tok = open_clip.get_tokenizer(spec["arch"])
        n_all = sum(p.numel() for p in m.model.parameters())
        n_img = sum(p.numel() for p in m.model.visual.parameters())
        self.meta = dict(spec, revision=m.revision, fileBytes=m.file_bytes, paramsImage=n_img, paramsOther=n_all - n_img,
                         inputSize=m.size, mean=[float(x) for x in m.mean], std=[float(x) for x in m.std],
                         torchThreads=torch.get_num_threads(),
                         imageHandling="whole photo resized (squashed) to the square input, bilinear, as bench_match.py")

    def image(self, rgb):
        return self.m.features(rgb)

    def text(self, s):
        with self.torch.no_grad():
            v = self.m.model.encode_text(self.tok([s]))[0].numpy()
        return v / self.np.linalg.norm(v)


def load_rgb(path):
    import bench_common
    return bench_common.load_rgb(path)


def embed_all(enc, photos, texts, load=load_rgb):
    """Image vectors of {key: photo path or None} (None stays None), text vectors of each distinct text, and the
    median wall-clock milliseconds per image (resize, normalise, encode; not the JPEG decode) and per text
    (tokenise, encode) after WARMUP calls."""
    iv, it, tv, tt = {}, [], {}, []
    for k in sorted(photos):
        if photos[k] is None:
            iv[k] = None
            continue
        rgb = load(photos[k])
        t0 = time.perf_counter()
        iv[k] = enc.image(rgb)
        it.append(time.perf_counter() - t0)
    for s in sorted(set(texts)):
        t0 = time.perf_counter()
        tv[s] = enc.text(s)
        tt.append(time.perf_counter() - t0)

    def med(xs):
        xs = xs[WARMUP:] or xs
        return 1000 * statistics.median(xs) if xs else None

    return iv, tv, {"msPerImage": med(it), "nImages": len(it), "msPerText": med(tt), "nTexts": len(tt),
                    "imagesTotalS": sum(it), "textsTotalS": sum(tt)}


def sims_for(tvec, ivecs, ids):
    """{item id: cosine} of one text vector against the image vectors of `ids` (None without a photo)."""
    return {i: (None if ivecs[i] is None else float(tvec @ ivecs[i])) for i in ids}


def text_of(query, variant):
    return PROMPT + query if variant == "prompt" else query


# ============================================================================ evaluation 1

def eval1_data(pull=PULL1, vlm_out=ns.VLM_OUT):
    """(items per condition, photos, queries [(record id, query)], provenance) as the first look built them."""
    scan = ns.Scan(pull)
    recs = ns.pull_records(pull)
    pkg, _ = ns.package_images(vlm_out)
    rec_ids = sorted(i[4:] for i, v in pkg.items() if v[2] == "R")
    typed = {}
    with open(os.path.join(pull, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "register" and e.get("nameSource") == "typed" and e.get("record"):
                typed[e["record"]] = e.get("userName")
    typed = {k: v for k, v in typed.items() if k in rec_ids and v}
    rec_photo = {rid: os.path.join(pull, recs[rid].get("cropFile") or "crops/%s.jpg" % rid) for rid in rec_ids}
    runs, answers = {}, {}
    for m in MODELS:
        run = ns.primary_rows(vlm_out, m)
        runs[m], rows, out = run["id"], run["rows"], {}
        for e in scan.entries:
            iid = "inv:" + e["id"]
            failed, _, ans = ns.check_output(rows.get(iid), pkg[iid][1], scan.sha[e["id"]], scan.size[e["id"]])
            if failed is None:
                out[e["id"]] = ans
        for rid in rec_ids:
            p = rec_photo[rid]
            sha, size = (ns.sha256_file(p), os.path.getsize(p)) if os.path.isfile(p) else (None, None)
            failed, _, ans = ns.check_output(rows.get("rec:" + rid), pkg["rec:" + rid][1], sha, size)
            if failed is None:
                out[rid] = ans
        answers[m] = out

    def items(ans):
        def extra(a):
            n = vc.answer_name(a) if a else None
            return ([ns.clean_text(n)] if n else [],
                    [ns.clean_text(x) for x in (a.get("attributes") or []) if isinstance(x, str) and x.strip()] if a else [])
        out = []
        for e in scan.entries:
            x, d = extra(ans.get(e["id"]))
            out.append(ns.Item(id=e["id"], name=e.get("name") or "", labels=ns.entry_labels(e), container=e.get("container"),
                               room=scan.rooms.get(e.get("roomId")), last_seen_ms=int(e.get("lastSeenMs") or 0),
                               extra_names=x, descriptors=d))
        for rid in rec_ids:
            r = recs[rid]
            x, d = extra(ans.get(rid))
            out.append(ns.Item(id="rec:" + rid, name=r.get("rawLabel") or "", labels=[r.get("rawLabel") or ""],
                               room=scan.rooms.get(r.get("roomId")), last_seen_ms=int(r.get("lastSeenMs") or 0),
                               extra_names=x, descriptors=d))
        return out

    conds = {"L": items({})}
    conds.update({m: items(answers[m]) for m in MODELS})
    photos = {e["id"]: scan.photo[e["id"]] for e in scan.entries}
    photos.update({"rec:" + rid: rec_photo[rid] if os.path.isfile(rec_photo[rid]) else None for rid in rec_ids})
    queries = [("rec:" + rid, typed[rid]) for rid in sorted(typed)]
    prov = {"pull": ns.rel(pull), "inventorySha256": ns.sha256_file(os.path.join(pull, "inventory.json")),
            "eventsSha256": ns.sha256_file(os.path.join(pull, "events.jsonl")), "nScanEntries": len(scan.entries),
            "nRecords": len(rec_ids), "namingRuns": runs, "usableAnswers": {m: len(answers[m]) for m in MODELS}}
    return conds, photos, queries, prov


def eval1_rule(conds, queries):
    """Per-query rule cells: [{"target", "query", "L": cell, model: cell, ...}]."""
    rows = []
    for target, q in queries:
        row = {"target": target, "query": q, "L": rule_cell(conds["L"], q, False, target)}
        for m in MODELS:
            row[m] = rule_cell(conds[m], q, True, target)
        rows.append(row)
    return rows


def eval1_summary(rows, enc_keys, n_items):
    s = {"L": sum_rule([r["L"] for r in rows])}
    s.update({m: sum_rule([r[m] for r in rows]) for m in MODELS})
    for k in enc_keys:
        for v in VARIANTS:
            s["T:%s:%s" % (k, v)] = sum_embed([r["T"][k][v] for r in rows], n_items)
    return s


def check_eval1(rows, expected=EXPECTED_EVAL1):
    """(ok, got): the number of queries and, per condition, how often the target is a result."""
    got = {"n": len(rows)}
    got.update({k: sum(1 for r in rows if r[k]["rank"] is not None) for k in ["L"] + list(MODELS)})
    return got == expected, got


# ============================================================================ evaluation 2

def eval2_data(pull=PULL2, names_dir=NAMES2):
    scan = ns.Scan(pull)
    bound, binding, sha, first = {}, {}, {}, {}
    for m in MODELS:
        p = os.path.join(names_dir, "names_%s.json" % m)
        doc = ns.read_json(p)
        problems = ns.validate_names(doc)
        if problems:
            raise SystemExit("%s: %s" % (p, "; ".join(problems)))
        bound[m], n = ns.bind(scan, doc)
        binding[m], sha[m] = dict(n), ns.sha256_file(p)
        first[m] = {eid: v[0][0] for eid, v in bound[m].items() if v[0]}
    conds = {"L": scan.items()}
    conds.update({m: scan.items(bound[m]) for m in MODELS})
    photos = {e["id"]: scan.photo[e["id"]] for e in scan.entries}
    prov = {"pull": ns.rel(pull), "inventorySha256": ns.sha256_file(os.path.join(pull, "inventory.json")),
            "nScanEntries": len(scan.entries), "namesFiles": {m: ns.rel(os.path.join(names_dir, "names_%s.json" % m))
                                                             for m in MODELS},
            "namesSha256": sha, "binding": binding, "withName": {m: len(first[m]) for m in MODELS}}
    return conds, photos, first, prov


def cross_queries(first):
    """[(X, target, query, accepted ids)]: every entry with a name in X's file, X's first name as the query; the
    accepted ids are the entries to which X gave exactly the same first name."""
    out = []
    for x in sorted(first):
        for eid in sorted(first[x]):
            q = first[x][eid]
            out.append((x, eid, q, sorted(e for e, n in first[x].items() if n == q)))
    return out


def eval2_rule(conds, first):
    """Per (X, query) the L cell and, per Y != X, the N(Y) cell."""
    rows = []
    for x, target, q, accept in cross_queries(first):
        acc = set(accept)
        rows.append({"x": x, "target": target, "query": q, "accept": accept,
                     "L": rule_cell(conds["L"], q, False, target, acc),
                     "N": {y: rule_cell(conds[y], q, True, target, acc) for y in sorted(first) if y != x}})
    return rows


def eval2_summary(rows, enc_keys, n_items):
    """Per index model Y (queries of every X != Y) and overall (all ordered pairs): L, N(Y), T(m) per variant,
    strict and lenient."""
    ys = sorted({y for r in rows for y in r["N"]})
    out = {}
    for y in ys + ["all"]:
        pairs = [(r, yy) for r in rows for yy in r["N"] if y in ("all", yy)]
        s = {}
        for lab, key in (("strict", "rank"), ("lenient", "rankLenient")):
            t = {"L": sum_rule([r["L"] for r, _ in pairs], key), "N": sum_rule([r["N"][yy] for r, yy in pairs], key)}
            for m in enc_keys:
                for v in VARIANTS:
                    t["T:%s:%s" % (m, v)] = sum_embed([r["T"][m][v] for r, _ in pairs], n_items, key)
            s[lab] = t
        s["viaExtra"] = sum(1 for r, yy in pairs if r["N"][yy]["viaExtra"])
        s["nPairs"] = len(pairs)
        s["sharedName"] = sum(1 for r, _ in pairs if len(r["accept"]) > 1)
        out[y] = s
    return out


# ============================================================================ T cells

def add_t(rows, enc_key, ivecs, tvecs, ids, accept_key=None):
    for r in rows:
        acc = set(r[accept_key]) if accept_key else None
        r.setdefault("T", {})[enc_key] = {v: embed_cell(sims_for(tvecs[text_of(r["query"], v)], ivecs, ids), r["target"], acc)
                                          for v in VARIANTS}


# ============================================================================ report (numbers only)

def fmt(s, k):
    return "%d/%d" % (s[k], s["n"])


def rule_line(name, s):
    return "| %s | %d | %s | %s | %s | %s | %s | %s |" % (name, s["n"], fmt(s, "result"), *(fmt(s, "found@%d" % k) for k in KS),
                                                         "%g" % s["medianShown"] if s["medianShown"] is not None else "–")


def embed_line(name, s):
    return "| %s | %d | – | %s | %s | %s | %s | all %d |" % (name, s["n"], *(fmt(s, "found@%d" % k) for k in KS), s["listSize"])


HEAD = ["| Method | n | Target is a result | found@1 | found@3 | found@5 | found@10 | Median results shown |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]


def report_md(res):
    L = []
    w = L.append
    e1, e2, encs = res["eval1"], res["eval2"], res["encoders"]
    lab = {k: v["label"] for k, v in encs.items()}
    w("# Find by name: the app's rule against text-to-image retrieval (offline)")
    w("")
    w("Generated by `analysis/name_retrieval.py` at %s (repository commit %s). Numbers only; the per-query rows "
      "(with the query words) are in results.json next to this file. T text: `\"%s\" + query`; the raw-query variant "
      "is secondary and reported separately. Ties in T are broken by id; the target tied with another item in %d "
      "T rankings." % (res["generatedUtc"], res["gitCommit"][:7], PROMPT, res["tTies"]))
    w("")
    w("## Evaluation 1: owner-typed names (development set)")
    w("")
    p = e1["provenance"]
    w("`%s`: %d items (%d scan entries, %d tapped records; %d without a photo). Queries: names typed when saving a "
      "tapped record, target = that record, n = %d. Check against the first look (target is a result): %s; expected "
      "%s: %s." % (p["pull"], e1["nItems"], p["nScanEntries"], p["nRecords"], e1["noPhoto"], e1["check"]["n"],
                   ", ".join("%s %d" % (k, v) for k, v in e1["check"].items() if k != "n"),
                   ", ".join("%s %d" % (k, v) for k, v in EXPECTED_EVAL1.items() if k != "n"),
                   "reproduced" if e1["reproduced"] else "NOT reproduced"))
    w("")
    for v in VARIANTS:
        if v == "raw":
            w("Raw-query variant (secondary; T text = the query alone; L and N do not change):")
            w("")
        L.extend(HEAD)
        if v == "prompt":
            w(rule_line("L (phone labels alone)", e1["summary"]["L"]))
            for m in MODELS:
                w(rule_line("N(%s)" % m, e1["summary"][m]))
        for k in encs:
            w(embed_line("T(%s)%s" % (lab[k], ", raw query" if v == "raw" else ""), e1["summary"]["T:%s:%s" % (k, v)]))
        w("")
        if v == "prompt":
            w("Target a result only via extra names: %s (of %d queries each)." % (", ".join(
                "%s %d" % (m, e1["viaExtra"][m]) for m in MODELS), e1["check"]["n"]))
            w("")
    w("## Evaluation 2: cross-describer proxy")
    w("")
    p = e2["provenance"]
    w("`%s`: %d scan entries (%d without a photo). Names files bound: %s. Entries with a name (queries per X): %s. "
      "Per index model Y, the queries of the three other models X are pooled; \"all\" pools the 12 ordered pairs, so "
      "each query of X counts three times for L and T." % (
          p["pull"], p["nScanEntries"], e2["noPhoto"],
          ", ".join("%s %d/%d" % (m, p["binding"][m]["bound"], p["binding"][m]["items"]) for m in MODELS),
          ", ".join("%s %d" % (m, p["withName"][m]) for m in MODELS)))
    w("")
    for v in VARIANTS:
        for crit, text in (("strict", "Strict: a hit is the target entry"),
                           ("lenient", "Lenient: a hit is any entry to which X gave exactly the same first name")):
            w("### %s%s" % (text, " (raw-query variant, T only)" if v == "raw" else ""))
            w("")
            w("| Index Y (queries from X != Y) " + HEAD[0])
            w("| --- " + HEAD[1])
            for y, s in e2["summary"].items():
                t = s[crit]
                if v == "prompt":
                    w("| %s " % y + rule_line("L", t["L"]))
                    w("| %s " % y + rule_line("N(%s)" % (y if y != "all" else "Y"), t["N"]))
                for k in encs:
                    w("| %s " % y + embed_line("T(%s)%s" % (lab[k], ", raw" if v == "raw" else ""), t["T:%s:%s" % (k, v)]))
            w("")
    w("N(Y): target a result only via extra names, and queries whose first name X gave to more than one entry:")
    w("")
    w("| Index Y | n | Target via extra | Target is a result | Query name shared by several entries |")
    w("| --- | --- | --- | --- | --- |")
    for y, s in e2["summary"].items():
        w("| %s | %d | %d/%d | %d/%d | %d/%d |" % (y, s["nPairs"], s["viaExtra"], s["nPairs"], s["strict"]["N"]["result"],
                                                 s["nPairs"], s["sharedName"], s["nPairs"]))
    w("")
    w("## Text-image encoders and cost on this Mac (CPU, batch 1; not phone time)")
    w("")
    w("| Encoder | open_clip model / pretrained | Weights (revision) | File on disk | Parameters image / text | Input | "
      "ms per image (median, n) | ms per query text (median, n) | Load s |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k, e in encs.items():
        c = e["cost"]
        w("| %s | %s / %s | %s (%s) | %.1f MB | %.1f M / %.1f M | %d px | %.1f (%d) | %.1f (%d) | %.1f |" % (
            e["label"], e["arch"], e["pretrained"], e["repo"], e["revision"][:7], e["fileBytes"] / 1e6,
            e["paramsImage"] / 1e6, e["paramsOther"] / 1e6, e["inputSize"], c["msPerImage"], c["nImages"],
            c["msPerText"], c["nTexts"], e["loadS"]))
    w("")
    w("CPU: %s; %s." % (res["environment"]["cpu"], ", ".join("%s %s" % kv for kv in res["environment"]["versions"].items())))
    w("")
    return "\n".join(L)


# ============================================================================ main

def versions():
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for pkg in ("torch", "open_clip_torch", "timm", "numpy", "pillow", "huggingface_hub", "safetensors", "tokenizers"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def git_commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def cpu_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(ENCODERS), help="comma-separated subset of %s" % ", ".join(ENCODERS))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in ENCODERS:
            ap.error("unknown encoder %s" % k)
    t_start = time.time()

    conds1, photos1, queries1, prov1 = eval1_data()
    rows1 = eval1_rule(conds1, queries1)
    ok, got = check_eval1(rows1)
    if not ok:
        raise SystemExit("Evaluation 1 does not reproduce the first look: got %s, expected %s. Stopping." % (got, EXPECTED_EVAL1))
    conds2, photos2, first2, prov2 = eval2_data()
    rows2 = eval2_rule(conds2, first2)
    ids1, ids2 = [it.id for it in conds1["L"]], [it.id for it in conds2["L"]]
    assert sorted(ids1) == sorted(photos1) and sorted(ids2) == sorted(photos2)

    encs = {}
    for k in keys:
        t0 = time.perf_counter()
        enc = Encoder(k)
        load_s = time.perf_counter() - t0
        texts = [text_of(r["query"], v) for r in rows1 + rows2 for v in VARIANTS]
        photos = {("1", i): p for i, p in photos1.items()}
        photos.update({("2", i): p for i, p in photos2.items()})
        iv, tv, cost = embed_all(enc, photos, texts)
        add_t(rows1, k, {i: iv[("1", i)] for i in ids1}, tv, ids1)
        add_t(rows2, k, {i: iv[("2", i)] for i in ids2}, tv, ids2, accept_key="accept")
        encs[k] = dict(enc.meta, cost=cost, loadS=load_s)
        print("%s: %.1f ms per image (%d), %.1f ms per text (%d)" % (k, cost["msPerImage"], cost["nImages"],
                                                                   cost["msPerText"], cost["nTexts"]), file=sys.stderr)
        del enc

    ties = sum(1 for r in rows1 + rows2 for k in keys for v in VARIANTS if r["T"][k][v]["tie"])
    res = {
        "generatedUtc": ns.utc_now(), "gitCommit": git_commit(), "scriptSha256": ns.sha256_file(os.path.abspath(__file__)),
        "nameSearchSha256": ns.sha256_file(os.path.join(HERE, "name_search.py")),
        "protocol": {"prompt": PROMPT, "variants": list(VARIANTS), "ks": list(KS), "tieBreak": "id ascending",
                     "lenient": "evaluation 2: any entry to which X gave exactly the same first name (string equality)"},
        "environment": {"cpu": cpu_name(), "versions": versions()},
        "encoders": encs, "tTies": ties, "wallS": time.time() - t_start,
        "eval1": {"provenance": prov1, "nItems": len(ids1), "noPhoto": sum(1 for p in photos1.values() if p is None),
                  "check": got, "reproduced": ok, "summary": eval1_summary(rows1, keys, len(ids1)),
                  "viaExtra": {m: sum(1 for r in rows1 if r[m]["viaExtra"]) for m in MODELS}, "perQuery": rows1},
        "eval2": {"provenance": prov2, "nItems": len(ids2), "noPhoto": sum(1 for p in photos2.values() if p is None),
                  "summary": eval2_summary(rows2, keys, len(ids2)), "perQuery": rows2},
    }
    os.makedirs(args.out, exist_ok=True)
    ns.write_json(os.path.join(args.out, "results.json"), res)
    md = report_md(res)
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    ns.warn_if_tracked(args.out, sys.stderr)
    print(md)
    print("Written: %s, %s" % (os.path.join(args.out, "results.json"), os.path.join(args.out, "results.md")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
