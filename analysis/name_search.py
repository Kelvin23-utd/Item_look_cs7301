#!/usr/bin/env python3
"""Find by name, offline: the v0.9 matching rule (FindBack/docs/PLAN_v0.9_find_by_name.md) run over one
saved scan, with the phone's labels alone and with extra names taken from the saved outputs of the VLM
naming experiment (analysis/notes/2026-11-17_vlm_naming_design.md).

    python3 -I analysis/name_search.py join <pull>
    python3 -I analysis/name_search.py export <pull> --model model_c [--out <names.json>]
    python3 -I analysis/name_search.py sheet <pull> [--out <sheet.html>]
    python3 -I analysis/name_search.py freeze <queries.csv>
    python3 -I analysis/name_search.py run <pull> --queries <queries.csv> --names <names.json> [--names ...]

<pull> is a pulled phone folder (inventory.json, inventory/<id>.jpg, map.json). The saved scan of the
naming experiment is backups/phone_files_2026-11-17_0008 (24 scan entries, map ce41deb2). Another pull is
named by a names run of its own (vlm_prepare.py --names-only): give its root with --vlm-out
analysis/out/vlm_names/<pull folder>. Outputs go to analysis/out/name_search/<pull folder>/ (git-ignored)
unless --out says otherwise.

join    Can each saved model output be tied to the exact entry and the exact photo? Key: the entry id and
        the SHA-256 of the photo bytes. The package's crop-full image of the entry (documented as the
        original file bytes when no resizing was needed) must hash like <pull>/<cropFile>, and the request
        must have carried that many image bytes. Per model, from its primary run as analysis/vlm_report.py
        chooses it (sequential phase, crop-full, repeat 1: the design's name of an item), a cumulative
        count: entries in the scan, with an output, parse OK, photo hash equal, the model said "one item",
        non-empty name (= usable). The tapped records (set R) are counted the same way.
export  A names.json for the phone (schema in the plan) from one model's usable outputs: `names` = the
        model's name (lower case, spaces collapsed, outer quotes and punctuation removed; left out when it
        equals the phone's own name after the plan's normalisation), `descriptors` = the answer's
        `attributes` (the only attribute field of the answer schema; `category` and `visible_text` are not
        used), `source` = <alias>:<date of the run>. Scan entries only.
sheet   A self-contained HTML page on which the owner writes queries before any ranking is seen: each scan
        entry's photo with a code (A, B, ...; entries in id order), nothing else about it (no name, label
        or model output). It downloads queries.csv: query,type,target_code,target_id with type broad,
        specific or none (none rows have no target).
freeze  Writes <queries.csv>.frozen.json: the SHA-256 of the file, the rows by type, the UTC time and the
        SHA-256 of this script. `run` refuses a queries file without a record of the same hash.
run     Every query in every condition (phone labels alone; each names file on, bound by photo hash):
        number of results, the target's rank (or not found) and match class; per condition and query type:
        targets found, in top 1 and top 3, median rank; queries for nothing in the scan: number of results.
        Examples: queries where the extra names helped, made no difference or hurt; results present only via
        extra names that are not the target, listed to review (only the owner can say whether they are
        wrong). Markdown and JSON, with the frozen hash.

Standard library only. No image is decoded: photos are hashed and, for the sheet, embedded as bytes. The
models are named by their aliases only.
"""
import argparse
import base64
import collections
import csv
import dataclasses
import datetime
import hashlib
import html
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import vlm_common as vc  # noqa: E402
import vlm_report as vr  # noqa: E402

VLM_OUT = os.path.join(HERE, "out", "vlm")
OUT = os.path.join(HERE, "out", "name_search")
QUERY_TYPES = ("broad", "specific", "none")
TARGET_TYPES = ("broad", "specific")
CSV_COLUMNS = ["query", "type", "target_code", "target_id"]
NAMES_VERSION = 1
CHECKS = ["output", "parse", "photo", "one_item", "name"]
CHECK_TEXT = collections.OrderedDict([
    ("output", "with an output in the primary run"), ("parse", "parse OK"), ("photo", "photo hash equal"),
    ("one_item", "the model said one item"), ("name", "non-empty name (usable)")])
HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ============================================================================ matching rule (the plan)

def normalise(s):
    """The plan's normal form: lower case; every character that is not a letter or a decimal digit (as
    Kotlin's Char.isLetterOrDigit) becomes a space; runs of spaces collapse; trimmed."""
    if not s:
        return ""
    return " ".join("".join(c if c.isalpha() or c.isdecimal() else " " for c in s.lower()).split())


def words(s):
    return normalise(s).split()


@dataclasses.dataclass
class Item:
    """One searchable item: the phone's own fields and, only when bound, its extra names and descriptors."""
    id: str
    name: str
    labels: list = dataclasses.field(default_factory=list)
    container: str = None
    room: str = None
    last_seen_ms: int = 0
    extra_names: list = dataclasses.field(default_factory=list)
    descriptors: list = dataclasses.field(default_factory=list)


def pools(item, extra):
    """(naming words, other words) of an item; the extra fields only when `extra`."""
    naming = [item.name] + (list(item.extra_names) if extra else [])
    other = list(item.labels) + (list(item.descriptors) if extra else []) + [item.container, item.room]
    return [w for s in naming for w in words(s)], [w for s in other for w in words(s)]


def match_class(item, query, extra):
    """Match class 0..3 of an item for a query with at least one word (lower is better), None when the
    item is not a result. `extra`: the switch is on (the item carries extra names only when bound)."""
    q = normalise(query)
    qw = q.split()
    if not qw:
        return None
    if q == normalise(item.name) or (extra and any(q == normalise(x) for x in item.extra_names)):
        return 0
    naming, other = pools(item, extra)
    if all(any(w.startswith(x) for w in naming) for x in qw):
        return 1
    both = naming + other
    if all(any(w.startswith(x) for w in both) for x in qw):
        return 2
    if all(any(w.startswith(x) or (len(x) >= 3 and x in w) for w in both) for x in qw):
        return 3
    return None


def search(items, query, extra):
    """Results in the plan's order: [(item, class)], class ascending, then last seen descending, then id
    ascending. A query without words lists every item by last seen descending (class None)."""
    if not words(query):
        return [(it, None) for it in sorted(items, key=lambda i: (-i.last_seen_ms, i.id))]
    res = [(it, c) for it in items for c in [match_class(it, query, extra)] if c is not None]
    res.sort(key=lambda r: (r[1], -r[0].last_seen_ms, r[0].id))
    return res


def via_extra(items, query):
    """Ids that are results with extra names on and not with them off (none for a query without words)."""
    if not words(query):
        return set()
    off = {it.id for it, _ in search(items, query, False)}
    return {it.id for it, _ in search(items, query, True)} - off


def extra_hits(item, query):
    """The extra names and descriptors of an item that the query matched: [(field, text)]. An extra name
    matches when it equals the whole normalised query; either matches when a query word is a prefix of one
    of its words or, with 3 characters or more, contained in one."""
    q = normalise(query)
    qw = q.split()
    out = []
    for field, values in (("name", item.extra_names), ("descriptor", item.descriptors)):
        for v in values:
            ws = words(v)
            if (field == "name" and q and q == normalise(v)) or any(
                    w.startswith(x) or (len(x) >= 3 and x in w) for x in qw for w in ws):
                out.append((field, v))
    return out


# ============================================================================ files

def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def default_dir(pull):
    return os.path.join(OUT, os.path.basename(os.path.normpath(pull)))


def warn_if_tracked(path, out):
    """A private output inside the repository must be git-ignored; say so when it is not."""
    try:
        r = subprocess.run(["git", "-C", REPO, "check-ignore", "-q", os.path.abspath(path)],
                           capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return
    if r.returncode == 1:
        print("WARNING: %s is inside the repository and not git-ignored; it holds private data" % path, file=out)


def code(i):
    """Short code of the i-th item (0-based): A..Z, AA, AB, ..."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def entry_labels(e):
    """Every detector label of an entry: the label, then the label-vote keys, without repeats."""
    out = []
    for x in [e.get("label")] + list(e.get("labelVotes") or {}):
        if x and x not in out:
            out.append(x)
    return out


class Scan:
    """The scan entries of a pull (inventory.json), their photos (<pull>/<cropFile>), the room names of
    map.json and a short code per entry (A, B, ... in id order)."""

    def __init__(self, pull):
        self.pull = pull
        inv = read_json(os.path.join(pull, "inventory.json"))
        if not isinstance(inv, dict):
            raise SystemExit("%s: no readable inventory.json" % pull)
        self.entries = sorted(inv.get("entries") or [], key=lambda e: e["id"])
        m = read_json(os.path.join(pull, "map.json"), {}) or {}
        self.map_id = m.get("mapId")
        self.rooms = {r.get("id"): r.get("name") for r in m.get("rooms") or []}
        self.entry_map_ids = sorted({e.get("mapId") for e in self.entries if e.get("mapId")})
        self.by_id = {e["id"]: e for e in self.entries}
        self.codes = {e["id"]: code(i) for i, e in enumerate(self.entries)}
        self.photo, self.sha, self.size = {}, {}, {}
        for e in self.entries:
            p = os.path.join(pull, e.get("cropFile") or "inventory/%s.jpg" % e["id"])
            self.photo[e["id"]] = p if os.path.isfile(p) else None
            self.sha[e["id"]] = sha256_file(p) if os.path.isfile(p) else None
            self.size[e["id"]] = os.path.getsize(p) if os.path.isfile(p) else None

    def items(self, bound=None):
        """Search items of every entry; `bound`: entry id -> (extra names, descriptors)."""
        out = []
        for e in self.entries:
            x = (bound or {}).get(e["id"])
            out.append(Item(id=e["id"], name=e.get("name") or "", labels=entry_labels(e), container=e.get("container"),
                            room=self.rooms.get(e.get("roomId")), last_seen_ms=int(e.get("lastSeenMs") or 0),
                            extra_names=list(x[0]) if x else [], descriptors=list(x[1]) if x else []))
        return out


def pull_records(pull):
    """record id -> record: records.json first, then the copies in records_log.jsonl (as vlm_prepare)."""
    recs = {}
    for r in read_json(os.path.join(pull, "records.json"), []) or []:
        if isinstance(r, dict) and r.get("id"):
            recs.setdefault(r["id"], r)
    try:
        with open(os.path.join(pull, "records_log.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line).get("record")
                except (ValueError, AttributeError):
                    continue
                if isinstance(r, dict) and r.get("id"):
                    recs.setdefault(r["id"], r)
    except OSError:
        pass
    return recs


# ============================================================================ join

def primary_rows(vlm_out, model):
    """The primary run of a model (vlm_report.primary_run) as {id, date, rows: item -> request row} with the
    rows of the sequential phase, crop-full input, repeat 1; None without a finished run. The run folder's
    path is never returned (a former key may be in it); `id` uses the alias."""
    run = vr.primary_run(vlm_out, model)
    if not run:
        return None
    rows = {}
    for r in vr.load_jsonl(os.path.join(run["dir"], "requests.jsonl")):
        if r.get("phase") == "seq" and r.get("variant") == vc.PRIMARY_VARIANT and r.get("repeat") == 1:
            rows.setdefault(r.get("item"), r)
    return {"id": run["id"], "date": run["id"].split("/", 1)[-1][:10], "rows": rows}


def check_output(row, pkg_sha, photo_sha, photo_size):
    """(first failed check or None, detail, parsed answer) of one saved output against one photo, in the
    order of CHECKS."""
    if row is None:
        return "output", "no request of the primary run for this item", None
    if row.get("status") != 200 or row.get("error"):
        return "parse", "request failed (status %s)" % row.get("status"), None
    if row.get("finish") == "length":
        return "parse", "answer truncated", None
    ans = vc.parse_answer(row.get("text"), required=vc.NAME_SCHEMA["required"])
    if ans is None:
        return "parse", "answer not parsable as the answer schema", None
    if photo_sha is None:
        return "photo", "no photo in the pull", ans
    if pkg_sha is None:
        return "photo", "no crop-full image in the package", ans
    if pkg_sha != photo_sha:
        return "photo", "package image differs from the pull's photo", ans
    if row.get("jpeg_bytes") != photo_size:
        return "photo", "the request carried %s image bytes, the photo has %s" % (row.get("jpeg_bytes"), photo_size), ans
    if ans.get("kind") != "one item":
        return "one_item", "kind = %s" % ans.get("kind"), ans
    if not vc.answer_name(ans):
        return "name", "empty name", ans
    return None, "", ans


def funnel(statuses):
    """Cumulative counts: n, then the number passing every check up to each of CHECKS."""
    out = collections.OrderedDict([("n", len(statuses))])
    order = {c: i for i, c in enumerate(CHECKS)}
    for i, c in enumerate(CHECKS):
        out[c] = sum(1 for s in statuses if s is None or order[s] > i)
    return out


def package_images(vlm_out):
    """item id -> (package path of the crop-full image or None, its SHA-256 or None), and the manifest."""
    man = vr.load_json(os.path.join(vlm_out, "package", "items.json"), {"items": []}) or {"items": []}
    out = {}
    for it in man.get("items") or []:
        files = (it.get("variants") or {}).get(vc.PRIMARY_VARIANT) or []
        p = os.path.join(vlm_out, "package", "images", files[0]) if len(files) == 1 else None
        out[it["id"]] = (p, sha256_file(p) if p and os.path.isfile(p) else None, it.get("set"))
    return out, man


def join(scan, vlm_out=VLM_OUT, models=None):
    """The join of the saved naming outputs to the scan's entries (and the counts for the records)."""
    models = list(models or vc.CONTESTANTS)
    pkg, _ = package_images(vlm_out)
    recs = pull_records(scan.pull)
    rec_items = sorted(i for i, v in pkg.items() if v[2] == "R")
    rec_photo = {}
    for iid in rec_items:
        r = recs.get(iid[len("rec:"):])
        p = os.path.join(scan.pull, r.get("cropFile") or "crops/%s.jpg" % r["id"]) if r else None
        rec_photo[iid] = (sha256_file(p), os.path.getsize(p)) if p and os.path.isfile(p) else (None, None)
    s_ids = {"inv:" + e["id"] for e in scan.entries}
    local = (vr.load_json(os.path.join(vlm_out, "local", "items_local.json"), {}) or {}).get("items") or {}

    def resized(iid):
        return bool((((local.get(iid) or {}).get("images") or {}).get("full") or {}).get("resized"))

    res = {
        "pull": rel(scan.pull), "mapId": scan.map_id,
        "entryMapIds": scan.entry_map_ids, "nEntries": len(scan.entries),
        "photosEqualPackage": sum(1 for e in scan.entries if scan.sha[e["id"]] and
                                  pkg.get("inv:" + e["id"], (None, None))[1] == scan.sha[e["id"]]),
        "packageMarkedResized": sum(1 for e in scan.entries if resized("inv:" + e["id"])),
        "packageEntriesNotInPull": sorted(i for i, v in pkg.items() if v[2] == "S" and i not in s_ids),
        "records": {"inPackage": len(rec_items), "inPull": sum(1 for i in rec_items if i[4:] in recs),
                    "photosEqualPackage": sum(1 for i in rec_items if rec_photo[i][0] and rec_photo[i][0] == pkg[i][1]),
                    "packageMarkedResized": sum(1 for i in rec_items if resized(i))},
        "models": collections.OrderedDict(),
    }
    for m in models:
        run = primary_rows(vlm_out, m)
        rows = run["rows"] if run else {}
        per = []
        for e in scan.entries:
            iid = "inv:" + e["id"]
            failed, detail, ans = check_output(rows.get(iid), pkg.get(iid, (None, None))[1], scan.sha[e["id"]],
                                               scan.size[e["id"]])
            per.append({"id": e["id"], "code": scan.codes[e["id"]], "failed": failed, "detail": detail,
                        "cropSha256": scan.sha[e["id"]], "_answer": ans})
        rstat = [check_output(rows.get(i), pkg[i][1], rec_photo[i][0], rec_photo[i][1])[0] for i in rec_items]
        usable = [p["_answer"] for p in per if p["failed"] is None]
        res["models"][m] = {
            "run": run["id"] if run else None, "runDate": run["date"] if run else None,
            "funnel": funnel([p["failed"] for p in per]), "entries": per, "recordFunnel": funnel(rstat),
            "fields": {"answerFields": sorted({k for a in usable for k in a}),
                       "withAttributes": sum(1 for a in usable if any(isinstance(x, str) and x.strip()
                                                                      for x in a.get("attributes") or [])),
                       "attributes": sum(len(a.get("attributes") or []) for a in usable),
                       "withVisibleText": sum(1 for a in usable if str(a.get("visible_text") or "").strip()),
                       "usable": len(usable)},
        }
    return res


def public_join(res):
    """The join result without the parsed answers (they hold the model's names)."""
    out = json.loads(json.dumps({k: v for k, v in res.items() if k != "models"}))
    out["models"] = collections.OrderedDict()
    for m, d in res["models"].items():
        out["models"][m] = dict(d, entries=[{k: v for k, v in p.items() if not k.startswith("_")} for p in d["entries"]])
    return out


def join_lines(res):
    L = []
    w = L.append
    n = res["nEntries"]
    w("Join of the saved naming outputs to %s (map %s; %d scan entries; entries' map ids: %s)" % (
        res["pull"], res["mapId"], n, ", ".join(res["entryMapIds"]) or "none"))
    w("Photos: %d/%d package crop-full images have the SHA-256 of <pull>/<cropFile> (%d marked resized by "
      "vlm_prepare); package entries not in this pull: %d" % (
          res["photosEqualPackage"], n, res["packageMarkedResized"], len(res["packageEntriesNotInPull"])))
    w("")
    w("| Model | Primary run | Entries in scan | %s |" % " | ".join(CHECK_TEXT.values()))
    w("| --- | --- | --- |%s" % (" --- |" * len(CHECKS)))
    for m, d in res["models"].items():
        f = d["funnel"]
        w("| %s | %s | %d | %s |" % (m, d["run"] or "no finished run", f["n"],
                                     " | ".join("%d/%d" % (f[c], f["n"]) for c in CHECKS)))
    w("")
    for m, d in res["models"].items():
        bad = [p for p in d["entries"] if p["failed"]]
        w("%s: %d/%d usable%s" % (m, d["funnel"]["name"], n, "; not usable:" if bad else ""))
        for p in bad:
            w("  %s (%s): fails '%s': %s" % (p["id"], p["code"], CHECK_TEXT[p["failed"]], p["detail"]))
    w("")
    w("Fields of the usable answers (beyond the name):")
    for m, d in res["models"].items():
        f = d["fields"]
        w("  %s: fields %s; %d/%d with at least one attribute (%d attributes); %d/%d with visible text" % (
            m, ", ".join(f["answerFields"]) or "none", f["withAttributes"], f["usable"], f["attributes"],
            f["withVisibleText"], f["usable"]))
    w("")
    r = res["records"]
    w("Tapped records (R): %d in the package, %d found in this pull, %d/%d photos equal to the package's crop-full "
      "image (%d package copies marked resized by vlm_prepare)." % (
          r["inPackage"], r["inPull"], r["photosEqualPackage"], r["inPackage"], r["packageMarkedResized"]))
    w("| Model | Records | %s |" % " | ".join(CHECK_TEXT.values()))
    w("| --- | --- |%s" % (" --- |" * len(CHECKS)))
    for m, d in res["models"].items():
        f = d["recordFunnel"]
        w("| %s | %d | %s |" % (m, f["n"], " | ".join("%d/%d" % (f[c], f["n"]) for c in CHECKS)))
    k = r["photosEqualPackage"]
    w("So the records join the same way (id and photo hash) for %d/%d; %s" % (
        k, r["inPackage"], "all of them." if k == r["inPackage"] else
        "the other %d were named from a package copy that is not the stored photo's bytes, so their outputs "
        "cannot be tied to the photo by hash." % (r["inPackage"] - k)))
    return L


# ============================================================================ export

def clean_text(s):
    """A model string as an extra name or descriptor: spaces collapsed, lower case, outer quotes and
    punctuation removed."""
    return " ".join(str(s).split()).lower().strip(" \"'`.,;:!?")


def dedupe(values, exclude=()):
    """Cleaned strings without repeats by the plan's normal form, without empty ones and without those whose
    normal form is in `exclude`."""
    seen, out = set(exclude), []
    for v in values:
        if not isinstance(v, str):
            continue
        c = clean_text(v)
        k = normalise(c)
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def names_doc(res, scan, model, created_ms):
    """(names.json document, counts) from the usable outputs of one model."""
    d = res["models"][model]
    items = []
    n = collections.Counter()
    for p in d["entries"]:
        if p["failed"] is not None:
            continue
        ans = p["_answer"]
        phone = normalise(scan.by_id[p["id"]].get("name") or "")
        name = vc.answer_name(ans)
        names = dedupe([name], exclude={phone})
        desc = dedupe(ans.get("attributes") or [])
        n["usable"] += 1
        n["nameEqualsPhone"] += normalise(clean_text(name)) == phone
        n["nameEmptyAfterCleaning"] += not normalise(clean_text(name))
        if not names and not desc:
            n["nothingToAdd"] += 1
            continue
        items.append({"kind": "entry", "id": p["id"], "cropSha256": p["cropSha256"], "names": names,
                      "descriptors": desc})
    n["items"] = len(items)
    n["withNames"] = sum(1 for i in items if i["names"])
    n["withDescriptors"] = sum(1 for i in items if i["descriptors"])
    doc = {"version": NAMES_VERSION, "source": "%s:%s" % (model, d["runDate"]), "createdMs": int(created_ms),
           "items": items}
    return doc, n


def validate_names(doc):
    """Problems of a names.json document against the plan's schema (empty when valid)."""
    p = []
    if not isinstance(doc, dict):
        return ["not a JSON object"]
    if doc.get("version") != NAMES_VERSION:
        p.append("version is not %d" % NAMES_VERSION)
    if not isinstance(doc.get("source"), str) or not doc.get("source"):
        p.append("source is not a non-empty string")
    if not isinstance(doc.get("createdMs"), int) or isinstance(doc.get("createdMs"), bool):
        p.append("createdMs is not an integer")
    if not isinstance(doc.get("items"), list):
        return p + ["items is not a list"]
    seen = set()
    for i, it in enumerate(doc["items"]):
        if not isinstance(it, dict):
            p.append("items[%d] is not an object" % i)
            continue
        if it.get("kind") not in ("entry", "record"):
            p.append("items[%d].kind is not entry or record" % i)
        if not isinstance(it.get("id"), str) or not it.get("id"):
            p.append("items[%d].id is not a non-empty string" % i)
        if not isinstance(it.get("cropSha256"), str) or not HEX64.match(it.get("cropSha256")):
            p.append("items[%d].cropSha256 is not 64 lower-case hex digits" % i)
        for k in ("names", "descriptors"):
            if not isinstance(it.get(k), list) or not all(isinstance(x, str) for x in it.get(k)):
                p.append("items[%d].%s is not a list of strings" % (i, k))
        key = (it.get("kind"), it.get("id"))
        if key in seen:
            p.append("items[%d]: %s %s listed twice" % (i, key[0], key[1]))
        seen.add(key)
    return p


def bind(scan, doc):
    """entry id -> (names, descriptors) for the items of a names file that are bound to this scan (kind
    entry, an entry of the scan, cropSha256 equal to the SHA-256 of its current photo), and the counts."""
    out = {}
    n = collections.Counter(items=len(doc["items"]))
    for it in doc["items"]:
        if it["kind"] != "entry":
            n["notEntry"] += 1
        elif it["id"] not in scan.by_id:
            n["notInScan"] += 1
        elif it["cropSha256"] != scan.sha[it["id"]]:
            n["photoDiffers"] += 1
        else:
            out[it["id"]] = (it["names"], it["descriptors"])
    n["bound"] = len(out)
    return out, n


# ============================================================================ sheet

SHEET_CSS = """
:root { color-scheme: light dark; --bg: #ffffff; --fg: #1b1b1b; --muted: #5a5a5a; --line: #d6d6d6;
  --card: #f7f7f7; --accent: #1f5fbf; --accent-fg: #ffffff; }
@media (prefers-color-scheme: dark) { :root { --bg: #151515; --fg: #ececec; --muted: #a9a9a9; --line: #3a3a3a;
  --card: #1f1f1f; --accent: #7aa7ff; --accent-fg: #0b0b0b; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 16px; }
h1 { font-size: 24px; margin: 8px 0 4px; }
h2 { font-size: 19px; margin: 28px 0 6px; }
p { margin: 6px 0; max-width: 70ch; }
.muted { color: var(--muted); font-size: 14px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; margin-top: 14px; }
.card { border: 1px solid var(--line); border-radius: 8px; background: var(--card); padding: 10px; }
.code { font-weight: 700; font-size: 20px; margin-bottom: 6px; }
.photo { width: 100%; height: 230px; object-fit: contain; background: #000; border-radius: 4px; display: block; }
.nophoto { height: 230px; display: flex; align-items: center; justify-content: center; border: 1px dashed var(--line); }
label { display: block; font-size: 14px; color: var(--muted); margin-top: 8px; }
input[type=text] { width: 100%; font: inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 4px;
  background: var(--bg); color: var(--fg); }
.none input[type=text] { max-width: 420px; display: block; margin-top: 6px; }
.bar { margin: 24px 0 40px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
button { font: inherit; padding: 9px 18px; border: 0; border-radius: 6px; background: var(--accent); color: var(--accent-fg); cursor: pointer; }
"""

SHEET_JS = r"""
var KEY = "findback_query_sheet_" + SHEET_ID;
function csvField(s) {
  s = String(s);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function value(id) { var el = document.getElementById(id); return el ? el.value.trim() : ""; }
function queryRows() {
  var rows = [];
  ITEMS.forEach(function (it) {
    ["broad", "specific"].forEach(function (t) {
      var v = value(t + "_" + it.code);
      if (v) rows.push([v, t, it.code, it.id]);
    });
  });
  for (var i = 1; i <= NONE_INPUTS; i++) {
    var v = value("none_" + i);
    if (v) rows.push([v, "none", "", ""]);
  }
  return rows;
}
function csvText(rows) {
  return [COLUMNS].concat(rows).map(function (r) { return r.map(csvField).join(","); }).join("\r\n") + "\r\n";
}
function save() {
  var d = {};
  Array.prototype.forEach.call(document.querySelectorAll("input[type=text]"), function (el) { d[el.id] = el.value; });
  try { localStorage.setItem(KEY, JSON.stringify(d)); } catch (e) {}
}
function restore() {
  var d = null;
  try { d = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) {}
  if (!d) return;
  Object.keys(d).forEach(function (k) { var el = document.getElementById(k); if (el) el.value = d[k]; });
}
function download() {
  var rows = queryRows();
  var n = { broad: 0, specific: 0, none: 0 };
  rows.forEach(function (r) { n[r[1]]++; });
  var blob = new Blob([csvText(rows)], { type: "text/csv;charset=utf-8" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "queries.csv";
  document.body.appendChild(a);
  a.click();
  setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  document.getElementById("status").textContent = rows.length + " queries in queries.csv (" + n.broad + " general, " +
    n.specific + " specific, " + n.none + " not in this scan).";
}
if (typeof document !== "undefined" && document.getElementById) {
  restore();
  document.addEventListener("input", save);
  document.getElementById("dl").addEventListener("click", download);
}
"""


def image_mime(b):
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def sheet_html(scan, none_inputs=3):
    """The query sheet: photos and codes only (no name, label, room or model output)."""
    items = [{"code": scan.codes[e["id"]], "id": e["id"]} for e in scan.entries]
    sheet_id = sha256_bytes("\n".join("%s %s" % (e["id"], scan.sha[e["id"]]) for e in scan.entries).encode())[:12]
    cards = []
    for it in items:
        c, p = it["code"], scan.photo[it["id"]]
        if p:
            with open(p, "rb") as f:
                b = f.read()
            img = '<img class="photo" alt="photo %s" src="data:%s;base64,%s">' % (
                c, image_mime(b), base64.b64encode(b).decode("ascii"))
        else:
            img = '<div class="nophoto muted">no photo</div>'
        cards.append(
            '<div class="card" data-code="%(c)s">\n<div class="code">%(c)s</div>\n%(img)s\n'
            '<label for="broad_%(c)s">A general word you would type to find this</label>\n'
            '<input type="text" id="broad_%(c)s" autocomplete="off">\n'
            '<label for="specific_%(c)s">A specific name you would type for this</label>\n'
            '<input type="text" id="specific_%(c)s" autocomplete="off">\n</div>' % {"c": html.escape(c), "img": img})
    nones = "\n".join('<input type="text" id="none_%d" autocomplete="off" aria-label="not in this scan %d">' % (i, i)
                      for i in range(1, none_inputs + 1))
    data = "var ITEMS = %s;\nvar COLUMNS = %s;\nvar NONE_INPUTS = %d;\nvar SHEET_ID = %s;\n" % (
        json.dumps(items), json.dumps(CSV_COLUMNS), none_inputs, json.dumps(sheet_id))
    script = (data + SHEET_JS).replace("</", "<\\/")
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search words sheet</title>
<style>%(css)s</style>
</head>
<body>
<main>
<h1>What would you type?</h1>
<p>Each photo below shows one thing your phone saved during a scan of your home. For each one, write what you
would type into the app's search box to find it: a general word, a more specific name, or both. Both boxes are
optional; leave a box empty when nothing comes to mind. Write the words before looking at any search results.</p>
<p class="muted">%(n)d photos. What you type is kept in this browser until you download the file. Nothing is
sent anywhere.</p>
<div class="grid">
%(cards)s
</div>
<h2>Things that are not in this scan</h2>
<div class="none">
<p>Up to three things you might search for that are not among these photos.</p>
%(nones)s
</div>
<div class="bar"><button id="dl" type="button">Download queries.csv</button><span id="status" class="muted"></span></div>
</main>
<script>
%(script)s
</script>
</body>
</html>
""" % {"css": SHEET_CSS, "n": len(items), "cards": "\n".join(cards), "nones": nones, "script": script}


# ============================================================================ queries and freeze

def read_queries(path):
    """(rows, problems) of a queries file; rows are dicts with the four columns and `row` (1-based)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []
        raw = list(rd)
    problems = []
    if header != CSV_COLUMNS:
        problems.append("header is %s, expected %s" % (",".join(header), ",".join(CSV_COLUMNS)))
        return [], problems
    rows = []
    for i, r in enumerate(raw, start=1):
        if None in r:
            problems.append("row %d: more columns than the header" % i)
            continue
        x = {k: (r.get(k) or "").strip() for k in CSV_COLUMNS}
        x["row"] = i
        if x["type"] not in QUERY_TYPES:
            problems.append("row %d: type %r is not one of %s" % (i, x["type"], ", ".join(QUERY_TYPES)))
        if not words(x["query"]):
            problems.append("row %d: the query has no letter or digit" % i)
        if x["type"] == "none" and (x["target_code"] or x["target_id"]):
            problems.append("row %d: a none query has a target" % i)
        if x["type"] in TARGET_TYPES and not x["target_id"]:
            problems.append("row %d: a %s query has no target_id" % (i, x["type"]))
        rows.append(x)
    return rows, problems


def freeze_path(queries):
    return queries + ".frozen.json"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_frozen(queries):
    """(freeze record, None) when the record exists and has the file's hash, else (None, reason)."""
    fp = freeze_path(queries)
    if not os.path.exists(fp):
        return None, "no freeze record %s (run: name_search.py freeze %s)" % (fp, queries)
    rec = read_json(fp)
    if not isinstance(rec, dict) or "sha256" not in rec:
        return None, "the freeze record %s is unreadable" % fp
    sha = sha256_file(queries)
    if rec["sha256"] != sha:
        return None, "%s changed after it was frozen (sha256 now %s, frozen %s)" % (queries, sha, rec["sha256"])
    return rec, None


def cmd_freeze(args, out):
    rows, problems = read_queries(args.queries)
    if problems:
        print("Not frozen: %s" % "; ".join(problems), file=out)
        return 2
    sha = sha256_file(args.queries)
    fp = freeze_path(args.queries)
    old = read_json(fp)
    if isinstance(old, dict) and old.get("sha256") == sha:
        print("Already frozen at %s: sha256 %s (%s)" % (old.get("frozenAtUtc"), sha, fp), file=out)
        return 0
    if os.path.exists(fp):
        print("Refusing: %s holds another hash (%s). The queries changed after they were frozen. If no ranking "
              "has been looked at, delete the record and freeze again." % (fp, (old or {}).get("sha256")), file=out)
        return 3
    by = collections.Counter(r["type"] for r in rows)
    rec = {"file": os.path.basename(args.queries), "sha256": sha, "rows": len(rows),
           "byType": {t: by.get(t, 0) for t in QUERY_TYPES}, "frozenAtUtc": utc_now(),
           "script": "analysis/name_search.py", "scriptSha256": sha256_file(os.path.abspath(__file__))}
    write_json(fp, rec)
    print("Frozen %s: sha256 %s, %d rows (%s) at %s -> %s" % (
        args.queries, sha, len(rows), ", ".join("%s %d" % (t, by.get(t, 0)) for t in QUERY_TYPES),
        rec["frozenAtUtc"], fp), file=out)
    return 0


# ============================================================================ run

def condition_label(doc, used):
    lab = doc["source"]
    k = 2
    while lab in used:
        lab = "%s #%d" % (doc["source"], k)
        k += 1
    return lab


def evaluate(scan, queries, docs):
    """Every query in every condition. `docs`: [(path, names document)]. Returns the result dict."""
    base = scan.items()
    conds = [{"key": "phone", "file": None, "items": base, "extra": False, "binding": None}]
    for path, doc in docs:
        bound, n = bind(scan, doc)
        conds.append({"key": condition_label(doc, {c["key"] for c in conds}), "file": path, "items": scan.items(bound),
                      "extra": True, "binding": dict(n), "source": doc["source"]})
    per = []
    for q in queries:
        target = q["target_id"] or None
        row = {"row": q["row"], "query": q["query"], "type": q["type"], "target": target,
               "targetCode": scan.codes.get(target) if target else None, "conditions": collections.OrderedDict()}
        for c in conds:
            res = search(c["items"], q["query"], c["extra"])
            ids = [it.id for it, _ in res]
            cls = {it.id: k for it, k in res}
            via = via_extra(c["items"], q["query"]) if c["extra"] else set()
            byid = {it.id: it for it in c["items"]}
            review = [{"id": i, "code": scan.codes[i], "rank": ids.index(i) + 1, "class": cls[i],
                       "phoneName": byid[i].name, "matched": [list(h) for h in extra_hits(byid[i], q["query"])]}
                      for i in ids if i in via and i != target]
            row["conditions"][c["key"]] = {
                "n": len(ids), "rank": ids.index(target) + 1 if target in cls else None,
                "class": cls.get(target) if target else None, "viaExtra": len(via),
                "targetViaExtra": bool(target and target in via), "toReview": review,
                "top": [scan.codes[i] for i in ids[:3]]}
        per.append(row)
    summary = collections.OrderedDict()
    for c in conds:
        s = collections.OrderedDict()
        for t in TARGET_TYPES + ("all",):
            sel = [r["conditions"][c["key"]] for r in per if (r["type"] == t if t != "all" else r["type"] in TARGET_TYPES)]
            ranks = [x["rank"] for x in sel if x["rank"]]
            s[t] = {"n": len(sel), "found": len(ranks), "top1": sum(1 for k in ranks if k == 1),
                    "top3": sum(1 for k in ranks if k <= 3), "medianRank": statistics.median(ranks) if ranks else None,
                    "classes": dict(sorted(collections.Counter(x["class"] for x in sel if x["rank"]).items()))}
        nn = [r["conditions"][c["key"]]["n"] for r in per if r["type"] == "none"]
        s["none"] = {"n": len(nn), "zeroResults": sum(1 for k in nn if k == 0), "results": nn}
        s["viaExtraResults"] = sum(r["conditions"][c["key"]]["viaExtra"] for r in per)
        s["toReview"] = sum(len(r["conditions"][c["key"]]["toReview"]) for r in per)
        summary[c["key"]] = s
    examples = collections.OrderedDict()
    for c in conds[1:]:
        ex = {"helped": [], "same": [], "hurt": []}
        for r in per:
            if r["type"] not in TARGET_TYPES:
                continue
            a, b = r["conditions"]["phone"]["rank"], r["conditions"][c["key"]]["rank"]
            if b and (not a or b < a):
                ex["helped"].append(r["row"])
            elif a == b:
                ex["same"].append(r["row"])
            else:
                ex["hurt"].append(r["row"])
        examples[c["key"]] = ex
    return {"conditions": [{k: v for k, v in c.items() if k != "items"} for c in conds], "perQuery": per,
            "summary": summary, "examples": examples}


def fmt_rank(x):
    if x["rank"] is None:
        return "not found (%d results)" % x["n"]
    return "%d of %d (class %d)%s" % (x["rank"], x["n"], x["class"], ", via extra" if x["targetViaExtra"] else "")


def fmt_med(m):
    return "–" if m is None else ("%g" % m)


def run_md(res):
    L = []
    w = L.append
    fz = res["frozen"]
    w("# Find by name: offline run")
    w("")
    w("Generated by `analysis/name_search.py run` at %s. Matching rule: `FindBack/docs/PLAN_v0.9_find_by_name.md`." %
      res["generatedUtc"])
    w("")
    w("- Inventory: `%s`, map %s, %d scan entries (the same inventory in every condition)." % (
        res["pull"], res["mapId"], res["nEntries"]))
    w("- Queries: `%s`, **frozen sha256 `%s`** at %s; %d rows (%s)." % (
        res["queriesFile"], fz["sha256"], fz.get("frozenAtUtc"), fz.get("rows", 0),
        ", ".join("%s %d" % (t, (fz.get("byType") or {}).get(t, 0)) for t in QUERY_TYPES)))
    if res.get("scriptChanged"):
        w("- NOTE: `analysis/name_search.py` changed after the queries were frozen; any change of rules is a change "
          "to report.")
    w("")
    w("## Conditions")
    w("")
    w("| Condition | Names file | Items in file | Bound to this scan | Not bound (photo differs / not in scan / record) |")
    w("| --- | --- | --- | --- | --- |")
    for c in res["conditions"]:
        if not c["extra"]:
            w("| phone | none: the phone's names and labels alone | – | – | – |")
            continue
        b = c["binding"]
        w("| %s | `%s` | %d | %d/%d | %d / %d / %d |" % (c["key"], c["file"], b["items"], b["bound"], b["items"],
                                                        b.get("photoDiffers", 0), b.get("notInScan", 0), b.get("notEntry", 0)))
    w("")
    w("## Summary")
    w("")
    w("Ranks are 1-based positions in the result list; the median rank is over the found targets.")
    w("")
    w("| Condition | Query type | Target found | In top 1 | In top 3 | Median rank (n found) | Classes of found targets |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for key, s in res["summary"].items():
        for t in TARGET_TYPES + ("all",):
            x = s[t]
            w("| %s | %s | %d/%d | %d/%d | %d/%d | %s (n = %d) | %s |" % (
                key, t, x["found"], x["n"], x["top1"], x["n"], x["top3"], x["n"], fmt_med(x["medianRank"]), x["found"],
                ", ".join("%s: %d" % (k, v) for k, v in x["classes"].items()) or "–"))
    w("")
    w("Queries for something not in this scan (every result is not the target; 0 is expected):")
    w("")
    w("| Condition | Queries | With 0 results | Results per query |")
    w("| --- | --- | --- | --- |")
    for key, s in res["summary"].items():
        x = s["none"]
        w("| %s | %d | %d/%d | %s |" % (key, x["n"], x["zeroResults"], x["n"], ", ".join(str(k) for k in x["results"]) or "–"))
    w("")
    w("Results present only via extra names, over all queries: %s." % ("; ".join(
        "%s %d (not the target: %d, to review)" % (k, s["viaExtraResults"], s["toReview"])
        for k, s in res["summary"].items() if k != "phone") or "none"))
    w("")
    w("## Per query")
    w("")
    keys = list(res["summary"])
    w("| Row | Query | Type | Target | %s |" % " | ".join(keys))
    w("| --- | --- | --- | --- |%s" % (" --- |" * len(keys)))
    for r in res["perQuery"]:
        w("| %d | %s | %s | %s | %s |" % (r["row"], md_cell(r["query"]), r["type"], r["targetCode"] or "–",
                                         " | ".join(fmt_rank(r["conditions"][k]) if r["target"]
                                                    else "%d results" % r["conditions"][k]["n"] for k in keys)))
    w("")
    byrow = {r["row"]: r for r in res["perQuery"]}
    for key, ex in res["examples"].items():
        w("## Examples: %s against phone" % key)
        w("")
        for bucket, text in (("helped", "Helped (the target appears, or its rank improves)"),
                             ("same", "No difference (same rank, or not found in both)"),
                             ("hurt", "Hurt (the target's rank is worse)")):
            rows = ex[bucket]
            w("%s: %d of %d targeted queries." % (text, len(rows), len(ex["helped"]) + len(ex["same"]) + len(ex["hurt"])))
            w("")
            for i in rows:
                r = byrow[i]
                w("- row %d `%s` (%s, target %s): phone %s; %s %s" % (
                    i, r["query"], r["type"], r["targetCode"], fmt_rank(r["conditions"]["phone"]), key,
                    fmt_rank(r["conditions"][key])))
            if rows:
                w("")
        rev = [(r, x) for r in res["perQuery"] for x in r["conditions"][key]["toReview"]]
        w("To review: results present only via extra names that are not the target (%d). Whether each is wrong is "
          "for the owner to say." % len(rev))
        w("")
        for r, x in rev:
            w("- row %d `%s` (%s%s): %s (phone name `%s`), rank %d, class %d, via %s" % (
                r["row"], r["query"], r["type"], ", target %s" % r["targetCode"] if r["target"] else "", x["code"],
                x["phoneName"], x["rank"], x["class"], ", ".join("%s `%s`" % (f, t) for f, t in x["matched"]) or "?"))
        w("")
    return "\n".join(L)


def md_cell(s):
    return "`%s`" % s.replace("|", "\\|").replace("`", "'")


def cmd_run(args, out):
    fz, why = check_frozen(args.queries)
    if why:
        print("Refusing to run: %s" % why, file=out)
        return 3
    queries, problems = read_queries(args.queries)
    if problems:
        print("Refusing to run: %s" % "; ".join(problems), file=out)
        return 2
    scan = Scan(args.pull)
    bad = []
    for q in queries:
        if q["target_id"] and q["target_id"] not in scan.by_id:
            bad.append("row %d: target %s is not an entry of this scan" % (q["row"], q["target_id"]))
        elif q["target_id"] and q["target_code"] and q["target_code"] != scan.codes[q["target_id"]]:
            bad.append("row %d: code %s, but this scan's code of %s is %s (sheet of another pull?)" % (
                q["row"], q["target_code"], q["target_id"], scan.codes[q["target_id"]]))
    if bad:
        print("Refusing to run: %s" % "; ".join(bad), file=out)
        return 2
    docs = []
    for p in args.names:
        doc = read_json(p)
        problems = validate_names(doc)
        if problems:
            print("Refusing to run: %s: %s" % (p, "; ".join(problems)), file=out)
            return 2
        docs.append((p, doc))
    res = evaluate(scan, queries, docs)
    script_changed = bool(fz.get("scriptSha256")) and fz["scriptSha256"] != sha256_file(os.path.abspath(__file__))
    res.update({"generatedUtc": utc_now(), "pull": rel(scan.pull), "mapId": scan.map_id, "nEntries": len(scan.entries),
                "queriesFile": rel(args.queries), "frozen": fz, "scriptChanged": script_changed})
    for c in res["conditions"]:
        if c["file"]:
            c["file"] = rel(c["file"])
    out_dir = args.out or os.path.join(default_dir(args.pull),
                                       "run_" + os.path.splitext(os.path.basename(args.queries))[0])
    os.makedirs(out_dir, exist_ok=True)
    md = run_md(res)
    with open(os.path.join(out_dir, "run.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    write_json(os.path.join(out_dir, "run.json"), res)
    warn_if_tracked(out_dir, out)
    print("Queries %s frozen sha256 %s (%s)%s" % (args.queries, fz["sha256"], fz.get("frozenAtUtc"),
                                                  "; NOTE: name_search.py changed since" if script_changed else ""), file=out)
    print("| Condition | Query type | Target found | In top 1 | In top 3 | Median rank (n found) |", file=out)
    print("| --- | --- | --- | --- | --- | --- |", file=out)
    for key, s in res["summary"].items():
        for t in TARGET_TYPES + ("all",):
            x = s[t]
            print("| %s | %s | %d/%d | %d/%d | %d/%d | %s (n = %d) |" % (
                key, t, x["found"], x["n"], x["top1"], x["n"], x["top3"], x["n"], fmt_med(x["medianRank"]), x["found"]),
                file=out)
        print("| %s | none | with 0 results: %d/%d | | | |" % (key, s["none"]["zeroResults"], s["none"]["n"]), file=out)
    print("Written: %s, %s" % (os.path.join(out_dir, "run.md"), os.path.join(out_dir, "run.json")), file=out)
    return 0


def rel(p):
    a = os.path.abspath(p)
    return os.path.relpath(a, REPO) if a.startswith(REPO + os.sep) else a


# ============================================================================ commands

def cmd_join(args, out):
    scan = Scan(args.pull)
    res = join(scan, args.vlm_out, args.model or None)
    for line in join_lines(res):
        print(line, file=out)
    path = args.out or os.path.join(default_dir(args.pull), "join.json")
    write_json(path, public_join(res))
    warn_if_tracked(path, out)
    print("Written: %s" % path, file=out)
    return 0


def cmd_export(args, out):
    scan = Scan(args.pull)
    res = join(scan, args.vlm_out, [args.model])
    if not res["models"][args.model]["run"]:
        print("No finished run of %s under %s" % (args.model, args.vlm_out), file=out)
        return 2
    created = args.created_ms if args.created_ms is not None else int(time.time() * 1000)
    doc, n = names_doc(res, scan, args.model, created)
    problems = validate_names(doc)
    if problems:
        print("Internal error, the document is not valid: %s" % "; ".join(problems), file=out)
        return 1
    path = args.out or os.path.join(default_dir(args.pull), "names_%s.json" % args.model)
    write_json(path, doc)
    warn_if_tracked(path, out)
    print("%s (%s): %d/%d entries usable; %d items written (%d with a name, %d with descriptors); names left out: "
          "%d equal to the phone's own name, %d empty after cleaning; %d usable entries with nothing to add" % (
              doc["source"], res["models"][args.model]["run"], n["usable"], len(scan.entries), n["items"],
              n["withNames"], n["withDescriptors"], n["nameEqualsPhone"], n["nameEmptyAfterCleaning"],
              n["nothingToAdd"]), file=out)
    print("Written: %s" % path, file=out)
    return 0


def cmd_sheet(args, out):
    scan = Scan(args.pull)
    path = args.out or os.path.join(default_dir(args.pull), "query_sheet.html")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(sheet_html(scan))
    warn_if_tracked(path, out)
    print("Sheet with %d photos (%d missing), codes %s..%s: %s" % (
        len(scan.entries), sum(1 for e in scan.entries if not scan.photo[e["id"]]),
        code(0), code(max(0, len(scan.entries) - 1)), path), file=out)
    return 0


def main(argv=None, out=sys.stdout):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("join", help="join the saved model outputs to the scan's entries and photos")
    p.add_argument("pull")
    p.add_argument("--vlm-out", default=VLM_OUT, help="the naming experiment's folder (default analysis/out/vlm)")
    p.add_argument("--model", action="append", choices=vc.CONTESTANTS, help="only these models (repeatable)")
    p.add_argument("--out", help="JSON file (default analysis/out/name_search/<pull>/join.json)")
    p = sub.add_parser("export", help="names.json for the phone from one model's usable outputs")
    p.add_argument("pull")
    p.add_argument("--model", required=True, choices=vc.CONTESTANTS)
    p.add_argument("--vlm-out", default=VLM_OUT)
    p.add_argument("--out", help="file (default analysis/out/name_search/<pull>/names_<model>.json)")
    p.add_argument("--created-ms", type=int, help=argparse.SUPPRESS)
    p = sub.add_parser("sheet", help="HTML page for the owner to write queries")
    p.add_argument("pull")
    p.add_argument("--out", help="file (default analysis/out/name_search/<pull>/query_sheet.html)")
    p = sub.add_parser("freeze", help="record the hash of a queries file")
    p.add_argument("queries")
    p = sub.add_parser("run", help="every frozen query in every condition")
    p.add_argument("pull")
    p.add_argument("--queries", required=True)
    p.add_argument("--names", action="append", required=True, help="names.json (repeatable)")
    p.add_argument("--out", help="folder (default analysis/out/name_search/<pull>/run_<queries file>/)")
    args = ap.parse_args(argv)
    return {"join": cmd_join, "export": cmd_export, "sheet": cmd_sheet, "freeze": cmd_freeze, "run": cmd_run}[args.cmd](
        args, out)


if __name__ == "__main__":
    sys.exit(main())
