#!/usr/bin/env python3
"""Tests of the offline find-by-name harness (analysis/name_search.py). Synthetic data only: the cases
shared with the app's unit test (FindBack/app/src/test/resources/name_search_cases.json), and pulls,
saved model outputs, names files and queries written to a temporary folder. No phone data and no image
(the "photos" are a few bytes that start like a JPEG file and are never decoded).

    python3 analysis/test_name_search.py
"""
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# The synthetic runs use aliases only: point the loaders at a model file that does not exist, so the private
# one is never read by these tests.
_NO_MODELS = tempfile.mkdtemp(prefix="ns_nomodels_")
os.environ["FB_VLM_MODELS"] = os.path.join(_NO_MODELS, "missing.json")
import name_search as ns  # noqa: E402

CASES = os.path.join(REPO, "FindBack", "app", "src", "test", "resources", "name_search_cases.json")
MAP_ID = "m0000001"


# ----------------------------------------------------------------------------- builders

def photo(tag):
    """A few bytes that start like a JPEG file (never decoded)."""
    return b"\xff\xd8\xff\xe0" + tag.encode() + b"\xff\xd9"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)


def write_pull(root, entries, records=()):
    """A pulled folder: inventory.json and inventory/<id>.jpg, map.json with two rooms, records.json and
    crops/<id>.jpg. entries: dicts id, name, labels, room, last, photo[, container]."""
    inv = []
    for e in entries:
        inv.append({"id": e["id"], "name": e["name"], "nameEdited": False, "label": e["labels"][0],
                    "labelVotes": {x: 0.5 for x in e["labels"]}, "mapId": MAP_ID, "roomId": e.get("room", "r1"),
                    "container": e.get("container"), "cropFile": "inventory/%s.jpg" % e["id"],
                    "lastSeenMs": e["last"]})
        write(os.path.join(root, "inventory", "%s.jpg" % e["id"]), e["photo"])
    write(os.path.join(root, "inventory.json"), json.dumps({"version": 1, "entries": inv}))
    write(os.path.join(root, "map.json"), json.dumps({"version": 1, "mapId": MAP_ID, "rooms": [
        {"id": "r1", "name": "Atrium"}, {"id": "r2", "name": "Den"}]}))
    recs = []
    for r in records:
        recs.append({"id": r["id"], "name": r["name"], "cropFile": "crops/%s.jpg" % r["id"], "sessionId": "s1"})
        write(os.path.join(root, "crops", "%s.jpg" % r["id"]), r["photo"])
    write(os.path.join(root, "records.json"), json.dumps(recs))
    return root


def answer(name, kind="one item", attributes=(), visible_text=""):
    return {"name": name, "category": "other", "attributes": list(attributes), "visible_text": visible_text,
            "kind": kind}


def req(item, ans=None, text=None, nbytes=0, repeat=1, phase="seq"):
    """One saved request row as vlm_remote writes it."""
    return {"phase": phase, "variant": "crop-full", "repeat": repeat, "item": item,
            "set": "S" if item.startswith("inv:") else "R", "status": 200, "error": None, "finish": "stop",
            "text": text if text is not None else json.dumps(ans), "answer": ans,
            "name": (ans or {}).get("name") or None, "jpeg_bytes": nbytes}


def write_vlm(root, package, runs):
    """A naming experiment folder: package/items.json and the crop-full images, and finished runs.
    package: [(item id, set, bytes)]; runs: {"<site folder>/<run folder>": (end time, [rows])}."""
    items = []
    for iid, s, data in package:
        fname = "%s__crop-full.jpg" % iid.replace(":", "_")
        write(os.path.join(root, "package", "images", fname), data)
        items.append({"id": iid, "set": s, "variants": {"crop-full": [fname]}})
    write(os.path.join(root, "package", "items.json"), json.dumps({"version": 1, "items": items}))
    for d, (t, rows) in runs.items():
        write(os.path.join(root, d, "status.json"), json.dumps({"state": "done", "t": t}))
        write(os.path.join(root, d, "requests.jsonl"), "".join(json.dumps(r) + "\n" for r in rows))
    return root


def run_main(argv):
    buf = _Buf()
    code = ns.main(argv, out=buf)
    return code, buf.text


class _Buf:
    def __init__(self):
        self.parts = []

    def write(self, s):
        self.parts.append(s)

    def flush(self):
        pass

    @property
    def text(self):
        return "".join(self.parts)


def join_scene(root):
    """Nine entries, each failing (or passing) one check of the join, and two tapped records."""
    P = {k: photo(k) for k in ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "r1", "r2")}
    ents = [
        {"id": "e1", "name": "widget", "labels": ["widget"], "last": 900, "photo": P["e1"]},    # usable
        {"id": "e2", "name": "gadget", "labels": ["gadget"], "last": 800, "photo": P["e2"]},    # usable, name = phone
        {"id": "e3", "name": "widget", "labels": ["widget"], "last": 700, "photo": P["e3"]},    # several items
        {"id": "e4", "name": "gizmo", "labels": ["gizmo"], "last": 600, "photo": P["e4"]},      # empty name
        {"id": "e5", "name": "gizmo", "labels": ["gizmo"], "last": 500, "photo": P["e5"]},      # no output
        {"id": "e6", "name": "sprocket", "labels": ["sprocket"], "last": 400, "photo": P["e6"]},  # package differs
        {"id": "e7", "name": "sprocket", "labels": ["sprocket"], "last": 300, "photo": P["e7"]},  # unparsable
        {"id": "e8", "name": "flange", "labels": ["flange"], "last": 200, "photo": P["e8"]},    # usable, nothing new
        {"id": "e9", "name": "flange", "labels": ["flange"], "last": 100, "photo": P["e9"]},    # request bytes differ
    ]
    pull = write_pull(os.path.join(root, "pull"), ents, records=[
        {"id": "r1", "name": "item 1", "photo": P["r1"]}, {"id": "r2", "name": "item 2", "photo": P["r2"]}])
    n = {k: len(v) for k, v in P.items()}
    rows = [
        req("inv:e1", answer("Blue Widget Box", attributes=["Blue", "plastic", "blue"]), nbytes=n["e1"]),
        req("inv:e1", answer("other name"), nbytes=n["e1"], repeat=2),
        req("inv:e2", answer("Gadget", attributes=["small"]), nbytes=n["e2"]),
        req("inv:e3", answer("pile of widgets", kind="several items"), nbytes=n["e3"]),
        req("inv:e4", answer("", attributes=["red"]), nbytes=n["e4"]),
        req("inv:e6", answer("cog"), nbytes=n["e6"]),
        req("inv:e7", None, text="sorry, no JSON here", nbytes=n["e7"]),
        req("inv:e8", answer("Flange.", attributes=[]), nbytes=n["e8"]),
        req("inv:e9", answer("flat flange"), nbytes=n["e9"] + 7),
        req("rec:r1", answer("bolt"), nbytes=n["r1"]),
        req("rec:r2", answer("nut"), nbytes=n["r2"] + 100),
    ]
    package = [("inv:" + e["id"], "S", photo("resized-" + e["id"]) if e["id"] == "e6" else e["photo"]) for e in ents]
    package += [("rec:r1", "R", P["r1"]), ("rec:r2", "R", photo("resized-r2"))]
    vlm = write_vlm(os.path.join(root, "vlm"), package, {
        "cluster_runs/2026-11-18_model_a": (100.0, rows),
        # an additional-only site, earlier in time: never the primary run (vlm_report.ADDITIONAL_SITES)
        "cluster2_runs/2026-11-18_model_a": (50.0, [req("inv:e1", answer("decoy"), nbytes=n["e1"])]),
        # a set-aside attempt: its folder name does not count as a run
        "ws_runs/2026-11-17_model_a_contaminated_1": (10.0, [req("inv:e5", answer("decoy"), nbytes=n["e5"])]),
    })
    return pull, vlm, P


# ----------------------------------------------------------------------------- matching rule

def case_items():
    with open(CASES) as f:
        d = json.load(f)
    items = [ns.Item(id=x["id"], name=x["name"], labels=x["labels"], container=x["container"], room=x["room"],
                     last_seen_ms=x["lastSeenMs"], extra_names=x["extraNames"], descriptors=x["descriptors"])
             for x in d["items"]]
    return items, d["cases"]


def test_shared_cases_of_the_plan():
    items, cases = case_items()
    assert len(cases) == 21, len(cases)
    bad = []
    for c in cases:
        q = c["query"]
        off, on = ns.search(items, q, False), ns.search(items, q, True)
        got = {"off": [i.id for i, _ in off], "on": [i.id for i, _ in on], "viaExtra": sorted(ns.via_extra(items, q))}
        want = {"off": c["off"], "on": c["on"], "viaExtra": sorted(c["viaExtra"])}
        if "offClasses" in c:
            got["offClasses"], want["offClasses"] = [k for _, k in off], c["offClasses"]
        if "onClasses" in c:
            got["onClasses"], want["onClasses"] = [k for _, k in on], c["onClasses"]
        if got != want:
            bad.append("%r: got %s, want %s" % (q, got, want))
    assert not bad, "\n".join(bad)


def test_normalise_and_words():
    assert ns.normalise("  CUP!! ") == "cup"
    assert ns.normalise("Coffee-Cup") == "coffee cup"
    assert ns.normalise("USB-C 2.0\tcable\n") == "usb c 2 0 cable"
    assert ns.normalise("Café Crème") == "café crème"
    assert ns.normalise("½ cup") == "cup"          # a vulgar fraction is neither a letter nor a decimal digit
    assert ns.normalise("") == "" and ns.normalise(None) == "" and ns.normalise("!?-") == ""
    assert ns.words("My keys") == ["my", "keys"]


def test_query_without_words_lists_everything_by_last_seen():
    items, _ = case_items()
    for q in ("", "   ", "!?"):
        r = ns.search(items, q, True)
        assert [i.id for i, _ in r] == ["f", "e", "d", "c", "b", "a"] and all(k is None for _, k in r)
        assert ns.via_extra(items, q) == set()


def test_ties_on_last_seen_go_by_id():
    a = ns.Item(id="b2", name="widget", last_seen_ms=5)
    b = ns.Item(id="a1", name="widget", last_seen_ms=5)
    c = ns.Item(id="c3", name="widget", last_seen_ms=9)
    assert [i.id for i, _ in ns.search([a, b, c], "widget", False)] == ["c3", "a1", "b2"]


def test_extra_hits_name_the_matching_extra():
    items, _ = case_items()
    c = [i for i in items if i.id == "c"][0]
    assert ns.extra_hits(c, "red cup") == [("name", "coffee cup"), ("descriptor", "red")]
    f = [i for i in items if i.id == "f"][0]
    assert ns.extra_hits(f, "cup") == [("name", "teacup")]
    assert ns.extra_hits(f, "up") == []


def test_codes():
    assert [ns.code(i) for i in (0, 1, 25, 26, 27, 51, 701, 702)] == ["A", "B", "Z", "AA", "AB", "AZ", "ZZ", "AAA"]


# ----------------------------------------------------------------------------- join and export

def test_join_on_a_synthetic_pull():
    root = tempfile.mkdtemp(prefix="ns_join_")
    try:
        pull, vlm, P = join_scene(root)
        scan = ns.Scan(pull)
        res = ns.join(scan, vlm, ["model_a", "model_b"])
        assert res["nEntries"] == 9 and res["mapId"] == MAP_ID and res["photosEqualPackage"] == 8
        a = res["models"]["model_a"]
        assert a["run"] == "cluster/2026-11-18_model_a" and a["runDate"] == "2026-11-18", a["run"]
        assert dict(a["funnel"]) == {"n": 9, "output": 8, "parse": 7, "photo": 5, "one_item": 4, "name": 3}, a["funnel"]
        failed = {p["id"]: p["failed"] for p in a["entries"]}
        assert failed == {"e1": None, "e2": None, "e3": "one_item", "e4": "name", "e5": "output", "e6": "photo",
                          "e7": "parse", "e8": None, "e9": "photo"}, failed
        e1 = [p for p in a["entries"] if p["id"] == "e1"][0]
        assert e1["cropSha256"] == sha(P["e1"]) and e1["_answer"]["name"] == "Blue Widget Box"
        assert dict(a["recordFunnel"]) == {"n": 2, "output": 2, "parse": 2, "photo": 1, "one_item": 1, "name": 1}
        assert res["records"] == {"inPackage": 2, "inPull": 2, "photosEqualPackage": 1, "packageMarkedResized": 0}
        assert a["fields"]["answerFields"] == ["attributes", "category", "kind", "name", "visible_text"]
        assert a["fields"]["usable"] == 3 and a["fields"]["withAttributes"] == 2
        b = res["models"]["model_b"]
        assert b["run"] is None and b["funnel"]["output"] == 0 and b["funnel"]["n"] == 9
        out = os.path.join(root, "join.json")
        code, text = run_main(["join", pull, "--vlm-out", vlm, "--model", "model_a", "--out", out])
        assert code == 0 and "| model_a | cluster/2026-11-18_model_a | 9 | 8/9 | 7/9 | 5/9 | 4/9 | 3/9 |" in text, text
        with open(out) as f:
            body = f.read()
        assert "Blue Widget Box" not in body and "_answer" not in body and json.loads(body)["nEntries"] == 9
    finally:
        shutil.rmtree(root)


def test_export_writes_the_plans_schema():
    root = tempfile.mkdtemp(prefix="ns_export_")
    try:
        pull, vlm, P = join_scene(root)
        out = os.path.join(root, "names.json")
        code, text = run_main(["export", pull, "--model", "model_a", "--vlm-out", vlm, "--out", out,
                               "--created-ms", "1795000000000"])
        assert code == 0, text
        with open(out) as f:
            doc = json.load(f)
        assert ns.validate_names(doc) == []
        assert set(doc) == {"version", "source", "createdMs", "items"}
        assert doc["version"] == 1 and doc["source"] == "model_a:2026-11-18" and doc["createdMs"] == 1795000000000
        assert all(set(i) == {"kind", "id", "cropSha256", "names", "descriptors"} for i in doc["items"])
        by = {i["id"]: i for i in doc["items"]}
        assert sorted(by) == ["e1", "e2"], sorted(by)          # e8: name equals the phone's, no attribute
        assert by["e1"] == {"kind": "entry", "id": "e1", "cropSha256": sha(P["e1"]), "names": ["blue widget box"],
                            "descriptors": ["blue", "plastic"]}, by["e1"]
        assert by["e2"]["names"] == [] and by["e2"]["descriptors"] == ["small"]
        assert "2 items written (1 with a name, 2 with descriptors)" in text, text
        assert "2 equal to the phone's own name" in text and "1 usable entries with nothing to add" in text, text
    finally:
        shutil.rmtree(root)


def test_names_validation_and_binding():
    good = {"version": 1, "source": "model_a:2026-11-18", "createdMs": 1, "items": [
        {"kind": "entry", "id": "e1", "cropSha256": "0" * 64, "names": ["x"], "descriptors": []}]}
    assert ns.validate_names(good) == []
    for broken in ({**good, "version": 2}, {**good, "createdMs": "1"}, {**good, "source": ""},
                   {**good, "items": [dict(good["items"][0], cropSha256="ABC")]},
                   {**good, "items": [dict(good["items"][0], names="x")]},
                   {**good, "items": [dict(good["items"][0], kind="thing")]},
                   {**good, "items": good["items"] * 2}):
        assert ns.validate_names(broken), broken
    root = tempfile.mkdtemp(prefix="ns_bind_")
    try:
        P = {k: photo(k) for k in ("a", "b")}
        pull = write_pull(os.path.join(root, "pull"), [
            {"id": "a", "name": "widget", "labels": ["widget"], "last": 2, "photo": P["a"]},
            {"id": "b", "name": "gadget", "labels": ["gadget"], "last": 1, "photo": P["b"]}])
        scan = ns.Scan(pull)
        doc = {"version": 1, "source": "model_b:2026-11-18", "createdMs": 1, "items": [
            {"kind": "entry", "id": "a", "cropSha256": sha(P["a"]), "names": ["sprocket"], "descriptors": []},
            {"kind": "entry", "id": "b", "cropSha256": sha(b"another photo"), "names": ["gizmo"], "descriptors": []},
            {"kind": "entry", "id": "zz", "cropSha256": sha(P["a"]), "names": ["flange"], "descriptors": []},
            {"kind": "record", "id": "a", "cropSha256": sha(P["a"]), "names": ["cog"], "descriptors": []}]}
        bound, n = ns.bind(scan, doc)
        assert bound == {"a": (["sprocket"], [])}
        assert dict(n) == {"items": 4, "bound": 1, "photoDiffers": 1, "notInScan": 1, "notEntry": 1}, n
        items = scan.items(bound)
        assert [i.id for i, _ in ns.search(items, "gizmo", True)] == []      # unbound: nothing is guessed
        assert [i.id for i, _ in ns.search(items, "sprocket", True)] == ["a"]
        assert [i.id for i, _ in ns.search(items, "atrium", False)] == ["a", "b"]   # the room name
    finally:
        shutil.rmtree(root)


# ----------------------------------------------------------------------------- sheet

def test_sheet_shows_photos_and_codes_only():
    root = tempfile.mkdtemp(prefix="ns_sheet_")
    try:
        P = {k: photo("bytes-of-" + k) for k in ("e0000002", "e0000001", "e0000003")}
        pull = write_pull(os.path.join(root, "pull"), [
            {"id": k, "name": "zorbexname", "labels": ["quuxlelabel", "frobvote"], "last": 1, "photo": v,
             "container": "blorpbox"} for k, v in P.items()])
        out = os.path.join(root, "sheet.html")
        code, text = run_main(["sheet", pull, "--out", out])
        assert code == 0, text
        with open(out, encoding="utf-8") as f:
            h = f.read()
        import base64
        for k, v in P.items():
            assert "data:image/jpeg;base64," + base64.b64encode(v).decode() in h
            assert '"id": "%s"' % k in h
        for secret in ("zorbexname", "quuxlelabel", "frobvote", "blorpbox", "Atrium"):
            assert secret not in h, secret
        assert "http://" not in h and "https://" not in h and "<link" not in h and "src=\"//" not in h
        for c, k in (("A", "e0000001"), ("B", "e0000002"), ("C", "e0000003")):
            assert '{"code": "%s", "id": "%s"}' % (c, k) in h
            assert 'id="broad_%s"' % c in h and 'id="specific_%s"' % c in h
        assert all('id="none_%d"' % i in h for i in (1, 2, 3)) and 'id="none_4"' not in h
        assert '["query", "type", "target_code", "target_id"]' in h and 'a.download = "queries.csv"' in h
    finally:
        shutil.rmtree(root)


# ----------------------------------------------------------------------------- freeze and run

def run_scene(root):
    P = {k: photo(k) for k in ("e0000001", "e0000002", "e0000003")}
    pull = write_pull(os.path.join(root, "pull"), [
        {"id": "e0000001", "name": "widget", "labels": ["widget"], "room": "r1", "last": 300, "photo": P["e0000001"]},
        {"id": "e0000002", "name": "widget", "labels": ["widget", "gadget"], "room": "r1", "last": 200,
         "photo": P["e0000002"]},
        {"id": "e0000003", "name": "gadget", "labels": ["gadget"], "room": "r2", "last": 100, "photo": P["e0000003"]}])
    doc = {"version": 1, "source": "model_a:2026-11-18", "createdMs": 1, "items": [
        {"kind": "entry", "id": "e0000001", "cropSha256": sha(P["e0000001"]), "names": ["sprocket", "gadget case"],
         "descriptors": ["metal"]},
        {"kind": "entry", "id": "e0000002", "cropSha256": sha(P["e0000002"]), "names": ["gizmo holder"],
         "descriptors": ["blue"]},
        {"kind": "entry", "id": "e0000003", "cropSha256": sha(b"an older photo"), "names": ["flange"],
         "descriptors": []}]}
    names = os.path.join(root, "names_model_a.json")
    write(names, json.dumps(doc))
    queries = os.path.join(root, "queries.csv")
    rows = [("sprocket", "specific", "A", "e0000001"),   # 1 helped: only an extra name has it
            ("widget", "broad", "B", "e0000002"),        # 2 same: rank 2 in both
            ("gizmo", "specific", "B", "e0000002"),      # 3 helped
            ("gad", "broad", "C", "e0000003"),           # 4 hurt: A's extra name "gadget case" comes first
            ("xylophone", "none", "", ""),               # 5 nothing in either
            ("case", "none", "", ""),                    # 6 one result via extra: to review
            ("Blue", "broad", "B", "e0000002"),          # 7 helped, via a descriptor (class 2)
            ("flange", "specific", "C", "e0000003")]     # 8 same: not found in both (the extra name of C is not bound)
    with open(queries, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(ns.CSV_COLUMNS)
        wr.writerows(rows)
    return pull, names, queries


def test_run_refuses_without_a_matching_freeze_record():
    root = tempfile.mkdtemp(prefix="ns_freeze_")
    try:
        pull, names, queries = run_scene(root)
        out = os.path.join(root, "run")
        code, text = run_main(["run", pull, "--queries", queries, "--names", names, "--out", out])
        assert code == 3 and "no freeze record" in text and not os.path.exists(out), text
        code, text = run_main(["freeze", queries])
        assert code == 0, text
        with open(queries + ".frozen.json") as f:
            fz = json.load(f)
        with open(queries, "rb") as f:
            assert fz["sha256"] == sha(f.read())
        assert fz["rows"] == 8 and fz["byType"] == {"broad": 3, "specific": 3, "none": 2}
        assert fz["frozenAtUtc"].endswith("Z") and len(fz["scriptSha256"]) == 64
        code, text = run_main(["freeze", queries])
        assert code == 0 and "Already frozen" in text
        code, text = run_main(["run", pull, "--queries", queries, "--names", names, "--out", out])
        assert code == 0 and fz["sha256"] in text, text
        with open(queries, "a", newline="") as f:
            csv.writer(f).writerow(["gizmo", "broad", "B", "e0000002"])
        code, text = run_main(["run", pull, "--queries", queries, "--names", names, "--out", out + "2"])
        assert code == 3 and "changed after it was frozen" in text and not os.path.exists(out + "2"), text
        code, text = run_main(["freeze", queries])
        assert code == 3 and "Refusing" in text, text
    finally:
        shutil.rmtree(root)


def test_freeze_refuses_invalid_queries():
    root = tempfile.mkdtemp(prefix="ns_badq_")
    try:
        for i, (header, row, why) in enumerate([
                (ns.CSV_COLUMNS, ["cup", "maybe", "A", "e1"], "type"),
                (ns.CSV_COLUMNS, ["cup", "none", "A", "e1"], "a none query has a target"),
                (ns.CSV_COLUMNS, ["cup", "broad", "A", ""], "no target_id"),
                (ns.CSV_COLUMNS, ["?!", "broad", "A", "e1"], "no letter or digit"),
                (["query", "kind", "code", "id"], ["cup", "broad", "A", "e1"], "header")]):
            p = os.path.join(root, "q%d.csv" % i)
            with open(p, "w", newline="") as f:
                csv.writer(f).writerows([header, row])
            code, text = run_main(["freeze", p])
            assert code == 2 and why in text and not os.path.exists(p + ".frozen.json"), (why, text)
    finally:
        shutil.rmtree(root)


def test_run_on_a_synthetic_scan():
    root = tempfile.mkdtemp(prefix="ns_run_")
    try:
        pull, names, queries = run_scene(root)
        assert run_main(["freeze", queries])[0] == 0
        out = os.path.join(root, "run")
        code, text = run_main(["run", pull, "--queries", queries, "--names", names, "--names", names, "--out", out])
        assert code == 0, text
        with open(os.path.join(out, "run.json")) as f:
            res = json.load(f)
        keys = [c["key"] for c in res["conditions"]]
        assert keys == ["phone", "model_a:2026-11-18", "model_a:2026-11-18 #2"], keys
        assert res["conditions"][1]["binding"] == {"items": 3, "bound": 2, "photoDiffers": 1}, res["conditions"][1]
        K = "model_a:2026-11-18"
        per = {r["row"]: r["conditions"] for r in res["perQuery"]}
        assert per[1]["phone"]["rank"] is None and per[1]["phone"]["n"] == 0
        assert (per[1][K]["rank"], per[1][K]["class"], per[1][K]["targetViaExtra"]) == (1, 0, True)
        assert (per[2]["phone"]["rank"], per[2]["phone"]["n"], per[2][K]["rank"]) == (2, 2, 2)
        assert (per[4]["phone"]["rank"], per[4]["phone"]["class"]) == (1, 1)
        assert (per[4][K]["rank"], per[4][K]["n"], per[4][K]["top"]) == (2, 3, ["A", "C", "B"])
        assert [(x["code"], x["matched"]) for x in per[4][K]["toReview"]] == [("A", [["name", "gadget case"]])]
        assert per[5]["phone"]["n"] == 0 and per[5][K]["n"] == 0
        assert per[6]["phone"]["n"] == 0 and per[6][K]["n"] == 1 and per[6][K]["toReview"][0]["code"] == "A"
        assert (per[7][K]["rank"], per[7][K]["class"]) == (1, 2)
        assert per[8]["phone"]["rank"] is None and per[8][K]["rank"] is None and per[8][K]["viaExtra"] == 0
        s = res["summary"]
        assert s["phone"]["broad"] == {"n": 3, "found": 2, "top1": 1, "top3": 2, "medianRank": 1.5,
                                       "classes": {"0": 1, "1": 1}}, s["phone"]["broad"]
        assert (s["phone"]["specific"]["n"], s["phone"]["specific"]["found"]) == (3, 0)
        assert (s[K]["all"]["n"], s[K]["all"]["found"], s[K]["all"]["top1"], s[K]["all"]["top3"]) == (6, 5, 3, 5)
        assert s[K]["broad"]["medianRank"] == 2 and s[K]["specific"]["medianRank"] == 1
        assert s["phone"]["none"] == {"n": 2, "zeroResults": 2, "results": [0, 0]}
        assert s[K]["none"] == {"n": 2, "zeroResults": 1, "results": [0, 1]}
        assert s[K]["viaExtraResults"] == 5 and s[K]["toReview"] == 2 and s["phone"]["viaExtraResults"] == 0
        assert res["examples"][K] == {"helped": [1, 3, 7], "same": [2, 8], "hurt": [4]}, res["examples"][K]
        assert res["frozen"]["sha256"] in text
        with open(os.path.join(out, "run.md")) as f:
            md = f.read()
        assert res["frozen"]["sha256"] in md and "Hurt (the target's rank is worse): 1 of 6" in md
        assert "To review: results present only via extra names that are not the target (2)" in md
        assert "| model_a:2026-11-18 | broad | 3/3 | 1/3 | 3/3 | 2 (n = 3) |" in md, md
        # a target that is not an entry of the scan, and an invalid names file, are refused
        bad_names = os.path.join(root, "bad.json")
        write(bad_names, json.dumps({"version": 1, "source": "model_a:x", "createdMs": 1, "items": [{"kind": "entry"}]}))
        code, text = run_main(["run", pull, "--queries", queries, "--names", bad_names, "--out", out + "b"])
        assert code == 2 and "cropSha256" in text, text
        q2 = os.path.join(root, "q2.csv")
        with open(q2, "w", newline="") as f:
            csv.writer(f).writerows([ns.CSV_COLUMNS, ["widget", "broad", "A", "e9999999"]])
        assert run_main(["freeze", q2])[0] == 0
        code, text = run_main(["run", pull, "--queries", q2, "--names", names, "--out", out + "c"])
        assert code == 2 and "not an entry of this scan" in text, text
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    n_failed = n_run = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        n_run += 1
        try:
            fn()
            print("ok   ", name)
        except Exception as e:   # noqa: BLE001
            n_failed += 1
            import traceback
            print("FAIL ", name, type(e).__name__, str(e)[:3000])
            traceback.print_exc(limit=3)
    shutil.rmtree(_NO_MODELS, ignore_errors=True)
    print("%d tests, %d failed" % (n_run, n_failed))
    sys.exit(1 if n_failed else 0)
