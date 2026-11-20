#!/usr/bin/env python3
"""Tests of analysis/name_retrieval.py. Synthetic items and a fake text-image encoder only: no phone data, no
image, no model weights.

    analysis/.venv-bench/bin/python analysis/test_name_retrieval.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Aliases only: point the model loader at a file that does not exist, so the private one is never read.
_NO_MODELS = tempfile.mkdtemp(prefix="nr_nomodels_")
os.environ["FB_VLM_MODELS"] = os.path.join(_NO_MODELS, "missing.json")
import name_retrieval as nr  # noqa: E402

Item = nr.ns.Item


def scene():
    """Four items: two named widget, a sprocket that a naming model calls a gadget case, an unnamed lumen rod.
    Invented words only."""
    return [Item(id="a", name="widget", labels=["widget"], room="Atrium", last_seen_ms=30),
            Item(id="b", name="sprocket", labels=["sprocket"], last_seen_ms=20, extra_names=["gadget case"],
                 descriptors=["blue"]),
            Item(id="c", name="widget", labels=["widget"], last_seen_ms=10),
            Item(id="d", name="", labels=[], last_seen_ms=5, extra_names=["lumen rod"])]


class FakeEncoder:
    """Vectors by table: a photo "path" names its vector; a text's vector is looked up by the text."""
    AXES = {"widget": 0, "gadget": 1, "lumen": 2}

    def __init__(self):
        self.texts = []

    @staticmethod
    def unit(*xs):
        v = np.asarray(xs, float)
        return v / np.linalg.norm(v)

    def image(self, rgb):
        return rgb      # the fake loader below returns the vector itself

    def text(self, s):
        self.texts.append(s)
        v = np.zeros(3)
        for w, k in self.AXES.items():
            if w in s:
                v[k] = 1.0
        if not v.any():
            v[:] = 1.0          # a text without a known word points between the axes
        if s.startswith(nr.PROMPT):
            v = v + 0.01        # the prompt moves the text a little, never across items here
        return v / np.linalg.norm(v)


def test_rank_of_and_summary():
    assert nr.rank_of(["x", "y", "z"], {"z"}) == 3 and nr.rank_of(["x"], {"q"}) is None
    assert nr.rank_of(["x", "y", "z"], {"z", "y"}) == 2
    s = nr.summary([1, 2, None, 5, 11], shown=[3, 4, 0, 6, 12])
    assert s == {"n": 5, "result": 4, "found@1": 1, "found@3": 2, "found@5": 3, "found@10": 3, "medianShown": 4}, s
    t = nr.summary([1, 2, 10, 11], list_size=44)
    assert t == {"n": 4, "found@1": 1, "found@3": 2, "found@5": 2, "found@10": 3, "listSize": 44}, t
    assert nr.summary([], shown=[])["medianShown"] is None


def test_rule_cell_uses_the_app_rule():
    items = scene()
    # phone labels alone: "gadget" matches nothing; with extra names the sprocket is a class-1 result, only via extra
    assert nr.rule_cell(items, "gadget", False, "b") == {"shown": 0, "rank": None}
    assert nr.rule_cell(items, "gadget", True, "b") == {"shown": 1, "rank": 1, "viaExtra": True}
    # "widget": a and c (class 0), last seen descending; the lenient rank is the best of the accepted ids
    c = nr.rule_cell(items, "widget", False, "c", {"a", "c"})
    assert c == {"shown": 2, "rank": 2, "rankLenient": 1}, c
    # a descriptor ("blue", other pool) and the room ("atrium") also match; the target found without extra names is
    # not via extra
    assert nr.rule_cell(items, "blue", True, "b")["rank"] == 1
    assert nr.rule_cell(items, "atr", True, "a") == {"shown": 1, "rank": 1, "viaExtra": False}
    # same order as name_search.search
    assert [it.id for it, _ in nr.ns.search(items, "wid", False)] == ["a", "c"]


def test_embed_cell_ties_and_missing_photos():
    sims = {"a": 0.5, "b": 0.9, "c": 0.5, "d": None}
    assert nr.embed_order(sims) == ["b", "a", "c", "d"]
    c = nr.embed_cell(sims, "c", {"a", "c"})
    assert (c["rank"], c["tie"], c["rankLenient"], c["sim"]) == (3, True, 2, 0.5), c
    assert nr.embed_cell(sims, "b")["tie"] is False and nr.embed_cell(sims, "b")["rank"] == 1
    d = nr.embed_cell(sims, "d")
    assert (d["rank"], d["tie"]) == (4, False)      # an item without a photo is ranked last


def test_embed_all_and_t_cells_with_a_fake_encoder():
    enc = FakeEncoder()
    u = FakeEncoder.unit
    photos = {"a": u(1, 0, 0), "b": u(0.2, 1, 0), "c": u(0.9, 0.1, 0), "d": None}
    texts = [nr.text_of(q, v) for q in ("gadget", "widget") for v in nr.VARIANTS] + ["gadget"]
    iv, tv, cost = nr.embed_all(enc, photos, texts, load=lambda p: p)
    assert iv["d"] is None and cost["nImages"] == 3 and cost["nTexts"] == 4 and sorted(tv) == sorted(set(texts))
    assert len(enc.texts) == 4                      # each distinct text is embedded once
    assert cost["msPerImage"] is not None and cost["msPerText"] is not None
    rows = [{"target": "b", "query": "gadget", "accept": ["b"]}, {"target": "c", "query": "widget", "accept": ["a", "c"]}]
    nr.add_t(rows, "fake", iv, tv, ["a", "b", "c", "d"], accept_key="accept")
    assert set(rows[0]["T"]["fake"]) == set(nr.VARIANTS)
    assert rows[0]["T"]["fake"]["prompt"]["rank"] == 1 and rows[0]["T"]["fake"]["raw"]["rank"] == 1
    assert rows[1]["T"]["fake"]["raw"]["rank"] == 2 and rows[1]["T"]["fake"]["raw"]["rankLenient"] == 1
    assert rows[1]["T"]["fake"]["raw"]["sim"] == float(tv["widget"] @ iv["c"])
    assert nr.text_of("gadget", "prompt") == "a photo of gadget" and nr.text_of("gadget", "raw") == "gadget"


def test_cross_queries_and_eval2_pooling():
    items = scene()
    conds = {"L": [Item(**dict(vars(i), extra_names=[], descriptors=[])) for i in items]}
    conds["model_a"] = items
    conds["model_b"] = [Item(**dict(vars(i), extra_names=["widget"] if i.id == "b" else [], descriptors=[])) for i in items]
    first = {"model_a": {"b": "gadget case", "d": "lumen rod", "a": "gadget case"}, "model_b": {"b": "widget"}}
    q = nr.cross_queries(first)
    assert q == [("model_a", "a", "gadget case", ["a", "b"]), ("model_a", "b", "gadget case", ["a", "b"]),
                 ("model_a", "d", "lumen rod", ["d"]), ("model_b", "b", "widget", ["b"])], q
    rows = nr.eval2_rule(conds, first)
    assert [sorted(r["N"]) for r in rows] == [["model_b"]] * 3 + [["model_a"]]
    # model_a's "gadget case" against model_b's index: no item is a result
    assert rows[1]["N"]["model_b"] == {"shown": 0, "rank": None, "rankLenient": None, "viaExtra": False}
    # model_b's "widget" against model_a's index: a and c by the phone name; b is not a result (its extra name is a gadget case)
    assert rows[3]["N"]["model_a"]["rank"] is None and rows[3]["N"]["model_a"]["shown"] == 2
    # "lumen rod" against model_b's index: d has no phone name or label, so no result
    assert rows[2]["L"]["shown"] == 0 and rows[2]["N"]["model_b"]["shown"] == 0
    enc = FakeEncoder()
    u = FakeEncoder.unit
    photos = {"a": u(1, 0, 0), "b": u(0.2, 1, 0), "c": u(0.9, 0.1, 0), "d": u(0, 0, 1)}
    iv, tv, _ = nr.embed_all(enc, photos, [nr.text_of(r["query"], v) for r in rows for v in nr.VARIANTS], load=lambda p: p)
    nr.add_t(rows, "fake", iv, tv, sorted(photos), accept_key="accept")
    s = nr.eval2_summary(rows, ["fake"], 4)
    assert list(s) == ["model_a", "model_b", "all"]
    assert s["model_a"]["nPairs"] == 1 and s["model_b"]["nPairs"] == 3 and s["all"]["nPairs"] == 4
    assert s["model_b"]["strict"]["T:fake:prompt"]["n"] == 3 and s["model_b"]["strict"]["T:fake:prompt"]["listSize"] == 4
    # the gadget photo b ranks first for "gadget case": strict rank 1 for target b, 2 for target a; lenient 1 for both
    assert s["model_b"]["strict"]["T:fake:prompt"]["found@1"] == 2      # b ("gadget case") and d ("lumen rod")
    assert s["model_b"]["lenient"]["T:fake:prompt"]["found@1"] == 3
    assert s["model_b"]["sharedName"] == 2 and s["all"]["sharedName"] == 2
    assert s["all"]["strict"]["L"]["n"] == 4 and s["all"]["strict"]["N"]["n"] == 4


def test_eval1_check_and_summary():
    cell = {"shown": 2, "rank": 1, "viaExtra": True}
    miss = {"shown": 0, "rank": None, "viaExtra": False}
    rows = [dict({"target": "r1", "query": "q", "L": {"shown": 0, "rank": None}},
                 **{m: (cell if m == "model_c" else miss) for m in nr.MODELS})]
    ok, got = nr.check_eval1(rows)
    assert not ok and got == {"n": 1, "L": 0, "model_a": 0, "model_b": 0, "model_c": 1, "model_d": 0}
    assert nr.check_eval1(rows, expected=got)[0]
    rows[0]["T"] = {"fake": {v: {"rank": 7, "tie": False} for v in nr.VARIANTS}}
    s = nr.eval1_summary(rows, ["fake"], 44)
    assert s["model_c"]["found@1"] == 1 and s["T:fake:raw"] == {"n": 1, "found@1": 0, "found@3": 0, "found@5": 0,
                                                                "found@10": 1, "listSize": 44}


def test_report_has_numbers_only():
    secret = "zebrawood teapot"
    items = scene() + [Item(id="rec:r1", name="", labels=[""], last_seen_ms=1)]
    conds = {"L": items}
    conds.update({m: items for m in nr.MODELS})
    rows1 = nr.eval1_rule(conds, [("rec:r1", secret)])
    first = {m: {"a": secret + " " + m} for m in nr.MODELS}
    rows2 = nr.eval2_rule(conds, first)
    enc = FakeEncoder()
    photos = {i.id: FakeEncoder.unit(1, 0, 0) for i in items}
    iv, tv, cost = nr.embed_all(enc, photos, [nr.text_of(r["query"], v) for r in rows1 + rows2 for v in nr.VARIANTS],
                                load=lambda p: p)
    ids = sorted(photos)
    nr.add_t(rows1, "fake", iv, tv, ids)
    nr.add_t(rows2, "fake", iv, tv, ids, accept_key="accept")
    res = {"generatedUtc": "2026-01-01T00:00:00Z", "gitCommit": "0" * 40, "tTies": 0,
           "environment": {"cpu": "test", "versions": {"python": "3"}},
           "encoders": {"fake": {"label": "Fake", "arch": "fake", "pretrained": "none", "repo": "none", "revision": "r" * 40,
                                 "fileBytes": 1e6, "paramsImage": 1e6, "paramsOther": 2e6, "inputSize": 8, "cost": cost,
                                 "loadS": 0.1}},
           "eval1": {"provenance": {"pull": "p1", "nScanEntries": 4, "nRecords": 1}, "nItems": 5, "noPhoto": 0,
                     "check": nr.check_eval1(rows1)[1], "reproduced": False, "summary": nr.eval1_summary(rows1, ["fake"], 5),
                     "viaExtra": {m: 0 for m in nr.MODELS}},
           "eval2": {"provenance": {"pull": "p2", "nScanEntries": 5, "withName": {m: 1 for m in nr.MODELS},
                                    "binding": {m: {"bound": 1, "items": 1} for m in nr.MODELS}},
                     "noPhoto": 0, "summary": nr.eval2_summary(rows2, ["fake"], 5)}}
    md = nr.report_md(res)
    assert "zebrawood" not in md and "teapot" not in md
    assert "| L (phone labels alone) | 1 | 0/1 |" in md, md
    assert "| all | N(Y) | 12 |" in md and "| model_a | T(Fake) | 3 |" in md, md
    assert "NOT reproduced" in md and "raw-query variant" in md


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
