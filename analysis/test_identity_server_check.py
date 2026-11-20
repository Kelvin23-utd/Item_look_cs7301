#!/usr/bin/env python3
"""Tests of analysis/identity_server_check.py on small synthetic cases. No phone data, no image, no
model weights.

    python3 analysis/test_identity_server_check.py
    analysis/.venv-bench/bin/python analysis/test_identity_server_check.py

Covers the ranking and margin rule (ties against the right answer, several right groups, groups an
encoder cannot score), the one-view and several-view scores, percentiles, the clear-negative pairs,
candidates from earlier scans only, the crop costs, and the whole command (phone embedding only) on a
small pull written to a temporary folder with placeholder view files.
"""
import base64
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import identity_server_check as isc  # noqa: E402


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def vec(angle_deg, dim=4):
    """Unit vector in the first two axes at this angle: cosine between two of them = cos(difference)."""
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


# ----------------------------------------------------------------------------- ranking and margin

def test_rank_right_on_top():
    r = isc.rank_right({"A": 0.8, "B": 0.6, "C": 0.5}, {"A"})
    assert r["rank"] == 1 and r["candidates"] == 3 and r["unscored"] == 0
    assert r["right"] == "A" and close(r["score"], 0.8)
    assert r["other"] == "B" and close(r["other_score"], 0.6) and close(r["margin"], 0.2)


def test_rank_right_not_on_top_gives_negative_margin():
    r = isc.rank_right({"A": 0.4, "B": 0.6, "C": 0.5, "D": 0.1}, {"A"})
    assert r["rank"] == 3
    assert r["other"] == "B" and close(r["margin"], -0.2)


def test_ties_count_against_the_right_answer():
    r = isc.rank_right({"A": 0.5, "B": 0.5, "C": 0.2}, {"A"})
    assert r["rank"] == 2 and close(r["margin"], 0.0)
    r = isc.rank_right({"A": 0.5, "B": 0.5, "C": 0.5}, {"A"})
    assert r["rank"] == 3


def test_several_right_groups_the_best_counts_and_none_is_a_wrong_candidate():
    scores = {"R1": 0.7, "R2": 0.9, "W1": 0.8, "W2": 0.3}
    r = isc.rank_right(scores, {"R1", "R2"})
    assert r["rank"] == 1 and r["right"] == "R2" and close(r["score"], 0.9)
    assert r["other"] == "W1" and close(r["margin"], 0.1)
    # each right group's own rank counts only the wrong ones above it
    assert r["right_ranks"] == {"R2": 1, "R1": 2}


def test_unscorable_groups_are_left_out_and_an_unscorable_right_group_has_no_rank():
    r = isc.rank_right({"A": 0.5, "B": None, "C": 0.6}, {"A"})
    assert r["candidates"] == 2 and r["unscored"] == 1 and r["rank"] == 2
    r = isc.rank_right({"A": None, "B": 0.6}, {"A"})
    assert r["rank"] is None and r["score"] is None and r["margin"] is None and r["other"] == "B"


def test_no_wrong_candidate_means_no_margin():
    r = isc.rank_right({"A": 0.5}, {"A"})
    assert r["rank"] == 1 and r["other"] is None and r["margin"] is None


def test_wrong_candidates_tied_on_score_break_by_name():
    r = isc.rank_right({"A": 0.9, "C": 0.5, "B": 0.5}, {"A"})
    assert r["other"] == "B"


# ----------------------------------------------------------------------------- scores

def test_one_view_uses_the_best_quality_view_of_each_side():
    v = {"q1": vec(0), "q2": vec(90), "c1": vec(10), "c2": vec(80)}
    sim = isc.Sim(v)
    # best query view is q2 (quality 5), best candidate view c1 (quality 9): cos(80 deg)
    s = isc.score_pair([("q1", 1.0), ("q2", 5.0)], [("c1", 9.0), ("c2", 2.0)], sim)
    assert close(s["one"], math.cos(math.radians(80)))
    pair = sorted([math.cos(math.radians(d)) for d in (10, 80, 80, 10)])
    assert close(s["max"], max(pair)) and close(s["median"], (pair[1] + pair[2]) / 2) and s["pairs"] == 4


def test_equal_quality_picks_the_first_view():
    assert isc.best_item([("a", 2.0), ("b", 2.0), ("c", 1.0)]) == "a"
    assert isc.best_item([]) is None


def test_an_empty_side_gives_no_score():
    sim = isc.Sim({"a": vec(0)})
    assert isc.score_pair([], [("a", 1.0)], sim) is None
    assert isc.score_pair([("a", 1.0)], [], sim) is None


def test_sim_is_memoised_and_symmetric():
    sim = isc.Sim({"a": vec(0), "b": vec(60)})
    assert close(sim("a", "b"), 0.5) and close(sim("b", "a"), 0.5) and len(sim.memo) == 1


def test_percentile_and_share():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert close(isc.percentile(xs, 0), 1.0) and close(isc.percentile(xs, 100), 5.0)
    assert close(isc.percentile(xs, 50), 3.0) and close(isc.percentile(xs, 95), 4.8)
    assert close(isc.percentile([3.0, 1.0], 25), 1.5)
    assert isc.percentile([], 50) is None
    try:
        import numpy as np
        ys = [0.3, -0.2, 0.71, 0.05, 0.9, 0.44, 0.12]
        for q in (5, 50, 95, 99):
            assert close(isc.percentile(ys, q), float(np.percentile(ys, q)), 1e-12)
    except ImportError:
        pass
    assert close(isc.share_at_or_above([0.1, 0.5, 0.5, 0.9], 0.5), 0.75)
    assert isc.share_at_or_above([], 0.5) is None and isc.share_at_or_above([0.1], None) is None
    s = isc.summary([0.2, 0.4])
    assert s["n"] == 2 and close(s["median"], 0.3) and close(s["max"], 0.4)


# ----------------------------------------------------------------------------- groups and pairs

def ob(oid, scan, entry, x, emb, view=True, q=1.0, label="cup", nbytes=100):
    return isc.Ob(oid, scan, entry, label, q, (x, 0.0, 0.0), emb, "/v/%s.jpg" % oid if view else None,
                  nbytes if view else 0)


def test_side_phone_all_uses_every_observation_the_others_only_views():
    obs = [ob("a", "s", "e", 0, vec(0), view=True), ob("b", "s", "e", 0, vec(0), view=False)]
    assert [k for k, _ in isc.side(obs, "phone_all")] == ["a", "b"]
    assert [k for k, _ in isc.side(obs, "phone")] == ["a"]
    assert [k for k, _ in isc.side(obs, "dinov2s")] == ["a"]


def test_far_pairs_need_every_observation_far_apart_within_one_scan():
    g = isc.make_groups({"s1": [ob("1", "s1", "A", 0.0, vec(0)), ob("2", "s1", "B", 1.5, vec(0)),
                                ob("3", "s1", "C", 0.5, vec(0)), ob("4", "s1", "C", 2.2, vec(0))],
                         "s2": [ob("5", "s2", "D", 5.0, vec(0))]})
    pairs = isc.far_pairs(g.values(), 1.0)
    # A-B 1.5 m: far. A-C: one C observation 0.5 m from A: not far. B-C: one C observation 0.7 m from
    # B: not far. No pair across scans.
    assert pairs == [("s1:A", "s1:B")], pairs
    assert isc.far_pairs(g.values(), 1.0, exclude={"s1:A"}) == []


def test_groups_split_by_scan_and_entry_and_carry_tentative_flag():
    g = isc.make_groups({"s1": [ob("1", "s1", "A", 0, vec(0)), ob("2", "s1", "A", 0, vec(0), label="bowl"),
                                ob("3", "s1", "A", 0, vec(0), label="bowl")],
                         "s2": [ob("4", "s2", "A", 0, vec(0), view=False)]}, {"s2": {"A"}})
    assert list(g) == ["s1:A", "s2:A"]
    assert g["s1:A"].label == "bowl" and len(g["s1:A"].viewed) == 3 and not g["s1:A"].tentative
    assert g["s2:A"].viewed == [] and g["s2:A"].tentative


def scene():
    """Three scans; entry X moves between s2 and s3. Phone embedding: X's views at 50-60 deg from the
    query; a strong encoder: 10-20 deg. A look-alike Y at 40 deg (phone) and 30 deg (strong)."""
    phone = {"h1": vec(55), "h2": vec(60), "y1": vec(40), "z1": vec(120), "w1": vec(150), "q1": vec(0), "q2": vec(5),
             "old": vec(170)}
    strong = {"h1": vec(10), "h2": vec(20), "y1": vec(30), "z1": vec(120), "w1": vec(150), "q1": vec(0), "q2": vec(2),
              "old": vec(170)}
    obs = {"s1": [ob("old", "s1", "X", 0.0, phone["old"])],
           "s2": [ob("h1", "s2", "X", 0.0, phone["h1"], q=5.0), ob("h2", "s2", "X", 0.1, phone["h2"], q=1.0),
                  ob("y1", "s2", "Y", 3.0, phone["y1"]), ob("z1", "s2", "Z", 0.5, phone["z1"]),
                  ob("w1", "s2", "W", 6.0, phone["w1"], view=False)],
           "s3": [ob("q1", "s3", "N", 2.0, phone["q1"], q=3.0), ob("q2", "s3", "N", 2.1, phone["q2"], q=1.0)]}
    groups = isc.make_groups(obs)
    by_id = {o.id: o for v in obs.values() for o in v}
    return groups, by_id, ["s1", "s2", "s3"], phone, strong


def test_candidates_come_from_earlier_scans_only_and_a_right_group_must_be_one():
    groups, by_id, order, _, _ = scene()
    q, cands, right = isc.resolve_case({"name": "c", "test": "s3", "query_entry": "N", "right": [["s2", "X"]]},
                                       groups, by_id, order)
    assert [o.id for o in q] == ["q1", "q2"] and right == {"s2:X"}
    assert sorted(g.key for g in cands) == ["s1:X", "s2:W", "s2:X", "s2:Y", "s2:Z"]
    q, cands, _ = isc.resolve_case({"name": "c", "test": "s2", "query_obs": ["y1"], "right": [["s1", "X"]]},
                                   groups, by_id, order)
    assert [g.key for g in cands] == ["s1:X"] and [o.id for o in q] == ["y1"]
    try:
        isc.resolve_case({"name": "c", "test": "s2", "query_obs": ["y1"], "right": [["s2", "X"]]}, groups, by_id, order)
        raise AssertionError("a right group of the test scan is not a candidate")
    except ValueError:
        pass


def test_costs_count_the_crops_of_both_sides():
    groups, by_id, order, _, _ = scene()
    q, cands, _ = isc.resolve_case({"name": "c", "test": "s3", "query_entry": "N", "right": [["s2", "X"]]},
                                   groups, by_id, order)
    c = isc.case_costs(q, cands)
    assert c["query_one"] == {"n": 1, "bytes": 100, "ids": ["q1"]}
    assert c["query_all"]["n"] == 2 and c["query_all"]["bytes"] == 200
    # groups with a view: s1:X, s2:X, s2:Y, s2:Z (s2:W has none); s2:X's best view is h1
    assert c["cand_one"]["n"] == 4 and "h1" in c["cand_one"]["ids"] and "h2" not in c["cand_one"]["ids"]
    assert c["cand_all"]["n"] == 5 and c["cand_all"]["bytes"] == 500


def test_evaluate_ranks_margins_and_negatives():
    groups, by_id, order, phone, strong = scene()
    cases = [{"name": "moved", "kind": "moved", "test": "s3", "query_entry": "N", "right": [["s2", "X"]]}]
    sims = {"phone_all": isc.Sim(phone), "phone": isc.Sim(phone), "strong": isc.Sim(strong)}
    res, neg, info = isc.evaluate(cases, groups, by_id, order, sims)
    r = res[0]
    assert r["candidates"] == 5 and r["candidates_with_view"] == 4 and r["query_views"] == ["q1", "q2"]
    p = r["by_encoder"]["phone"]
    # phone: X at best 50 deg (q2-h1), Y at 35 deg (q2-y1): Y wins, X second
    assert p["max"]["rank"] == 2 and p["max"]["other"] == "s2:Y"
    assert close(p["max"]["score"], math.cos(math.radians(50))) and close(p["max"]["other_score"], math.cos(math.radians(35)))
    # one view: q1 (quality 3) against h1 (quality 5): 55 deg; against Y: 40 deg
    assert close(p["one"]["score"], math.cos(math.radians(55))) and p["one"]["rank"] == 2
    s = r["by_encoder"]["strong"]
    assert s["max"]["rank"] == 1 and close(s["max"]["score"], math.cos(math.radians(8)))
    assert close(s["max"]["margin"], math.cos(math.radians(8)) - math.cos(math.radians(28)))
    # X's four pairs lie at 10, 20, 8 and 18 deg: the median is the mean of the cosines of 18 and 10 deg
    cs = sorted(math.cos(math.radians(a)) for a in (10, 20, 8, 18))
    assert close(s["median"]["score"], (cs[1] + cs[2]) / 2)
    # phone_all scores s2:W (no view); the view-based encoders cannot
    assert r["by_encoder"]["phone_all"]["max"]["candidates"] == 5 and s["max"]["candidates"] == 4
    # negatives: within s2, groups more than 1 m apart: X-Y (2.9 m), X-W, Y-W, Z-Y, Z-W; X-Z is 0.4 m;
    # the moved query group s3:N is left out
    assert info["pairs"] == 5 and info["excluded_groups"] == ["s3:N"], info
    assert len(neg["phone_all"]["max"]) == 5 and len(neg["phone"]["max"]) == 2   # W has no view
    assert all(v is not None for v in neg["strong"]["one"])
    # the share of negatives at or above the right score
    assert close(s["max"]["share_negatives_at_or_above"], 0.0)


# ----------------------------------------------------------------------------- the command on a pull

def b64(v):
    return base64.b64encode(struct.pack("<%df" % len(v), *v)).decode()


def write_pull(folder):
    """Two scans of one map; entry E1 in scan A, its object seen in scan B as new entry E9."""
    t = {"sa": 1_000_000, "sb": 2_000_000}
    events = []
    for s, t0 in t.items():
        events.append({"type": "scan_start", "scan": s, "t": t0, "session": "ss" + s, "mapId": "m1", "studyLogging": True})
        events.append({"type": "scan_end", "scan": s, "t": t0 + 50_000, "session": "ss" + s})
    with open(os.path.join(folder, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    open(os.path.join(folder, "inventory_log.jsonl"), "w").close()
    specs = {
        "sa": [("sa-0-0", "E1", (0.0, 0.0, 0.0), vec(30, 8), True, 900, 4000.0),
               ("sa-1-0", "E1", (0.05, 0.0, 0.0), vec(35, 8), False, 0, 6000.0),
               ("sa-1-1", "E2", (2.0, 0.0, 0.0), vec(70, 8), True, 700, 3000.0),
               ("sa-2-0", "E3", (4.0, 0.0, 0.0), vec(160, 8), True, 500, 2000.0)],
        "sb": [("sb-0-0", "E9", (1.0, 0.0, 1.0), vec(0, 8), True, 1200, 5000.0),
               ("sb-1-0", "E9", (1.0, 0.0, 1.05), vec(5, 8), True, 1100, 1000.0)],
    }
    for s, rows in specs.items():
        os.makedirs(os.path.join(folder, "obs"), exist_ok=True)
        os.makedirs(os.path.join(folder, "views", s), exist_ok=True)
        retained = []
        with open(os.path.join(folder, "obs", s + ".jsonl"), "w") as f:
            for oid, entry, p, e, has_view, nbytes, q in rows:
                kf, det = int(oid.split("-")[1]), int(oid.split("-")[2])
                view = None
                if has_view:
                    view = "views/%s/%s.jpg" % (s, oid)
                    with open(os.path.join(folder, view), "wb") as vf:
                        vf.write(b"\0" * nbytes)          # a placeholder, never decoded here
                    retained.append(oid)
                f.write(json.dumps({"v": 1, "type": "obs", "obs": oid, "kind": "scan", "scan": s, "kf": kf, "det": det,
                                    "t": t[s] + 500 * kf, "label": "cup", "score": 0.6, "quality": q,
                                    "box": [0, 0, 10, 10], "mapPoint": list(p), "sizeM": 0.1, "emb": b64(e), "embDim": 8,
                                    "embModel": "test", "viewGroup": 0, "view": view,
                                    "viewReason": "kept" if has_view else "person",
                                    "decision": {"matcher": "baseline", "kind": "new", "entry": entry}}) + "\n")
            f.write(json.dumps({"v": 1, "type": "scan_end", "scan": s, "t": t[s] + 50_000, "retained": retained,
                                "created": [], "tentativeDropped": []}) + "\n")


def test_command_on_a_written_pull_phone_only():
    folder = tempfile.mkdtemp(prefix="isc_test_")
    try:
        write_pull(folder)
        cases_path = os.path.join(folder, "cases.json")
        with open(cases_path, "w") as f:
            json.dump([{"name": "moved", "kind": "moved", "test": "sb", "query_entry": "E9", "right": [["sa", "E1"]]}], f)
        out_dir = os.path.join(folder, "out")
        buf = io.StringIO()
        code = isc.main([folder, "--encoders", "none", "--cases", cases_path, "--out", out_dir], out=buf)
        assert code == 0, buf.getvalue()
        for n in ("report.txt", "tables.md", "result.json"):
            assert os.path.getsize(os.path.join(out_dir, n)) > 0
        with open(os.path.join(out_dir, "result.json")) as f:
            res = json.load(f)
        assert res["scans"] == ["sa", "sb"] and res["info"]["observations"] == 6 and res["info"]["with view"] == 5
        c = res["cases"][0]
        assert c["candidates"] == 3 and c["right_sizes"] == {"sa:E1": [2, 1]}
        pa = c["by_encoder"]["phone_all"]["max"]
        # phone_all: every observation; best pair sb-1-0 (5 deg) with sa-0-0 (30 deg): 25 deg
        assert pa["rank"] == 1 and close(pa["score"], math.cos(math.radians(25)), 1e-6)
        assert close(pa["other_score"], math.cos(math.radians(65)), 1e-6) and pa["other"] == "sa:E2"
        # one view over every observation: sb-0-0 (quality 5000) against sa-1-0 (6000, no view): 35 deg
        assert close(c["by_encoder"]["phone_all"]["one"]["score"], math.cos(math.radians(35)), 1e-6)
        pv = c["by_encoder"]["phone"]
        # views only: E1 has one view (sa-0-0, 30 deg from sb-0-0, 25 deg from sb-1-0)
        assert close(pv["max"]["score"], math.cos(math.radians(25)), 1e-6)
        assert close(pv["one"]["score"], math.cos(math.radians(30)), 1e-6)    # best views: sb-0-0 and sa-0-0
        assert c["costs"]["query_one"]["bytes"] == 1200 and c["costs"]["query_all"]["bytes"] == 2300
        assert c["costs"]["cand_one"]["bytes"] == 900 + 700 + 500 and c["costs"]["cand_all"]["n"] == 3
        # negatives: scan sa, E1-E2 (1.95 m), E1-E3, E2-E3; the query group sb:E9 left out
        assert res["negatives"]["info"]["pairs"] == 3 and res["negatives"]["info"]["excluded_groups"] == ["sb:E9"]
        assert c["sightings"] and sum(s["query_obs"] for s in c["sightings"]) == 2
        assert res["skipped"] == {} and res["reproduction"] is None
    finally:
        shutil.rmtree(folder)


def test_audited_cases_are_well_formed():
    names = [c["name"] for c in isc.CASES]
    assert len(names) == len(set(names)) == 7
    assert sum(1 for c in isc.CASES if c["kind"] == "moved") == 3
    for c in isc.CASES:
        assert bool(c.get("query_obs")) != bool(c.get("query_entry"))
        assert c["right"] and all(len(x) == 2 for x in c["right"])
        assert all(s != c["test"] for s, _ in c["right"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print("ok  %s" % n)
    print("%d tests passed" % len(tests))
