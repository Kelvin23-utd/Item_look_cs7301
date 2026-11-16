#!/usr/bin/env python3
"""Tests for the analysis code.

Run:  python3 analysis/test_analysis.py            (prints one line per test, exit code 1 on failure)
      python3 analysis/test_analysis.py -k boot    (only tests whose name contains "boot")
      pytest analysis/test_analysis.py             (if pytest is installed)

The core is a hand-built log (HB below): eleven paired queries whose ground truth, measured
error, capture similarity and logged policy decision are chosen so that the count of every
outcome class under every policy can be worked out by hand. The expected counts are written
next to the table, not computed by the code under test.

Nothing here uses or produces a result: all inputs are hand-built or synthetic and live in
temporary directories.
"""
import base64
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import fb_load as L  # noqa: E402
import fb_metrics as M  # noqa: E402
import make_trial_sheet as TS  # noqa: E402
import paper_figures as PF  # noqa: E402
import power_measure as PM  # noqa: E402

TAG3 = "home=H;exp=E3;round=1"

# ----------------------------------------------------------------------------- the hand-built log
# On-device threshold 0.75. "at marker" = present with error <= 0.30 m, or present unmeasured.
#  query session truth       measured sim   policyVerify calls   at marker  capture confirms
HB = [
    ("A", "s1", "present",   0.10,    0.90, False,       3),   # yes        yes
    ("B", "s1", "present",   None,    0.60, True,        3),   # yes (unmeasured)  no
    ("C", "s1", "present",   0.30,    0.75, True,        5),   # yes (boundary)    yes (boundary)
    ("D", "s1", "present",   0.31,    0.80, False,       3),   # NO: present but 31 cm off    yes
    ("E", "s2", "present",   0.50,    0.20, True,        3),   # NO         no
    ("F", "s2", "absent",    None,    0.10, True,        5),   # no         no
    ("G", "s2", "absent",    None,    0.80, False,       3),   # no         yes (a false confirmation)
    ("H", "s2", "displaced", 0.80,    0.30, True,        3),   # no         no
    ("I", "s3", "displaced", 0.20,    0.76, False,       3),   # no (displaced is never at the marker)  yes
    ("J", "s3", "replaced",  None,    0.74, True,        5),   # no         no (just under)
    ("K", "s3", "replaced",  None,    0.95, True,        3),   # no         yes
]
# At the marker: A B C (3). Stale: D E F G H I J K (8).
# Capture confirms: A C D G I K. Does not: B E F H J.
HB_EXPECT = {
    # Memory only: no capture, always a lead. 3 correct leads, 8 wrong leads.
    "Memory": {"captures": 0, "correct_lead": 3, "wrong_lead": 8, "correct_warning": 0, "lost_retrieval": 0, "calls": 0},
    # Always verify: leads where the capture confirms. A C correct; D G I K wrong; B lost; E F H J warned.
    "Always": {"captures": 11, "correct_lead": 2, "wrong_lead": 4, "correct_warning": 4, "lost_retrieval": 1, "calls": 39},
    # Logged heuristic captures B C E F H J K. Not captured (lead): A correct, D G I wrong.
    # Captured: C confirmed correct, K confirmed wrong, B lost, E F H J warned.
    "Heuristic": {"captures": 7, "correct_lead": 2, "wrong_lead": 4, "correct_warning": 4, "lost_retrieval": 1, "calls": 27},
    # Oracle captures exactly the stale D..K. A B C lead (correct). D G I K confirm (wrong), E F H J warned.
    "Oracle": {"captures": 8, "correct_lead": 3, "wrong_lead": 4, "correct_warning": 4, "lost_retrieval": 0, "calls": 28},
}


def event(session, t, etype, tag, mode="PAIRED", **fields):
    e = {"t": t, "session": session, "type": etype}
    if etype in ("session_start", "register", "query_start", "verify", "query_end"):
        e["studyTag"] = tag
        e["captureMode"] = mode
    e.update(fields)
    return e


def query_events(clock, qid, session, truth, measured, sim, policy_verify, calls, tag, mode="PAIRED", source="MAP",
                 no_capture=None, end=True, condition_label=None, extra_start=None, extra_verify=None, threshold=0.75,
                 extra_end=None):
    """query_start (+ verify) (+ query_end) for one query, in v0.4 shape."""
    out = []
    start = dict(query="q" + qid, record="r" + qid, name="item " + qid, ageMin=50.0, maxSimOthers=0.1,
                 positionSource=source, policyVerify=policy_verify, policyReasons=[], metersSinceFix=5.0,
                 trackingLossesSinceFix=0, msSinceFix=60000, fixType="SPOT", fixPoint="f1", mobility="some", moveCount=0)
    start.update(extra_start or {})
    out.append(event(session, next(clock), "query_start", tag, mode, **start))
    if sim is not None:
        ver = dict(query="q" + qid, record="r" + qid, attempt=1, positionSource=source, bestSimilarity=sim,
                   bestKind="projected", scores=[{"kind": "projected", "sim": sim, "rect": [0, 0, 1, 1]}],
                   threshold=threshold, outcome="CONFIRMED" if sim >= threshold else "INSUFFICIENT_EVIDENCE",
                   inferenceCalls=calls, latency={"convertMs": 50.0, "detectMs": 80.0, "embedMs": 30.0, "totalMs": 170.0},
                   msSinceNear=4000, msSinceQueryStart=9000)
        ver.update(extra_verify or {})
        out.append(event(session, next(clock), "verify", tag, mode, **ver))
        if condition_label:
            out.append(event(session, next(clock), "verify_label", tag, mode, query="q" + qid, record="r" + qid, attempt=1,
                             label=condition_label[0], condition=condition_label[1]))
    if end:
        marker = None
        if measured is not None:
            marker = {"errorM": measured, "horizontalErrorM": measured, "verticalErrorM": 0.0, "positionSource": source,
                      "displaced": truth == "displaced", "metersSinceFix": start["metersSinceFix"], "fixPoint": "f1"}
        out.append(event(session, next(clock), "query_end", tag, mode, query="q" + qid, record="r" + qid, groundTruth=truth,
                         markerError=marker, noCaptureReason=no_capture, verifications=1 if sim is not None else 0,
                         **(extra_end or {})))
    return out


def counter(start=1000):
    t = start
    while True:
        t += 1000
        yield t


def write_folder(path, events, records=None, records_log=None, map_json=None, files=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    if records is not None:
        with open(os.path.join(path, "records.json"), "w") as f:
            json.dump(records, f)
    if records_log is not None:
        with open(os.path.join(path, "records_log.jsonl"), "w") as f:
            for e in records_log:
                f.write(json.dumps(e) + "\n")
    if map_json is not None:
        with open(os.path.join(path, "map.json"), "w") as f:
            json.dump(map_json, f)
    for rel, doc in (files or {}).items():
        os.makedirs(os.path.dirname(os.path.join(path, rel)), exist_ok=True)
        with open(os.path.join(path, rel), "w") as f:
            json.dump(doc, f)
    return path


def handbuilt_e3(path):
    clock = counter()
    ev = []
    for s in ("s1", "s2", "s3", "fr"):
        ev.append(event(s, next(clock), "session_start", TAG3, "FREE" if s == "fr" else "PAIRED", records=0))
    ev.append(event("u0", next(clock), "session_start", "", "PAIRED", records=0))
    for q, session, truth, measured, sim, pv, calls in HB:
        ev += query_events(clock, q, session, truth, measured, sim, pv, calls, TAG3)
    # Not replayed, each for a different reason:
    ev += query_events(clock, "L", "s3", "skip", None, 0.90, True, 3, TAG3)                                    # skip
    ev += query_events(clock, "M", "s3", "present", 0.05, None, False, 0, TAG3, no_capture="user_skipped")     # no capture
    ev += query_events(clock, "N", "s3", "absent", None, None, True, 0, TAG3, source="NONE", no_capture="not_located")
    ev += query_events(clock, "O", "s3", None, None, None, True, 0, TAG3, end=False)                           # unfinished
    ev += query_events(clock, "P", "u0", "present", 0.01, 0.99, False, 3, "")                                  # no study tag
    ev += query_events(clock, "Q", "fr", "absent", None, 0.99, False, 3, TAG3, mode="FREE")                    # FREE mode
    return write_folder(path, ev)


def analyse(paths, power_dirs=(), **kw):
    ds, runs, synthetic = PF.load_inputs(paths, power_dirs)
    return PF.analyse(ds, runs, synthetic, **kw), ds, synthetic


class tmp:
    """A temporary directory as a context manager that returns its path."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="findback_test_")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ----------------------------------------------------------------------------- section 2 definitions

def test_at_marker_rule():
    assert M.R_PRESENT_M == 0.30
    assert M.at_marker("present", None) is True            # no measurement: at the marker
    assert M.at_marker("present", 0.0) is True
    assert M.at_marker("present", 0.30) is True            # "at most" 0.30 m
    assert M.at_marker("present", 0.3000001) is False
    assert M.at_marker("present", 5.67) is False
    for truth in ("displaced", "absent", "replaced"):
        assert M.at_marker(truth, None) is False
        assert M.at_marker(truth, 0.01) is False            # a small displacement is still displaced
    assert M.at_marker("skip", None) is None
    assert M.at_marker(None, None) is None
    assert M.at_marker("moved", 0.1) is None                # not a ground-truth value


def test_answers_and_outcomes():
    assert M.system_answer(False, True) == M.LAST_SEEN_HERE     # no capture: the confirmation is irrelevant
    assert M.system_answer(False, False) == M.LAST_SEEN_HERE
    assert M.system_answer(True, True) == M.HERE_NOW
    assert M.system_answer(True, False) == M.NOT_CONFIRMED
    assert M.outcome(M.LAST_SEEN_HERE, True) == "correct_lead"
    assert M.outcome(M.HERE_NOW, True) == "correct_lead"
    assert M.outcome(M.LAST_SEEN_HERE, False) == "wrong_lead"
    assert M.outcome(M.HERE_NOW, False) == "wrong_lead"
    assert M.outcome(M.NOT_CONFIRMED, False) == "correct_warning"
    assert M.outcome(M.NOT_CONFIRMED, True) == "lost_retrieval"
    assert M.confirmed_by_threshold(0.75, 0.75) and not M.confirmed_by_threshold(0.7499, 0.75)
    assert not M.confirmed_by_threshold(None, 0.75)


def hb_arrays():
    stale = np.array([not M.at_marker(t, m if t == "present" else None) for _, _, t, m, _, _, _ in HB])
    conf = np.array([s >= 0.75 for _, _, _, _, s, _, _ in HB])
    heur = np.array([pv for _, _, _, _, _, pv, _ in HB])
    calls = np.array([c for *_, c in HB], float)
    return stale, conf, heur, calls


def test_handbuilt_table_is_what_the_comments_say():
    stale, conf, heur, _ = hb_arrays()
    names = [q for q, *_ in HB]
    assert [n for n, s in zip(names, stale) if not s] == ["A", "B", "C"]
    assert [n for n, c in zip(names, conf) if c] == ["A", "C", "D", "G", "I", "K"]
    assert [n for n, h in zip(names, heur) if h] == ["B", "C", "E", "F", "H", "J", "K"]


def reference_replay(stale, conf, capture):
    """The same counts from the scalar definitions, one query at a time."""
    out = {k: 0 for k in M.OUTCOMES}
    for s, c, cap in zip(stale, conf, capture):
        out[M.outcome(M.system_answer(bool(cap), bool(c)), not s)] += 1
    return out


def test_every_policy_on_the_handbuilt_arrays():
    stale, conf, heur, calls = hb_arrays()
    masks = {"Memory": M.mask_memory_only(11), "Always": M.mask_always_verify(11), "Heuristic": heur, "Oracle": M.mask_oracle(stale)}
    for name, mask in masks.items():
        got = M.replay(stale, conf, mask, calls)
        want = HB_EXPECT[name]
        for k, v in want.items():
            assert got[k] == v, (name, k, got[k], v)
        assert got["n"] == 11
        assert sum(got[k] for k in M.OUTCOMES) == 11
        assert reference_replay(stale, conf, mask) == {k: got[k] for k in M.OUTCOMES}
        r = M.rates(got)
        assert close(sum(r[k + "_rate"] for k in M.OUTCOMES), 1.0)
        assert close(r["captures_per_query"], want["captures"] / 11)
        assert close(r["calls_per_query"], want["calls"] / 11)


def test_handbuilt_log_through_the_pipeline():
    with tmp() as d:
        A, ds, synthetic = analyse([handbuilt_e3(d)])
        assert not synthetic
        P = A.PS["E3"]
        assert P["n_all"] == 16 and P["n_usable"] == 14 and P["n_skip"] == 1 and P["n_unfinished"] == 1
        assert P["n_mode_excluded"] == 1                               # Q, logged in FREE mode
        assert dict(P["no_capture"]) == {"user_skipped": 1, "not_located": 1} and P["n_no_capture"] == 2
        assert P["n_paired_all"] == 11 and P["n"] == 11 and P["n_stale"] == 8 and P["n_sessions"] == 3
        assert (P["n_present_far"], P["n_present_unmeasured"], P["n_displaced_near"]) == (2, 1, 1)   # D E / B / I
        assert P["protocol"] == "leave-one-session-out"
        assert P["learned"] is None and "n too small" in P["learned_reason"]
        for name, want in HB_EXPECT.items():
            got = P["policies"][name]["counts"]
            for k, v in want.items():
                assert got[k] == v, (name, k, got[k], v)
        N = A.N
        assert N.get("EThreeQueries") == "14" and N.get("EThreePaired") == "11" and N.get("EThreeNoCapture") == "2"
        assert N.get("EThreeStaleRate") == "73\\% (8/11)"
        assert N.get("EThreeMemoryWrongLeadRate") == "73\\% (8/11)" and N.get("EThreeMemoryLostRate") == "0\\% (0/11)"
        assert N.get("EThreeAlwaysWrongLeadRate") == "36\\% (4/11)" and N.get("EThreeAlwaysLostRate") == "9\\% (1/11)"
        assert N.get("EThreeAlwaysLostOfAtMarkerRate") == "33\\% (1/3)"
        assert N.get("EThreeHeuristicWrongLeadRate") == "36\\% (4/11)" and N.get("EThreeHeuristicCaptures") == "0.64"
        assert N.get("EThreeOracleWrongLeadRate") == "36\\% (4/11)" and N.get("EThreeOracleLostRate") == "0\\% (0/11)"
        assert N.get("EThreeOracleCaptures") == "0.73" and N.get("EThreeAlwaysCalls") == "3.55"
        assert N.get("EThreeThreshold") == "0.75" and "on-device" in N.get("EThreeThresholdSource")
        # Three sessions: too few for an interval, and eleven queries: too few for the learned policy.
        assert N.get("EThreeMemoryWrongLeadCI") == PF.NSMALL and N.get("EThreeRiskAuc") == PF.NSMALL
        # Random baseline: exact expectation, linear in p.
        assert N.get("EThreeRandomHalfWrongLead") == "54.5\\%"         # (8/11 + 4/11) / 2
        assert N.get("EThreeRandomHalfLost") == "4.5\\%"               # (1/11) / 2
        # The paper's bare numbers refuse to print a percentage of 11 queries without its counts.
        assert N.get("NQueriesPaired") == "11"
        assert "n too small (8/11)" in N.get("WrongLeadMemoryPct")
        assert N.get("CapturesSelective") == "0.64"
        # T1: the untagged session P is nowhere.
        assert N.get("DataSessions") == "4" and N.get("DataSites") == "1"
        assert [N.get("Data" + t.capitalize()) for t in L.GROUND_TRUTHS] == ["6", "2", "4", "2", "1"]
        assert N.get("DataQueries") == "15" and N.get("DataCaptures") == "13"
        assert not any(q["query"] == "qP" for q in A.queries)
        # The tap after `displaced` is a displacement, never a marker error.
        rows = {q["query"]: q for q in A.queries}
        assert rows["qI"]["marker_error_m"] is None and rows["qI"]["displacement_m"] == 0.20
        assert rows["qD"]["marker_error_m"] == 0.31 and rows["qD"]["displacement_m"] is None
        assert rows["qG"]["marker_error_m"] is None and rows["qG"]["displacement_m"] is None
        # E5 has no data at all.
        assert N.get("EFiveQueries") == PF.TBD and N.get("EFiveMemoryWrongLeadRate") == PF.TBD
        # Measured verification behaviour, from the table at the top of this file.
        # At the marker A B C: A C confirm. Not at the marker D..K: D G I K confirm.
        assert P["confirm"]["counts"] == {"AtMarker": (2, 3), "Stale": (4, 8), "Absent": (1, 2), "Displaced": (1, 2),
                                          "Replaced": (1, 2), "PresentFar": (1, 2)}      # G of F G; I of H I; K of J K; D of D E
        assert N.get("EThreeConfirmQueries") == "11" and N.get("EThreeConfirmSessions") == "3"
        assert N.get("EThreeConfirmAtMarkerRate") == "67\\% (2/3)" and N.get("EThreeConfirmStaleRate") == "50\\% (4/8)"
        assert N.get("EThreeConfirmPresentFarRate") == "50\\% (1/2)" and N.get("EThreeConfirmAbsentCI") == PF.NSMALL   # three sessions
        assert "n too small (2/3)" in N.get("ConfirmAtMarkerPct") and "n too small (4/8)" in N.get("ConfirmStalePct")
        # Three and eight captures cannot feed the what-if model (20 of each needed): no curve, no number.
        assert A.W["state"] == "n too small" and A.W["curves"] is None and A.W["c_s"] is None
        assert all(N.get("WhatIfCapturesAuc" + k) == PF.NSMALL for k in PF.WHATIF_AUCS)
        # The on-device rule under its own name is the same number as the old macro.
        assert N.get("OnDeviceCaptures") == N.get("CapturesSelective") == "0.64"
        assert N.get("OnDeviceWrongLeadPct") == N.get("WrongLeadSelectivePct")
        assert N.get("OnDeviceCaptureSavingPct") == N.get("CaptureSavingPct")
        # No dev split, so no operating point can be fixed beforehand; eleven queries are too few anyway.
        assert P["op"] is None and N.get("FramePolicyCaptures") == PF.NSMALL


# ----------------------------------------------------------------------------- oracle, random, learned sweep

def test_oracle_properties():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(1, 60))
        stale = rng.random(n) < rng.random()
        conf = rng.random(n) < rng.random()
        got = M.replay(stale, conf, M.mask_oracle(stale))
        assert got["captures"] == int(stale.sum())                 # captures exactly the stale queries
        assert got["lost_retrieval"] == 0                          # never captures an item that is at the marker
        assert got["correct_lead"] == int((~stale).sum())
        assert got["wrong_lead"] == int((stale & conf).sum())      # only false confirmations remain
        assert got["wrong_lead"] <= M.replay(stale, conf, M.mask_memory_only(n))["wrong_lead"]


def test_random_baseline_is_the_diagonal():
    stale, conf, _, calls = hb_arrays()
    mem = M.rates(M.replay(stale, conf, M.mask_memory_only(11), calls))
    alw = M.rates(M.replay(stale, conf, M.mask_always_verify(11), calls))
    for k in ("wrong_lead_rate", "lost_retrieval_rate", "correct_lead_rate", "correct_warning_rate", "captures_per_query", "calls_per_query"):
        assert close(M.random_policy_expected(stale, conf, 0.0, calls)[k], mem[k]), k
        assert close(M.random_policy_expected(stale, conf, 1.0, calls)[k], alw[k]), k
        for p in (0.1, 0.25, 0.5, 0.9):                              # a straight line between the two
            assert close(M.random_policy_expected(stale, conf, p, calls)[k], (1 - p) * mem[k] + p * alw[k]), (k, p)
    half = M.random_policy_expected(stale, conf, 0.5)
    assert close(half["wrong_lead_rate"], 6 / 11) and close(half["lost_retrieval_rate"], 0.5 / 11)
    assert close(sum(half[k + "_rate"] for k in M.OUTCOMES), 1.0)


def test_random_baseline_matches_simulation():
    stale, conf, _, _ = hb_arrays()
    rng = np.random.default_rng(11)
    p, reps = 0.3, 20000
    sims = np.array([[r["wrong_lead"], r["lost_retrieval"], r["captures"]] for r in
                     (M.replay(stale, conf, rng.random(11) < p) for _ in range(reps))], float) / 11
    want = M.random_policy_expected(stale, conf, p)
    for col, key in enumerate(("wrong_lead_rate", "lost_retrieval_rate", "captures_per_query")):
        se = sims[:, col].std() / math.sqrt(reps)
        assert abs(sims[:, col].mean() - want[key]) < 4 * se, (key, sims[:, col].mean(), want[key])


def test_learned_sweep_by_hand_and_against_reference():
    stale, conf, _, _ = hb_arrays()
    #                A    B    C     D    E    F    G    H    I    J     K
    risk = np.array([.1, .2, .3, .9, .8, .7, .6, .5, .4, .35, .33])
    x, wl, lost, thr = M.sweep_risk(stale, conf, risk)
    # Captured in the order D E F G H I J K C B A. D G I K confirm, so capturing them changes nothing.
    assert np.allclose(x * 11, np.arange(12))
    assert np.allclose(wl * 11, [8, 8, 7, 6, 6, 5, 5, 4, 4, 4, 4, 4])
    assert np.allclose(lost * 11, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    assert thr[0] == np.inf and close(thr[-1], 0.1)
    for xi, wi, li, t in zip(x, wl, lost, thr):                   # each point equals a replay at that threshold
        ref = reference_replay(stale, conf, M.mask_risk(risk, t))
        assert close(wi * 11, ref["wrong_lead"]) and close(li * 11, ref["lost_retrieval"])
    # Ties are captured together: no point between them.
    x2, wl2, _, _ = M.sweep_risk(stale, conf, np.array([.5] * 11))
    assert np.allclose(x2, [0, 1]) and np.allclose(wl2 * 11, [8, 4])
    # The end points are memory only and always verify.
    assert close(wl[0], 8 / 11) and close(wl[-1], 4 / 11) and close(lost[-1], 1 / 11)
    assert close(M.curve_at(x, wl, 0.5), (5 + 5) / 2 / 11)        # between 5 and 6 captures of 11
    assert close(M.curve_area([0, 1], [8 / 11, 4 / 11]), 6 / 11)


# ----------------------------------------------------------------------------- ROC and thresholds

def test_auc_by_hand_and_against_sklearn():
    target = [0.9, 0.8, 0.7, 0.6]
    assert close(M.auc(target, [0.1, 0.2]), 1.0)
    assert close(M.auc(target, [0.3, 0.65]), 7 / 8)                 # only 0.6 < 0.65 is the wrong way round
    assert close(M.auc(target, [0.85, 0.5]), 5 / 8)
    assert close(M.auc(target, [0.1, 0.2, 0.3, 0.65, 0.85, 0.5]), 20 / 24)
    assert close(M.auc([0.5, 0.5], [0.5, 0.5]), 0.5)                # ties count one half
    assert close(M.auc([0.1], [0.9]), 0.0)
    assert math.isnan(M.auc([], [0.5]))
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(5)
    for _ in range(20):
        pos, neg = rng.normal(0.6, 0.2, 30).round(1), rng.normal(0.4, 0.2, 40).round(1)
        assert close(M.auc(pos, neg), roc_auc_score(np.r_[np.ones(30), np.zeros(40)], np.r_[pos, neg]), 1e-12)


def test_roc_points_and_youden():
    pos, neg = [0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.65, 0.85, 0.5]
    fpr, tpr, thr = M.roc_points(pos, neg)
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    i = list(thr).index(0.7)
    assert close(tpr[i], 3 / 4) and close(fpr[i], 1 / 6)            # confirm when similarity >= 0.7
    assert close(np.trapezoid(tpr, fpr), M.auc(pos, neg))
    # J = 4/4 - 2/6 for every threshold in (0.50, 0.60]; the tie goes to the highest, 0.60.
    assert M.youden_threshold(pos, neg) == 0.60
    assert M.youden_threshold([], neg) is None
    # Rule 2. Three positives pass rule 1 with margins .3 .2 -.1, three negatives with -.2 .05 nan.
    m = M.youden_margin([True] * 3, [0.3, 0.2, -0.1], [True] * 3, [-0.2, 0.05, float("nan")])
    # m <= 0.05: TP 2/3, FP 2/3 (nan always passes). m in (0.05, 0.20]: TP 2/3, FP 1/3 -> best; smallest such m on the grid.
    assert m == 0.06
    assert M.confirmed_by_margin(0.8, 0.75, None, 0.1) is True     # nothing to confuse it with
    assert M.confirmed_by_margin(0.8, 0.75, 0.05, 0.1) is False
    assert M.confirmed_by_margin(0.7, 0.75, 0.50, 0.1) is False    # rule 1 fails first


# ----------------------------------------------------------------------------- bootstrap

def test_bootstrap_resamples_whole_clusters():
    values = np.array([0, 0, 1, 1, 1, 1, 1], float)
    clusters = np.array(["a", "a", "a", "b", "b", "b", "b"])
    seen = []

    def stat(idx):
        seen.append(float(values[idx].mean()))
        return seen[-1]
    out = M.cluster_bootstrap(stat, clusters, n_boot=2000, min_clusters=2)
    draws = np.array(seen[1:])                                      # the first call is the full sample
    # With two clusters there are only three possible resamples: aa, ab (either order), bb.
    possible = [1 / 3, 5 / 7, 1.0]
    assert all(min(abs(d - p) for p in possible) < 1e-12 for d in draws)
    share = [np.mean(np.abs(draws - p) < 1e-12) for p in possible]
    assert abs(share[0] - 0.25) < 0.04 and abs(share[1] - 0.5) < 0.04 and abs(share[2] - 0.25) < 0.04
    assert out["ok"] and out["n_clusters"] == 2 and close(out["lo"], 1 / 3) and close(out["hi"], 1.0)


def test_bootstrap_is_wider_when_rows_are_clustered():
    values = np.repeat([0, 1] * 5, 20).astype(float)                # 10 sessions of 20 identical rows
    sessions = np.repeat(np.arange(10), 20)
    by_session = M.cluster_bootstrap(lambda i: values[i].mean(), sessions)
    by_row = M.cluster_bootstrap(lambda i: values[i].mean(), np.arange(200))
    w_session, w_row = by_session["hi"] - by_session["lo"], by_row["hi"] - by_row["lo"]
    assert 0.5 <= w_session <= 0.7, w_session                        # about 2 * 1.96 * sqrt(0.25 / 10)
    assert 0.10 <= w_row <= 0.18, w_row                              # about 2 * 1.96 * sqrt(0.25 / 200)
    assert w_session > 3 * w_row


def test_bootstrap_edge_cases():
    x = np.arange(12, dtype=float)
    same = M.cluster_bootstrap(lambda i: 7.0, np.repeat(np.arange(6), 2))
    assert same["lo"] == 7.0 and same["hi"] == 7.0                   # a constant has a zero-width interval
    few = M.cluster_bootstrap(lambda i: x[i].mean(), np.repeat(np.arange(3), 4))
    assert not few["ok"] and math.isnan(few["lo"]) and few["n_clusters"] == 3    # below MIN_BOOT_CLUSTERS
    def draws(seed):                                                  # the statistic of every resample, in order
        seen = []
        M.cluster_bootstrap(lambda i: seen.append(float(x[i].mean())) or seen[-1], np.repeat(np.arange(6), 2), n_boot=50, seed=seed)
        return seen
    assert draws(1) == draws(1) and draws(1) != draws(2)             # the seed fixes the resamples
    vec = M.cluster_bootstrap(lambda i: np.array([x[i].mean(), np.nan]), np.repeat(np.arange(6), 2))
    assert not math.isnan(vec["lo"][0]) and math.isnan(vec["lo"][1]) and vec["valid"][1] == 0.0
    empty = M.cluster_bootstrap(lambda i: 0.0, np.array([]))
    assert not empty["ok"]


def test_bootstrap_coverage():
    """With 30 sessions the 95% interval should cover the true mean about 95% of the time."""
    rng = np.random.default_rng(2026)
    reps, covered = 200, 0
    sessions = np.repeat(np.arange(30), 5)
    for r in range(reps):
        values = np.repeat(rng.normal(0, 1, 30), 5) + rng.normal(0, 0.5, 150)   # true mean 0
        out = M.cluster_bootstrap(lambda i: values[i].mean(), sessions, n_boot=300, seed=r)
        covered += out["lo"] <= 0.0 <= out["hi"]
    assert 0.88 <= covered / reps <= 0.99, covered / reps


# ----------------------------------------------------------------------------- learned policy: no leakage

def spy_fit(log):
    def fit(x_train, y_train, x_eval):
        log.append((x_train[:, 0].astype(int).tolist(), x_eval[:, 0].astype(int).tolist()))
        return np.full(len(x_eval), 0.5)
    return fit


def toy_rows(n_sessions=4, per_session=12):
    n = n_sessions * per_session
    x = np.zeros((n, len(M.FEATURES)))
    x[:, 0] = np.arange(n)                                           # column 0 carries the row index
    stale = np.tile([True, False], n // 2)
    sessions = np.repeat(["s%d" % i for i in range(n_sessions)], per_session)
    return x, stale, sessions


def test_dev_test_split_trains_on_dev_only():
    x, stale, sessions = toy_rows()
    splits = ["dev"] * 24 + ["test"] * 18 + [None] * 6               # six rows carry no split
    log = []
    out = M.out_of_sample_risk(x, stale, sessions, splits, fit=spy_fit(log))
    assert out["protocol"] == "dev/test" and len(log) == 1
    train, ev = log[0]
    assert train == list(range(24)) and ev == list(range(24, 42))
    assert not set(train) & set(ev)
    assert out["eval"].tolist() == [False] * 24 + [True] * 18 + [False] * 6
    assert np.isnan(out["risk"][:24]).all() and np.isnan(out["risk"][42:]).all() and not np.isnan(out["risk"][24:42]).any()
    assert out["n_train"] == 24 and out["warning"] is None
    # A session that holds both dev and test queries is reported.
    out2 = M.out_of_sample_risk(x, stale, sessions, ["dev"] * 30 + ["test"] * 18, fit=spy_fit([]))
    assert out2["warning"] and "s2" in out2["warning"]


def test_leave_one_session_out_never_scores_its_training_rows():
    x, stale, sessions = toy_rows()
    for splits in ([None] * 48, ["dev"] * 48, ["test"] * 48):        # one split only is not a split
        log = []
        out = M.out_of_sample_risk(x, stale, sessions, splits, fit=spy_fit(log))
        assert out["protocol"] == "leave-one-session-out" and len(log) == 4
        scored = []
        for train, ev in log:
            assert not set(train) & set(ev)
            assert len({sessions[i] for i in ev}) == 1               # one whole session is held out
            assert sessions[ev[0]] not in {sessions[i] for i in train}
            assert len(train) + len(ev) == 48
            scored += ev
        assert sorted(scored) == list(range(48)) and out["eval"].all()


def test_learned_policy_needs_enough_training_data():
    x, stale, sessions = toy_rows(2, 12)
    assert M.out_of_sample_risk(x, stale, sessions, [None] * 24)["risk"] is None          # two sessions
    x, stale, sessions = toy_rows(4, 6)
    out = M.out_of_sample_risk(x, stale, sessions, [None] * 24)                            # folds of 18 rows
    assert out["risk"] is None and "n too small" in out["reason"]
    x, stale, sessions = toy_rows(4, 12)
    one_class = np.zeros(48, bool)
    one_class[:3] = True                                                                    # 3 stale in all
    assert M.out_of_sample_risk(x, one_class, sessions, [None] * 48)["risk"] is None


def test_fit_predict_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    x_train = np.c_[rng.normal(0, 1, 200), np.full(200, 3.0)]        # the second column is constant
    y_train = x_train[:, 0] > 0
    x_eval = np.c_[np.linspace(-2, 2, 50), rng.normal(100, 50, 50)]  # wild values in the constant column
    risk = M.fit_predict(x_train, y_train, x_eval)
    assert M.auc(risk[x_eval[:, 0] > 0], risk[x_eval[:, 0] <= 0]) == 1.0
    assert np.all(np.diff(risk) > 0)                                  # the constant column is ignored
    # The score of a query must not depend on which other queries are scored with it: nothing
    # about the evaluation rows (their mean, their spread) may enter the model.
    one = np.array([[0.3, 3.0]])
    alone = M.fit_predict(x_train, y_train, one)[0]
    with_twin = M.fit_predict(x_train, y_train, np.r_[one, [[0.0, 3.0]]])[0]
    with_outliers = M.fit_predict(x_train, y_train, np.r_[one, [[500.0, -40.0]] * 30])[0]
    assert alone == with_twin == with_outliers


def test_features_are_only_the_pre_capture_signals():
    assert list(M.SIGNALS) == ["memory age", "mobility class", "earlier moves", "position source", "metres since fix",
                               "tracking losses", "time since fix", "fix type"]
    q = {"age_min": math.e - 1, "mobility": "often", "move_count": 2.0, "position_source": "MAP", "meters_since_fix": 0.0,
         "losses_since_fix": 1.0, "ms_since_fix": 60000.0 * (math.e - 1), "fix_type": "IMAGE",
         # Things a policy must not see; they must not change the features.
         "first_capture": {"sim": 0.99}, "truth": "absent", "marker_error_m": 3.0}
    f = dict(zip(M.FEATURES, M.features_of(q)))
    assert f == {"log_age_min": 1.0, "mobility": 2.0, "move_count": 2.0, "source_is_map": 1.0, "log_m_since_fix": 0.0,
                 "losses_since_fix": 1.0, "log_min_since_fix": 1.0, "fix_is_image": 1.0, "fix_is_spot": 0.0}
    blank = dict(zip(M.FEATURES, M.features_of({})))                 # everything missing: not located, defaults
    assert blank["mobility"] == 1.0 and blank["move_count"] == 0.0 and blank["fix_is_image"] == blank["fix_is_spot"] == 0.0
    assert M.columns_for(["fix type"]) == [7, 8]


def test_pipeline_learned_policy_on_synthetic_arrays():
    """End to end on a larger hand-made set with a dev and a test split: the learned policy is
    fitted on dev, every policy is replayed on test only."""
    with tmp() as d:
        clock, ev = counter(), []
        rng = np.random.default_rng(4)
        k = 0
        for s, split in (("d1", "dev"), ("d2", "dev"), ("t1", "test"), ("t2", "test"), ("t3", "test"), ("t4", "test")):
            tag = "home=H;exp=E3;split=" + split
            ev.append(event(s, next(clock), "session_start", tag))
            for _ in range(15):
                k += 1
                walked = float(rng.uniform(0, 60))
                stale = walked > 30                                   # a signal that fully explains staleness
                ev += query_events(clock, "x%d" % k, s, "present", 0.5 if stale else 0.05, 0.2 if stale else 0.9,
                                   True, 3, tag, extra_start={"metersSinceFix": walked})
        A, _, _ = analyse([write_folder(d, ev)])
        P = A.PS["E3"]
        assert P["protocol"] == "dev/test" and P["n_train"] == 30 and P["n"] == 60 and P["n_sessions"] == 4
        assert P["learned"] is not None and P["learned"]["auc"] > 0.95
        assert P["policies"]["Memory"]["counts"]["n"] == 60          # replayed on the test split only
        assert P["policies"]["Memory"]["boot"]["ok"]                 # four sessions: an interval exists
        # Metres walked is a frame signal: the frame-only score has it, the world-only score and the age do not.
        assert P["curves"]["Frame"]["auc"] > 0.95
        assert P["curves"]["World"]["auc"] == 0.5 and P["curves"]["Age"]["auc"] == 0.5   # constant in this data: all ties
        assert "memory age" in P["constant_signals"]
        N = A.N
        assert N.get("EThreeProtocol") == "dev/test" and N.get("EThreeTrainQueries") == "30"
        assert float(N.get("EThreeFrameAuc")) > 0.95 and N.get("EThreeWorldAuc") == "0.500" and N.get("EThreeAgeAuc") == "0.500"
        # The frame-signals operating point: threshold fixed on the 30 dev queries, replayed on the 60 test queries.
        op = P["op"]
        assert op is not None and op["dev_n"] == 30 and op["counts"]["n"] == 60
        dev = [q for q in A.queries if q["split"] == "dev"]
        dev_stale = sum(q["marker_error_m"] > 0.30 for q in dev)
        # Always verifying leaves no wrong lead here (a stale capture never confirms), so within 5 points on
        # 30 dev queries means at most one wrong lead (1/30 = 3.3%, 2/30 = 6.7%): all stale dev queries but one.
        assert close(op["dev_captures"], (dev_stale - 1) / 30.0) and close(op["dev_wrong"], 1 / 30.0) and op["dev_always_wrong"] == 0.0
        assert (op["mask"] == (P["curves"]["Frame"]["risk"] >= op["threshold"])).all()
        assert op["counts"]["lost_retrieval"] == 0                    # a fresh capture always confirms here
        assert op["rates"]["wrong_lead_rate"] <= 0.15 and 0.2 < op["rates"]["captures_per_query"] < 0.7
        assert N.get("EThreeFrameOpDevQueries") == "30" and N.get("FramePolicyCaptures") == N.get("EThreeFrameOpCaptures")
        assert N.get("EThreeFrameOpCaptures") == "%.2f" % op["rates"]["captures_per_query"]
        assert N.get("FramePolicyCaptureSavingPct") == "%.0f" % (100.0 * (60 - op["counts"]["captures"]) / 60)
        # Verification behaviour and the what-if model on the 90 paired queries: c_p = 1, c_s = 0.
        n_stale = sum(q["marker_error_m"] > 0.30 for q in A.queries)
        assert P["confirm"]["counts"]["AtMarker"] == (90 - n_stale, 90 - n_stale) and P["confirm"]["counts"]["Stale"] == (0, n_stale)
        assert P["confirm"]["counts"]["PresentFar"] == (0, n_stale) and P["confirm"]["counts"]["Absent"] == (0, 0)
        assert N.get("EThreeConfirmAbsentRate") == PF.TBD and N.get("ConfirmStalePct") == "0"
        assert A.W["state"] is None and A.W["c_p"] == 1.0 and A.W["c_s"] == 0.0
        # Base rate 0.3, c_s 0: TPR needed 1 - 0.05/0.3 = 5/6. AUC 0.5: 5/6 captures. AUC 1: 0.3 * 5/6 = 0.25.
        assert N.get("WhatIfCapturesAucFifty") == "0.83" and N.get("WhatIfCapturesAucHundred") == "0.25"
        assert float(N.get("WhatIfCapturesAucFifty")) > float(N.get("WhatIfCapturesAucSeventy")) > float(N.get("WhatIfCapturesAucEighty")) \
            > float(N.get("WhatIfCapturesAucNinety")) > float(N.get("WhatIfCapturesAucHundred"))
        labels = [label for label, _ in A.W["marks"]]
        assert labels == ["age only, E3", "all signals, E3", "frame signals, frame staleness"], labels
        # Frame staleness on the same present queries: the label is the marker error, so it is the same split.
        assert A.FR["protocol"] == "dev/test" and A.FR["n_train"] == 30 and A.FR["n"] == 60 and A.FR["auc"] > 0.95
        # The measured policy figure: three swept curves, the random line, four fixed points and the dev-chosen point.
        plt = PF.setup_matplotlib()
        fig = PF.fig_policy(plt, P)
        legend = [t.get_text() for t in fig.legends[0].get_texts()]
        assert sorted(legend) == sorted(["Random capture", "All signals (learned)", "Frame signals only (learned)", "Age-only threshold",
                                         "Memory only", "Always verify", "On-device rule (logged)", "Oracle",
                                         "Frame signals, point chosen on dev"]), legend
        for key in ("All", "Frame", "Age"):
            for ax, field in zip(fig.axes, ("wl", "lost")):
                style = PF.POLICY_STYLE[key]
                line = [ln for ln in ax.lines if ln.get_ls() == style["ls"] and ln.get_color() == style["color"]]
                assert len(line) == 1 and np.allclose(line[0].get_xdata(), P["curves"][key]["x"])
                assert np.allclose(line[0].get_ydata(), 100 * P["curves"][key][field]), (key, field)
        point = [ln for ln in fig.axes[0].lines if ln.get_marker() == "P"]
        assert len(point) == 1 and close(point[0].get_xdata()[0], op["rates"]["captures_per_query"])
        assert close(point[0].get_ydata()[0], 100 * op["rates"]["wrong_lead_rate"])
        assert not fig.findobj(lambda o: getattr(o, "get_gid", lambda: None)() in ("fb-note", "fb-missing"))
        plt.close(fig)


# ----------------------------------------------------------------------------- operating point, what-if model

def test_operating_threshold_by_hand():
    #                 risk  stale  capture confirms
    rows = [(0.90, True, False), (0.80, True, False), (0.70, False, True), (0.60, True, False), (0.50, False, True),
            (0.40, True, True), (0.30, False, True), (0.20, False, False), (0.10, False, True), (0.05, False, True)]
    risk, stale, conf = (np.array(c) for c in zip(*rows))
    # Always verifying leaves one wrong lead (the 0.40 row confirms a stale lead): 10%.
    # Wrong leads after capturing the k riskiest: k=0: 4, k=1: 3, k=2: 2, k=3: 2, k=4: 1, ... k=10: 1.
    # Within 5 points of 10% means at most one wrong lead: first reached at k = 4, the row with risk 0.60.
    thr, caps, wl = M.choose_operating_threshold(stale, conf, risk)
    assert (thr, caps) == (0.60, 0.4) and close(wl, 0.1)
    assert M.mask_risk(risk, thr).tolist() == [True] * 4 + [False] * 6
    # Within 10 points: two wrong leads are allowed, reached at k = 2.
    thr, caps, wl = M.choose_operating_threshold(stale, conf, risk, tolerance=0.10)
    assert (thr, caps) == (0.80, 0.2) and close(wl, 0.2)
    # Within 30 points: memory alone (4 wrong leads, 40%) already qualifies, so nothing is captured.
    thr, caps, wl = M.choose_operating_threshold(stale, conf, risk, tolerance=0.30)
    assert thr == float("inf") and caps == 0.0 and close(wl, 0.4) and not M.mask_risk(risk, thr).any()
    # Tolerance 0: as good as always verifying. The 0.40 row need not be captured (it confirms anyway).
    thr, caps, wl = M.choose_operating_threshold(stale, conf, risk, tolerance=0.0)
    assert (thr, caps) == (0.60, 0.4)
    # Equal scores are captured together: a threshold cannot split them.
    thr, caps, wl = M.choose_operating_threshold([True, True, False, False], [False, False, True, True], [0.5, 0.5, 0.5, 0.1], tolerance=0.0)
    assert (thr, caps, wl) == (0.5, 0.75, 0.0)
    # Brute force over every threshold on random data.
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(5, 40))
        st, cf, rk = rng.random(n) < 0.4, rng.random(n) < 0.5, np.round(rng.random(n), 1)
        always = np.mean(st & cf)
        best = (0.0, float("inf"))
        if np.mean(st) > always + 0.05 + 1e-12:
            ok = [(float(np.mean(rk >= t)), float(t)) for t in np.unique(rk)
                  if reference_replay(st, cf, rk >= t)["wrong_lead"] / n <= always + 0.05 + 1e-12]
            best = min(ok)
        thr, caps, _ = M.choose_operating_threshold(st, cf, rk)
        assert (caps, thr) == best, (caps, thr, best)
    assert M.choose_operating_threshold([], [], []) [0] == float("inf")


def test_operating_point_is_fitted_and_chosen_on_dev_rows_only():
    x, stale, _ = toy_rows(4, 12)
    splits = ["dev"] * 24 + ["test"] * 12 + [None] * 12
    log = []
    dev, risk = M.dev_risk_for_operating_point(x, stale, splits, fit=spy_fit(log))
    assert dev.tolist() == list(range(24)) and risk.shape == (24,)
    assert log == [(list(range(24)), list(range(24)))]                # trained on dev, scored on dev, nothing else touched
    # A column subset reaches the model as given.
    seen = []
    M.dev_risk_for_operating_point(x, stale, splits, columns=[2, 5], fit=lambda a, b, c: seen.append((a.shape, c.shape)) or np.zeros(len(c)))
    assert seen == [((24, 2), (24, 2))]
    # Too few dev rows of one kind: no model, no operating point.
    few = stale.copy()
    few[:24] = [True] * 4 + [False] * 20
    assert M.dev_risk_for_operating_point(x, few, splits)[1] is None
    assert M.dev_risk_for_operating_point(x, stale, [None] * 48)[1] is None


def phi(x):
    """Standard normal CDF from math.erf: independent of the scipy functions the code under test uses."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_inv(p):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if phi(mid) < p else (lo, mid)
    return (lo + hi) / 2.0


def test_whatif_formulas_by_hand():
    # The three formulas of the spec with b = 0.3, TPR = 0.8, FPR = 0.2, c_p = 0.9, c_s = 0.1.
    r = M.whatif_rates(0.3, 0.8, 0.2, 0.9, 0.1)
    assert close(r["wrong_lead_rate"], 0.084)                         # 0.3 * (1 - 0.8 * 0.9)
    assert close(r["lost_retrieval_rate"], 0.014)                     # 0.7 * 0.2 * 0.1
    assert close(r["captures_per_query"], 0.38)                       # 0.3 * 0.8 + 0.7 * 0.2
    # The equal-variance binormal curve. With d = 1 the AUC is Phi(1/sqrt 2) = 0.76025, and at TPR 0.5
    # (the stale mean) the FPR is Phi(-1) = 0.158655.
    auc_d1 = phi(1 / math.sqrt(2))
    assert close(auc_d1, 0.760250, 1e-6) and close(M.binormal_fpr(0.5, auc_d1), 0.158655, 1e-6)
    assert close(M.binormal_fpr(phi(1.0), auc_d1), 0.5, 1e-9)         # and at FPR 0.5 the TPR is Phi(1)
    for auc in (0.5, 0.6, 0.76, 0.9, 0.99):
        tpr = np.linspace(0, 1, 20001)
        fpr = np.array([M.binormal_fpr(t, auc) for t in tpr])
        assert fpr[0] == 0.0 and fpr[-1] == 1.0 and (np.diff(fpr) >= 0).all()
        assert close(1.0 - np.trapezoid(fpr, tpr), auc, 2e-4)         # the area under the curve is the AUC it was asked for
        assert (fpr <= tpr + 1e-12).all()

    # A population with exactly b = 0.3, c_p = 0.9, c_s = 0.1: 30 stale (3 confirm), 70 at the marker (63 confirm).
    stale = np.array([True] * 30 + [False] * 70)
    conf = np.array([True] * 3 + [False] * 27 + [True] * 63 + [False] * 7)
    # End 1, AUC 0.5: the curve is the diagonal, and the model is the random-capture line of the replay.
    for p_capture in (0.0, 0.25, 0.5, 22 / 27, 1.0):
        assert close(M.binormal_fpr(p_capture, 0.5), p_capture, 1e-12)
        model = M.whatif_rates(0.3, p_capture, M.binormal_fpr(p_capture, 0.5), 0.9, 0.1)
        measured = M.random_policy_expected(stale, conf, p_capture)
        for k in ("wrong_lead_rate", "lost_retrieval_rate", "captures_per_query"):
            assert close(model[k], measured[k], 1e-12), (p_capture, k)
    # End 2, AUC 1: no fresh query is captured, and at TPR 1 the model is the oracle of the replay.
    for tpr in (0.0, 0.3, 22 / 27, 1.0):
        assert M.binormal_fpr(tpr, 1.0) == 0.0
    model = M.whatif_rates(0.3, 1.0, M.binormal_fpr(1.0, 1.0), 0.9, 0.1)
    oracle = M.rates(M.replay(stale, conf, M.mask_oracle(stale)))
    for k in ("wrong_lead_rate", "lost_retrieval_rate", "captures_per_query"):
        assert close(model[k], oracle[k], 1e-12), k
    assert close(model["wrong_lead_rate"], 0.03) and model["lost_retrieval_rate"] == 0.0 and close(model["captures_per_query"], 0.3)
    # TPR 0 is memory only, and TPR = FPR = 1 is always verify, whatever the AUC.
    for name, t in (("Memory", 0.0), ("Always", 1.0)):
        mask = M.mask_memory_only(100) if t == 0.0 else M.mask_always_verify(100)
        want = M.rates(M.replay(stale, conf, mask))
        got = M.whatif_rates(0.3, t, t, 0.9, 0.1)
        assert all(close(got[k], want[k], 1e-12) for k in got), name

    # Captures needed. Always verifying leaves 0.3 * 0.1 = 3% wrong leads; within 5 points is 8%:
    # 0.3 * (1 - 0.9 TPR) <= 0.08 gives TPR >= 22/27 = 0.8148.
    caps, tpr, fpr = M.whatif_captures_needed(0.3, 0.5, 0.1)
    assert close(tpr, 22 / 27) and close(fpr, 22 / 27) and close(caps, 22 / 27)          # AUC 0.5: a random 81.5% of all queries
    assert close(M.whatif_rates(0.3, tpr, fpr, 0.9, 0.1)["wrong_lead_rate"], 0.08)
    caps, tpr, fpr = M.whatif_captures_needed(0.3, 1.0, 0.1)
    assert close(tpr, 22 / 27) and fpr == 0.0 and close(caps, 0.3 * 22 / 27)             # AUC 1: 81.5% of the stale 30% = 0.2444
    # In between, by the formula with an independent normal CDF: FPR = Phi(Phi^-1(TPR) - sqrt(2) Phi^-1(AUC)).
    for auc in (0.6, 0.7, 0.8, 0.9, 0.95):
        want_fpr = phi(phi_inv(22 / 27) - math.sqrt(2) * phi_inv(auc))
        caps, tpr, fpr = M.whatif_captures_needed(0.3, auc, 0.1)
        assert close(fpr, want_fpr, 1e-7) and close(caps, 0.3 * 22 / 27 + 0.7 * want_fpr, 1e-7), auc
    assert close(M.whatif_captures_needed(0.3, 0.8, 0.1)[0], 0.51339, 1e-5)              # the value worked out on paper
    needed = [M.whatif_captures_needed(0.3, a, 0.1)[0] for a in np.linspace(0.5, 1.0, 51)]
    assert all(b < a for a, b in zip(needed, needed[1:]))             # a better predictor never needs more captures
    # Other tolerances, and the flat case: when memory alone is within the tolerance no capture is needed.
    assert close(M.whatif_captures_needed(0.5, 0.5, 0.0, tolerance=0.10)[0], 0.8)        # TPR 1 - 0.10/0.5
    assert M.whatif_captures_needed(0.1, 0.9, 0.6) == (0.0, 0.0, 0.0)                    # 0.1 * 0.4 = 4 points to gain
    assert M.whatif_captures_needed(0.05, 0.5, 0.0) == (0.0, 0.0, 0.0)                   # exactly 5 points: still none
    assert M.whatif_captures_needed(0.3, 0.7, 1.0) == (0.0, 0.0, 0.0)                    # captures that always confirm help nobody
    assert M.WL_TOLERANCE == 0.05 and PF.WHATIF_BASE_RATES == (0.1, 0.3, 0.5) and PF.WHATIF_HEADLINE_BASE == 0.3


def whatif_log(path, n_marker=(18, 20), n_stale=(2, 20)):
    """Paired E3 queries with chosen confirmation counts: (confirming, all) at the marker and not."""
    clock, ev = counter(), []
    tag = "home=H;exp=E3"
    for s in ("s1", "s2", "s3", "s4"):
        ev.append(event(s, next(clock), "session_start", tag))
    k = 0
    for (yes, n), truth in ((n_marker, "present"), (n_stale, "absent")):
        for i in range(n):
            k += 1
            ev += query_events(clock, "w%d" % k, "s%d" % (1 + k % 4), truth, 0.05 if truth == "present" else None,
                               0.9 if i < yes else 0.2, True, 3, tag, extra_start={"ageMin": float(k)})
    return write_folder(path, ev)


def test_whatif_model_takes_the_measured_shares_and_needs_enough_of_them():
    plt = PF.setup_matplotlib()
    with tmp() as d:
        A, _, _ = analyse([whatif_log(os.path.join(d, "a"))])
        W, N = A.W, A.N
        assert W["state"] is None and W["counts"] == ((18, 20), (2, 20)) and W["c_p"] == 0.9 and W["c_s"] == 0.1
        assert N.get("ConfirmAtMarkerPct") == "90" and N.get("ConfirmStalePct") == "10"
        assert N.get("EThreeConfirmAtMarkerRate") == "90\\% (18/20)" and N.get("EThreeConfirmAbsentRate") == "10\\% (2/20)"
        assert N.get("EThreeConfirmAtMarkerCI").startswith("[") and N.get("EThreeConfirmDisplacedRate") == PF.TBD
        # The headline numbers are the model at base rate 0.3 with the measured c_s = 0.1 (hand values in the test above).
        assert N.get("WhatIfCapturesAucFifty") == "0.81" and N.get("WhatIfCapturesAucEighty") == "0.51" and N.get("WhatIfCapturesAucHundred") == "0.24"
        assert N.get("WhatIfBaseRatePct") == "30" and N.get("WhatIfTolerancePoints") == "5"
        assert list(W["curves"]) == [0.1, 0.3, 0.5] and all(c.shape == PF.WHATIF_GRID.shape for c in W["curves"].values())
        assert PF.WHATIF_GRID[0] == 0.5 and PF.WHATIF_GRID[-1] == 1.0
        assert close(W["curves"][0.3][0], 22 / 27) and close(W["curves"][0.3][-1], 0.3 * 22 / 27)      # both ends of the 0.3 curve
        assert close(W["curves"][0.5][0], 1 - 0.05 / 0.45) and close(W["curves"][0.1][-1], 0.1 * (1 - 0.05 / 0.09))
        fig = PF.fig_whatif(plt, W)
        ax = fig.axes[0]
        assert "model" in ax.get_xlabel() and "model" in ax.get_ylabel()
        curves = [ln for ln in ax.lines if len(ln.get_xdata()) > 1]
        assert len(curves) == 3 and all(ln.get_label().startswith("model, ") for ln in curves)
        assert [ln.get_label() for ln in curves] == ["model, 10% of queries stale", "model, 30% of queries stale", "model, 50% of queries stale"]
        assert np.allclose(curves[1].get_ydata(), W["curves"][0.3])
        # The measured AUCs sit on the AUC axis, each labelled as measured.
        marks = [ln for ln in ax.lines if len(ln.get_xdata()) == 1]
        assert len(marks) == len(W["marks"]) >= 1 and all(ln.get_label().startswith("measured AUC: ") for ln in marks)
        assert all(ln.get_ydata()[0] == 0 and 0.5 <= ln.get_xdata()[0] <= 1.0 for ln in marks)
        assert not fig.findobj(lambda o: getattr(o, "get_gid", lambda: None)() == "fb-note")
        plt.close(fig)
        # The age-only AUC of this log by brute force: age is the query number, the stale queries are the later ones.
        age = PF.compute_policy("E3", A.queries, A.E2["tau"])["curves"]["Age"]
        assert age["auc"] == 1.0 and ("age only, E3", 1.0) in W["marks"]

        # One capture short on either side: a placeholder, no curve, no number.
        for j, kw in enumerate(({"n_marker": (17, 19)}, {"n_stale": (2, 19)})):
            A, _, _ = analyse([whatif_log(os.path.join(d, "b%d" % j), **kw)])
            assert A.W["state"] == "n too small" and A.W["curves"] is None and A.W["headline"] is None
            assert all(A.N.get("WhatIfCapturesAuc" + k) == PF.NSMALL for k in PF.WHATIF_AUCS)
            fig = PF.fig_whatif(plt, A.W)
            notes = [t.get_text() for t in fig.findobj(lambda o: getattr(o, "get_gid", lambda: None)() == "fb-note")]
            assert len(notes) == 1 and notes[0].startswith("n too small (") and not fig.axes[0].lines
            assert "model" in fig.axes[0].get_xlabel() and "model" in fig.axes[0].get_ylabel()
            plt.close(fig)
    assert M.MIN_N_SHARE == 20


# ----------------------------------------------------------------------------- frame staleness, age-only policy, cross-room guidance

def brute_auc(score, label):
    """Share of (positive, negative) pairs ranked the right way round, ties counting a half."""
    pos = [s for s, y in zip(score, label) if y]
    neg = [s for s, y in zip(score, label) if not y]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def frame_log(path, far_in_test=12):
    """E1 and E3 present queries whose marker error depends on the metres walked, plus rows that must not count.
    `far_in_test`: how many test queries in all get a marker error over 0.30 m."""
    clock, ev, k = counter(), [], 0
    rows = []                                                         # (exp, split, metres, far?) of the rows that count
    plan = (("d1", "E3", "dev"), ("d2", "E3", "dev"), ("t1", "E3", "test"), ("t2", "E3", "test"), ("e1", "E1", "test"), ("e2", "E1", "test"))
    for s, exp, split in plan:
        tag = "home=H;exp=%s;split=%s" % (exp, split)
        ev.append(event(s, next(clock), "session_start", tag, "FREE" if exp == "E1" else "PAIRED"))
        for i in range(12):
            k += 1
            metres = float(3 * i + (k % 3))                           # 0 .. 35, different in every session
            far = metres > 22
            if split == "test":
                far = far and sum(r[3] for r in rows if r[1] == "test") < far_in_test
                if s == "t1" and i == 0:
                    far = True                                        # one query that walked nothing and is far off anyway
                if s == "e1" and i == 11:
                    far = False                                       # and one long walk that stayed accurate
            measured = 0.45 if far else 0.08
            if s == "e2" and i == 5:
                measured, far = None, False                           # present but never measured: counts as not stale
            if s == "e2" and i == 6:
                measured = 0.30                                       # exactly on the rule: not over it, so not stale
            # E1 queries have no capture at all (sim None); E3 queries have one.
            ev += query_events(clock, "f%d" % k, s, "present", measured, None if exp == "E1" else 0.9, True, 3, tag,
                               "FREE" if exp == "E1" else "PAIRED", extra_start={"metersSinceFix": metres})
            rows.append((exp, split, metres, far))
        # Never part of frame staleness: the item was moved, or the query is not usable, or it is another experiment.
        ev += query_events(clock, "a%s" % s, s, "absent", None, 0.2, True, 3, tag, extra_start={"metersSinceFix": 99.0})
        ev += query_events(clock, "m%s" % s, s, "displaced", 0.9, 0.2, True, 3, tag, extra_start={"metersSinceFix": 99.0})
        ev += query_events(clock, "k%s" % s, s, "skip", 0.9, 0.2, True, 3, tag, extra_start={"metersSinceFix": 99.0})
    ev.append(event("b1", next(clock), "session_start", "home=H;exp=E2;split=test", "BENCHMARK"))
    ev += query_events(clock, "bench", "b1", "present", 0.9, 0.9, True, 3, "home=H;exp=E2;split=test", "BENCHMARK")
    return write_folder(path, ev), rows


def test_frame_staleness_auc_from_e1_and_e3_present_queries():
    with tmp() as d:
        folder, rows = frame_log(os.path.join(d, "a"))
        A, _, _ = analyse([folder])
        F, N = A.FR, A.N
        test = [r for r in rows if r[1] == "test"]
        assert F["n_rows"] == 72 and dict(F["by_exp"]) == {"E3": 48, "E1": 24} and F["n_unmeasured"] == 1
        assert F["protocol"] == "dev/test" and F["n_train"] == 24 and F["n"] == 48 and F["n_sessions"] == 4
        assert F["n_stale"] == sum(r[3] for r in test) == 12 and F["n_split_ignored"] == 0
        # Only the metres vary, so the fitted score orders the test queries by metres walked: the AUC is
        # that of the metres themselves, counted pair by pair.
        want = brute_auc([r[2] for r in test], [r[3] for r in test])
        assert 0.6 < want < 1.0 and close(F["auc"], want, 1e-12)
        assert N.get("FrameStaleAuc") == N.get("FrameStaleAUC") == "%.3f" % want
        assert N.get("FrameStaleQueries") == "48" and N.get("FrameStaleSessions") == "4" and N.get("FrameStaleTrainQueries") == "24"
        assert N.get("FrameStaleRate") == "25\\% (12/48)" and N.get("FrameStalePct") == "25" and N.get("FrameStaleProtocol") == "dev/test"
        ci = re.fullmatch(r"\[(\d\.\d{3}), (\d\.\d{3})\]", N.get("FrameStaleAucCI"))
        assert ci and float(ci.group(1)) <= want <= float(ci.group(2))
        assert ("frame signals, frame staleness", F["auc"]) in A.W["marks"]
        # E1 rows are in it although they have no capture; in the policy replay they are not.
        assert A.PS["E3"]["n_paired_all"] == 48 + 8 and A.PS["E3"]["n"] == 24 + 4       # E3 only: present + absent + displaced
        # The label is the marker error alone: world staleness (absent, displaced) is not in these rows at all.
        assert A.PS["E3"]["n_stale"] > sum(r[3] for r in test if r[0] == "E3")

        # Nine far-off test queries: an AUC needs ten of each kind.
        folder, rows = frame_log(os.path.join(d, "b"), far_in_test=9)
        A, _, _ = analyse([folder])
        assert A.FR["n_stale"] == sum(r[3] for r in rows if r[1] == "test") == 9 and A.FR["auc"] is None and A.FR["auc_small"]
        assert A.N.get("FrameStaleAuc") == PF.NSMALL and A.N.get("FrameStaleAucCI") == PF.NSMALL and A.N.get("FrameStaleRate") == "19\\% (9/48)"
        assert not any(label.startswith("frame signals") for label, _ in A.W["marks"])


def test_frame_scores_never_see_the_world_signals():
    """Memory age gives staleness away completely here and the frame signals say nothing. A score that
    is supposed to use the frame signals only must therefore be poor; if it is good, it has seen the age."""
    with tmp() as d:
        clock, ev = counter(), []
        rng = np.random.default_rng(7)
        k = 0
        for s, split in (("d1", "dev"), ("d2", "dev"), ("t1", "test"), ("t2", "test"), ("t3", "test"), ("t4", "test")):
            tag = "home=H;exp=E3;split=" + split
            ev.append(event(s, next(clock), "session_start", tag))
            for _ in range(15):
                k += 1
                age, walked = float(rng.uniform(0, 100)), float(rng.uniform(0, 60))
                far = age > 50
                ev += query_events(clock, "y%d" % k, s, "present", 0.5 if far else 0.05, 0.2 if far else 0.9, True, 3, tag,
                                   extra_start={"ageMin": age, "metersSinceFix": walked})
        A, _, _ = analyse([write_folder(d, ev)])
        P, F = A.PS["E3"], A.FR
        cv = P["curves"]
        assert cv["Age"]["auc"] == 1.0 and cv["World"]["auc"] > 0.95 and cv["All"]["auc"] > 0.9
        assert cv["Frame"]["auc"] < 0.75                              # the frame-only policy curve
        assert F["auc"] is not None and F["auc"] < 0.75               # the frame-staleness AUC
        assert close(F["auc"], cv["Frame"]["auc"], 1e-12)             # same rows, same label, same columns here
        # The operating point: the threshold comes from the frame-only model fitted on dev, and is applied to
        # the frame-only score of the test queries. Recomputed here from the frame columns named one by one.
        frame_cols = [M.FEATURES.index(c) for c in ("source_is_map", "log_m_since_fix", "losses_since_fix", "log_min_since_fix",
                                                    "fix_is_image", "fix_is_spot")]
        assert M.columns_for(M.FRAME_SIGNALS) == frame_cols == [3, 4, 5, 6, 7, 8] and M.columns_for(M.WORLD_SIGNALS) == [0, 1, 2]
        rows = [q for q in A.queries if q["exp"] == "E3"]
        x = M.feature_matrix(rows)
        stale = np.array([q["marker_error_m"] > 0.30 for q in rows])
        conf = np.array([q["first_capture"]["sim"] >= 0.75 for q in rows])
        is_dev = np.array([q["split"] == "dev" for q in rows])
        dev_risk = M.fit_predict(x[is_dev][:, frame_cols], stale[is_dev], x[is_dev][:, frame_cols])
        thr, caps, _ = M.choose_operating_threshold(stale[is_dev], conf[is_dev], dev_risk)
        test_risk = M.fit_predict(x[is_dev][:, frame_cols], stale[is_dev], x[~is_dev][:, frame_cols])
        op = P["op"]
        assert op["threshold"] == thr and close(op["dev_captures"], caps)
        assert np.allclose(cv["Frame"]["risk"], test_risk) and (op["mask"] == (test_risk >= thr)).all()
        assert op["counts"] == M.replay(stale[~is_dev], conf[~is_dev], test_risk >= thr, np.full(60, 3.0))
        all_risk = M.fit_predict(x[is_dev], stale[is_dev], x[~is_dev])
        assert not (op["mask"] == (all_risk >= thr)).all()            # and it is not the all-signals score


def age_log(path):
    clock, ev = counter(), []
    tag = "home=H;exp=E3"
    #        age (min), truth: older memories are more often stale, with exceptions
    spec = [(5, "present"), (10, "present"), (15, "absent"), (20, "present"), (25, "present"), (30, "present"),
            (35, "present"), (40, "absent"), (45, "present"), (50, "present"), (55, "present"), (60, "displaced"),
            (65, "present"), (70, "absent"), (75, "present"), (80, "absent"), (85, "replaced"), (90, "present"),
            (95, "absent"), (100, "absent"), (105, "replaced"), (110, "absent"), (None, "present"), (None, "absent")]
    for s in ("s1", "s2", "s3", "s4"):
        ev.append(event(s, next(clock), "session_start", tag))
    for i, (age, truth) in enumerate(spec):
        sim = 0.9 if truth == "present" or i in (7, 19) else 0.2      # two stale captures confirm wrongly
        ev += query_events(clock, "g%d" % i, "s%d" % (1 + i % 4), truth, 0.05 if truth == "present" else (0.8 if truth == "displaced" else None),
                           sim, True, 3, tag, extra_start={"ageMin": age})
    return write_folder(path, ev), spec


def test_age_only_policy_is_a_plain_threshold_on_the_logged_age():
    with tmp() as d:
        folder, spec = age_log(d)
        A, _, _ = analyse([folder])
        P = A.PS["E3"]
        assert P["n"] == 24 and P["n_stale"] == 11 and P["n_age_missing"] == 2      # 8 absent, 1 displaced, 2 replaced
        cv = P["curves"]["Age"]
        assert cv is not None and P["curve_reason"]["Age"] is None
        # Leave-one-session-out folds of 18 queries are too small to train on; the age threshold needs no training.
        assert P["curves"]["All"] is None and P["curves"]["Frame"] is None and "n too small" in P["curve_reason"]["All"]
        stale = [t != "present" for _, t in spec]
        conf = [t == "present" or i in (7, 19) for i, (_, t) in enumerate(spec)]
        # Capture the k oldest; a query without an age comes last, and the two of them together.
        order = sorted(range(22), key=lambda i: -spec[i][0])
        want = []
        for k in list(range(23)) + [24]:
            cap = set(order[:k]) | ({22, 23} if k == 24 else set())
            want.append((k / 24.0, sum(stale[i] and (i not in cap or conf[i]) for i in range(24)) / 24.0,
                         sum((i in cap) and not stale[i] and not conf[i] for i in range(24)) / 24.0))
        got = list(zip(cv["x"].tolist(), cv["wl"].tolist(), cv["lost"].tolist()))
        assert len(got) == len(want) and all(close(a, b, 1e-12) for g, w in zip(got, want) for a, b in zip(g, w)), (got, want)
        assert close(cv["wl"][0], 11 / 24.0) and close(cv["wl"][1], 10 / 24.0)   # the first capture is the oldest memory (110 min, absent)
        # Its AUC is that of the age itself (missing = youngest), pair by pair.
        ages = [a if a is not None else -1.0 for a, _ in spec]
        assert close(cv["auc"], brute_auc(ages, stale), 1e-12) and A.N.get("EThreeAgeAuc") == "%.3f" % cv["auc"]
        # Nothing is fitted: the age-only row exists even where a learned row cannot be trained, and the
        # ablation table has the four rows the spec names, in order.
        tables = PF.build_tables(A.N, A.D, A.E1, A.E2, A.PS, A.C, [])
        body = [ln for ln in tables["table_ablation_E3.tex"].splitlines() if "&" in ln][1:]
        assert [ln.split("&")[0].strip() for ln in body] == ["Age only (threshold, not fitted)", "World signals only", "Frame signals only", "All signals"]
        assert body[0].split("&")[1].strip() == "%.3f" % cv["auc"]
        assert "Frame signals, point chosen on dev" in tables["table_policies_E3.tex"]
        # The figure draws the age curve and says in the legend which curves it could not draw.
        plt = PF.setup_matplotlib()
        fig = PF.fig_policy(plt, P)
        legend = [t.get_text() for t in fig.legends[0].get_texts()]
        assert "Age-only threshold" in legend and "All signals (learned): n too small" in legend and "Frame signals only (learned): n too small" in legend
        assert "Frame signals, point chosen on dev" not in legend     # no dev split here, so no such point
        drawn = [ln for ln in fig.axes[0].lines if ln.get_label() == "Age-only threshold"]
        assert len(drawn) == 1 and np.allclose(drawn[0].get_ydata(), 100 * cv["wl"]) and np.allclose(drawn[0].get_xdata(), cv["x"])
        plt.close(fig)
        assert "On-device rule (logged)" in tables["table_policies_E3.tex"] and "heuristic" not in tables["table_policies_E3.tex"].lower()
        confirm = tables["table_confirm_E3.tex"]
        assert all(word in confirm for word in ("Item at the marker", "Item not at the marker", "absent", "displaced", "replaced", "marker error over 30 cm"))

        # A log with no age at all: no age-only curve, and the reason is said.
        clock, ev = counter(), [event("s%d" % i, i, "session_start", "home=H;exp=E3") for i in range(1, 5)]
        for i in range(24):
            ev += query_events(clock, "n%d" % i, "s%d" % (1 + i % 4), "present" if i % 2 else "absent", 0.05 if i % 2 else None,
                               0.9 if i % 2 else 0.2, True, 3, "home=H;exp=E3", extra_start={"ageMin": None})
        A, _, _ = analyse([write_folder(os.path.join(d, "noage"), ev)])
        assert A.PS["E3"]["curves"]["Age"] is None and "not logged" in A.PS["E3"]["curve_reason"]["Age"]
        assert A.N.get("EThreeAgeAuc") == PF.TBD


def test_cross_room_queries_share_and_time_to_near():
    with tmp() as d:
        clock, ev = counter(), []
        t1, t3, t2 = "home=H;exp=E1;split=test", "home=H;exp=E3;split=test", "home=H;exp=E2;split=test"
        ev += [event("s1", next(clock), "session_start", t1, "FREE"), event("s3", next(clock), "session_start", t3),
               event("s2", next(clock), "session_start", t2, "BENCHMARK")]

        def q(qid, session, tag, user, item, reached, ms, truth="present", mode="PAIRED", **end):
            extra_end = dict({"nearReached": reached, "timeToNearMs": ms}, **end)
            if reached is None:
                del extra_end["nearReached"]
            return query_events(clock, qid, session, truth, 0.05 if truth == "present" else None, None, False, 0, tag, mode,
                                extra_start={"userRoom": user, "itemRoom": item}, extra_end=extra_end)
        ev += q("c1", "s1", t1, "r1", "r2", True, 10000, mode="FREE")           # E1, other room, reached in 10 s
        ev += q("c2", "s1", t1, "r1", "r2", True, 20000, mode="FREE")
        ev += q("c3", "s3", t3, "r1", "r2", True, 30000)                        # E3
        ev += q("c4", "s3", t3, "r2", "r1", False, None, truth="absent")        # never reached NEAR
        ev += q("c5", "s3", t3, "r1", "r3", True, 40000, truth="displaced")
        ev += q("c6", "s3", t3, "r1", "r2", True, 50000, truth="replaced")
        ev += q("c7", "s3", t3, "r1", "r2", True, None)                         # reached, time not logged: in the share, not in the median
        ev += q("c8", "s3", t3, "r1", "r2", None, None)                         # nearReached not logged: counted, not judged
        ev += q("x1", "s3", t3, "r1", "r1", True, 1000)                         # same room
        ev += q("x2", "s3", t3, None, "r2", True, 2000)                         # user's room unknown
        ev += q("x3", "s3", t3, "r1", None, False, None)                        # item's room unknown
        ev += q("x4", "s3", t3, "r1", "r2", False, None, truth="skip")          # not usable
        ev += q("x5", "s2", t2, "r1", "r2", False, None, mode="BENCHMARK")      # another experiment
        A, _, _ = analyse([write_folder(d, ev)])
        NAV, N = A.NAV, A.N
        assert (NAV["n_usable"], NAV["n_cross"], NAV["n_same_room"], NAV["n_room_unknown"]) == (11, 8, 1, 2)
        assert (NAV["n_judged"], NAV["n_reached"]) == (7, 6) and sorted(NAV["seconds"]) == [10.0, 20.0, 30.0, 40.0, 50.0]
        assert (NAV["n_left_out"], NAV["n_left_out_not_reached"]) == (1, 1)     # x4: skipped after a walk that never got there
        assert N.get("NavCrossRoomQueries") == "8" and N.get("NavCrossRoomNearRate") == "86\\% (6/7)"
        assert N.get("NavCrossRoomSecondsMedian") == "30.0" and N.get("NavCrossRoomSecondsN") == "5"
        assert N.get("CrossRoomSecondsToNear") == "30.0" and "n too small (6/7)" in N.get("CrossRoomNearPct")
        rows = {r["query"]: r for r in A.queries}
        assert rows["qc1"]["user_room"] == "r1" and rows["qc4"]["near_reached"] is False and rows["qc8"]["near_reached"] is None
        assert rows["qc3"]["time_to_near_ms"] == 30000


# ----------------------------------------------------------------------------- loading and selection

def test_tag_parsing():
    t = L.parse_tag(" home=A ; EXP=e3;round=2;light=evening;split=Dev;;junk ")
    assert t["home"] == "A" and t["exp"] == "e3" and t["round"] == "2" and t["_malformed"] == ["junk"]
    assert L.tag_exp(t) == "E3" and L.tag_split(t) == "dev" and L.tag_home(t) == "A"
    assert L.parse_tag("") == {} and L.parse_tag(None) == {}
    assert L.tag_split(L.parse_tag("split=train")) is None and L.tag_exp({}) is None


def test_tag_in_effect_and_duplicate_folders():
    with tmp() as d:
        ev = [
            event("s", 1, "session_start", ""),                       # starts without a tag
            {"t": 2, "session": "s", "type": "map_fix", "kind": "manual", "jumpM": 0.5, "fixType": "SPOT"},
            {"t": 3, "session": "s", "type": "study_settings", "studyTag": "home=A;exp=E1", "captureMode": "FREE"},
            {"t": 4, "session": "s", "type": "map_fix", "kind": "manual", "jumpM": 0.02, "fixType": "SPOT"},
            {"t": 5, "session": "s", "type": "study_settings", "studyTag": "", "captureMode": "FREE"},
            {"t": 6, "session": "s", "type": "map_fix", "kind": "manual", "jumpM": 0.9, "fixType": "SPOT"},
        ]
        a = write_folder(os.path.join(d, "a"), ev)
        b = write_folder(os.path.join(d, "b"), ev[:4])                # an earlier pull of the same log
        ds = L.Dataset([a, b])
        assert len(ds.events) == 6 and ds.duplicates == 4
        kept = ds.tagged()
        assert [e["t"] for e in kept] == [3, 4]                       # only what was logged under the tag
        assert [f["jump_m"] for f in L.fixes_from(kept)] == [0.02]
        assert ds.tagged(excluded_sessions=["s"]) == []
        assert not ds.any_synthetic


def test_untagged_data_gives_only_tbd():
    with tmp() as d:
        clock = counter()
        ev = [event("s", next(clock), "session_start", "")] + query_events(clock, "A", "s", "absent", None, 0.9, False, 3, "")
        A, _, synthetic = analyse([write_folder(d, ev)])
        assert not synthetic and A.events == []
        values = {name: A.N.get(name) for name in PF.MACROS}
        # The three constants: the 0.30 m rule, and the base rate and tolerance of the what-if model.
        assert values.pop("RPresentCm") == "30" and values.pop("WhatIfBaseRatePct") == "30" and values.pop("WhatIfTolerancePoints") == "5"
        assert set(values.values()) == {PF.TBD}
        assert A.W["state"] == "no data" and A.FR["n_rows"] == 0 and A.NAV["n_cross"] == 0


def test_missing_optional_fields_are_tolerated():
    """A v0.3-shaped log that only gained studyTag: no captureMode, mobility, lineage, embFile..."""
    with tmp() as d:
        ev = [{"t": 1, "session": "s", "type": "session_start", "studyTag": TAG3}]
        for i, (q, _, truth, measured, sim, pv, calls) in enumerate(HB):
            ev.append({"t": 10 * i + 2, "session": "s", "type": "query_start", "studyTag": TAG3, "query": q, "record": q,
                       "ageMin": 1.0, "positionSource": "SESSION", "policyVerify": pv})
            ev.append({"t": 10 * i + 3, "session": "s", "type": "verify", "studyTag": TAG3, "query": q,
                       "scores": [{"kind": "projected", "sim": sim}, {"kind": "detector", "sim": sim - 0.2}], "threshold": 0.75})
            end = {"t": 10 * i + 4, "session": "s", "type": "query_end", "studyTag": TAG3, "query": q, "groundTruth": truth}
            if measured is not None:
                end["markerError"] = {"errorM": measured}
            ev.append(end)
        ev.append({"t": 999, "session": "s", "type": "query_end", "studyTag": TAG3, "query": "zz", "groundTruth": "present"})
        ev.append({"t": 1000, "session": "s", "type": "verify", "studyTag": TAG3})          # no query id at all
        A, _, _ = analyse([write_folder(d, ev)])
        P = A.PS["E3"]
        assert P["n"] == 11 and dict(P["by_mode"]) == {"not logged": 11}
        assert A.stats["queries_without_start"] == 1
        for name in ("Memory", "Always", "Heuristic", "Oracle"):      # same counts; similarity taken from scores[]
            for k in M.OUTCOMES:
                assert P["policies"][name]["counts"][k] == HB_EXPECT[name][k], (name, k)
        assert A.N.get("EThreeAlwaysCalls") == PF.TBD                 # inferenceCalls was not logged


# ----------------------------------------------------------------------------- E1

def handbuilt_e1(path):
    tag = "home=H;exp=E1;split=test"
    clock, ev = counter(), [event("s1", 1, "session_start", tag, "FREE")]
    spec = ([("SESSION", "r1", e) for e in (0.01, 0.02, 0.03, 0.04, 0.06)] +                       # median 3 cm
            [("MAP", "r1", e) for e in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.40)] +   # the fix point's room
            [("MAP", "r2", e) for e in (0.20, 0.22, 0.24, 0.26, 0.28)] +                            # another room
            [("MAP", None, 0.50)])                                                                   # item room unknown
    for i, (source, room, err) in enumerate(spec):
        ev += query_events(clock, "e%d" % i, "s1", "present", err, None, False, 0, tag, "FREE", source=source,
                           extra_start={"itemRoom": room, "itemRoomName": "Room " + room if room else None, "mapId": "m1",
                                        "metersSinceFix": float(i)})
    ev += query_events(clock, "disp", "s1", "displaced", 2.0, None, False, 0, tag, "FREE", source="MAP")   # not a marker error
    for jump, deg, ftype, kind, extra in ((0.02, 0.5, "SPOT", "manual", {}), (0.04, 1.0, "SPOT", "manual", {}),
                                          (0.06, 1.5, "SPOT", "manual", {}), (0.08, 2.0, "SPOT", "manual", {}),
                                          (0.10, 2.5, "SPOT", "manual", {}), (0.01, 0.1, "IMAGE", "refresh", {}),
                                          (9.9, 90.0, "SPOT", "first", {}),                              # not a re-fix
                                          (1.5, 20.0, "IMAGE", "refresh", {"disagreement": True, "accepted": False, "applied": False})):
        ev.append(dict({"t": next(clock), "session": "s1", "type": "map_fix", "kind": kind, "fixType": ftype,
                        "jumpM": jump, "jumpDeg": deg, "applied": True}, **extra))
    m = {"mapId": "m1", "fixPoints": [{"id": "f1", "type": "SPOT", "x": 0.0, "y": 1.0, "z": 0.0}],
         "rooms": [{"id": "r1", "name": "Room r1"}, {"id": "r2", "name": "Room r2"}],
         "nodes": [{"id": 0, "x": 5.0, "y": 1.0, "z": 0.0, "roomId": "r2"},      # too far (5 m)
                   {"id": 1, "x": 1.0, "y": 9.0, "z": 1.0, "roomId": "r1"},      # 1.41 m horizontally: the nearest labelled
                   {"id": 2, "x": 0.1, "y": 1.0, "z": 0.0, "roomId": None}]}     # nearer, but has no room
    return write_folder(path, ev, map_json=m)


def test_e1_groups_and_refix_jumps():
    with tmp() as d:
        folder = handbuilt_e1(d)
        assert L.fix_point_room(json.load(open(os.path.join(folder, "map.json"))), "f1") == "r1"
        A, _, _ = analyse([folder])
        E1, N = A.E1, A.N
        assert {g: len(v) for g, v in E1["groups"].items()} == {"Session": 5, "MapSame": 11, "MapOther": 5, "MapUnknown": 1,
                                                                 "Other": 0, "MapAll": 17}
        assert N.get("EOneQueries") == "23" and N.get("EOneMeasured") == "22"          # the displaced tap is left out
        assert N.get("EOneSessionMedianCm") == "3" and N.get("EOneSessionPNinetyCm") == PF.NSMALL   # 5 < 10 values
        assert N.get("EOneMapSameMedianCm") == "10" and N.get("EOneMapSamePNinetyCm") == "14"
        assert N.get("EOneMapOtherMedianCm") == "24" and N.get("EOneMapAllN") == "17" and N.get("EOneMapRoomUnknownN") == "1"
        assert N.get("EOneJumpSpotN") == "5" and N.get("EOneJumpSpotMedianCm") == "6.0" and N.get("EOneJumpSpotMaxCm") == "10.0"
        assert N.get("EOneJumpSpotMedianDeg") == "1.50" and N.get("EOneJumpImageN") == "1"
        assert N.get("EOneJumpImageMedianCm") == PF.NSMALL and N.get("EOneJumpImageMaxCm") == "1.0"   # the declined 1.5 m is not applied
        assert N.get("EOneDisagreements") == "1" and N.get("EOneDisagreementsAccepted") == "0"
        assert N.get("EOneWalkN") == "17" and float(N.get("EOneWalkSpearman")) > 0   # metres grow with the row index
        assert N.get("EOneRoomNamedRate") == "91\\% (21/23)"       # 5 + 11 + 5 carry a room name; the unknown-room and displaced rows do not
        # Room-naming accuracy is not measured (spec v0.4 dropped it): no reader, no macro.
        assert "EOneRoomCorrectRate" not in PF.MACROS and "RoomCorrectPct" not in PF.MACROS
        assert all("room_correct" not in q for q in A.queries)
        assert N.get("MarkerErrMapMedianCm") == N.get("EOneMapAllMedianCm") and N.get("RelockJumpMedianCm") == "6.0"


# ----------------------------------------------------------------------------- E2

def handbuilt_e2(path, with_split=True):
    clock, ev = counter(), []
    k = [0]

    def cap(session, split, label, cond, sim):
        k[0] += 1
        tag = "home=H;exp=E2" + (";split=" + split if with_split else "")
        return query_events(clock, "b%d" % k[0], session, None, None, sim, False, 3, tag, "BENCHMARK", end=False,
                            condition_label=(label, cond))
    for i in range(10):                                               # dev: perfectly separable
        ev += cap("d1", "dev", "target", "default", 0.70 + 0.02 * i)
        ev += cap("d1", "dev", "empty", "default", 0.10 + 0.05 * i)
    for i in range(12):                                               # test targets: 9 of 12 reach 0.70
        cond = "default" if i < 6 else "dim"
        ev += cap("t1", "test", "target", cond, 0.69 if i in (0, 6, 7) else 0.71)
    for i in range(10):
        ev += cap("t1", "test", "empty", "default", 0.20)
        ev += cap("t2", "test", "other", "default", 0.72 if i < 2 else 0.30)
        ev += cap("t2", "test", "lookalike", "default", 0.80 if i < 5 else 0.50)
    ev += cap("t2", "test", None, "default", 0.99)                    # dismissed: no label
    return write_folder(path, ev)


def test_e2_threshold_from_dev_and_numbers_from_test():
    with tmp() as d:
        A, _, _ = analyse([handbuilt_e2(d)])
        E2, N = A.E2, A.N
        assert "dev split" in E2["protocol"] and E2["n_dev"] == 20 and len(E2["ev"]) == 42 and E2["n_unlabelled"] == 1
        assert N.get("ETwoThreshold") == "0.70"                       # J = 1 on (0.55, 0.70]; the highest wins
        assert "E2 dev split" in N.get("ETwoThresholdSource")
        assert N.get("ETwoThrTprRate") == "75\\% (9/12)"
        assert N.get("ETwoThrFprEmptyRate") == "0\\% (0/10)" and N.get("ETwoThrFprOtherRate") == "20\\% (2/10)"
        assert N.get("ETwoThrFprLookalikeRate") == "50\\% (5/10)" and N.get("ETwoThrFprAllRate") == "23\\% (7/30)"
        assert N.get("ETwoAucEmpty") == "1.000" and N.get("ETwoAucOther") == "0.800" and N.get("ETwoAucLookalike") == "0.500"
        assert N.get("ETwoAucAll") == "0.767"                         # (120 + 96 + 60) / 360
        assert N.get("ETwoAucAllCI") == PF.NSMALL and N.get("ETwoClusters") == "2"     # two test sessions
        assert N.get("ETwoTprDefaultRate") == "83\\% (5/6)" and N.get("ETwoTprDimRate") == "67\\% (4/6)"
        assert N.get("ETwoTprOccludedRate") == PF.TBD
        assert N.get("ETwoMargin") == PF.TBD and N.get("ETwoMarTprRate") == PF.TBD      # no embedding files
        assert dict(E2["margin_missing"]) == {"no embedding file": 62}
        # The AUC quoted for all labelled captures uses both splits; check against brute force.
        lab = [c for c in A.captures if c["label"] in L.LABELS]
        pos = [c["sim"] for c in lab if c["label"] == "target"]
        neg = [c["sim"] for c in lab if c["label"] != "target"]
        brute = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
        assert close(A.N.raw["ETwoAucAllLabelled"], brute) and N.get("NBenchCaptures") == "62"
        assert "n too small (9/12)" in N.get("VerifyTPRPct") and N.get("VerifyFPRPct") == "23"
        # By item there are enough clusters for an interval.
        B, _, _ = analyse([handbuilt_e2(os.path.join(d, "again"))], e2_cluster="item")
        assert B.N.get("ETwoAucAllCI") != PF.NSMALL and B.N.get("ETwoClusterUnit") == "item"


def test_e2_without_a_split_tunes_nothing():
    with tmp() as d:
        A, _, _ = analyse([handbuilt_e2(d, with_split=False)])
        assert "no dev/test split" in A.E2["protocol"] and len(A.E2["ev"]) == 62 and A.E2["n_dev"] == 0
        assert A.N.get("ETwoThreshold") == "0.75" and "on-device" in A.N.get("ETwoThresholdSource")
        assert A.N.get("ETwoThrTprRate") == "32\\% (7/22)"            # dev targets 0.76..0.88 only
        B, _, _ = analyse([handbuilt_e2(os.path.join(d, "b"), with_split=False)], threshold=0.5)
        assert B.N.get("ETwoThreshold") == "0.50" and "command line" in B.N.get("ETwoThresholdSource")


def test_replay_uses_the_e2_dev_threshold_when_there_is_one():
    with tmp() as d:
        A, _, _ = analyse([handbuilt_e3(os.path.join(d, "e3")), handbuilt_e2(os.path.join(d, "e2"))])
        N = A.N
        assert N.get("EThreeThreshold") == "0.70" and "E2 dev split" in N.get("EThreeThresholdSource")
        # At 0.70 capture J (0.74) confirms as well: always verify now gives 5 wrong leads, 3 warnings.
        c = A.PS["E3"]["policies"]["Always"]["counts"]
        assert (c["wrong_lead"], c["correct_warning"], c["correct_lead"], c["lost_retrieval"]) == (5, 3, 2, 1)
        assert N.get("EThreeAlwaysWrongLeadRate") == "45\\% (5/11)"


def b64(v):
    return base64.b64encode(np.asarray(v, "<f4").tobytes()).decode()


def test_record_history_as_of_capture_time():
    rec = lambda rid, emb, **kw: dict({"id": rid, "name": rid, "embedding": emb, "lastSeenMs": 0}, **kw)  # noqa: E731
    r1, r2, r3 = rec("R1", [1, 0, 0]), rec("R2", [0, 1, 0]), rec("R3", [0, 0, 1])
    r4 = rec("R4", [1, 0, 0], supersedes="R1", lineageId="R1", moveCount=1)
    log = [{"t": 100, "op": "add", "record": r1}, {"t": 200, "op": "add", "record": r2}, {"t": 300, "op": "add", "record": r3},
           {"t": 400, "op": "delete", "record": r2}, {"t": 600, "op": "rename", "record": dict(r3, name="renamed")},
           {"t": 700, "op": "frobnicate", "record": r3}]
    shapes = {"new record names the old": [{"t": 500, "op": "supersede", "record": r4}],
              "old leaves, new is added": [{"t": 500, "op": "supersede", "record": r1}, {"t": 500, "op": "add", "record": r4}]}
    for shape, sup in shapes.items():
        with tmp() as d:
            folder = write_folder(d, [], records=[r3, r4, rec("R0", [0, 0, 1], lastSeenMs=50)], records_log=log + sup)
            h = L.RecordHistory(L.Folder(folder))
            assert h.source == "records_log.jsonl"
            # R0 is in records.json but never in the log: active from its lastSeenMs.
            assert sorted(h.active_at(10)) == [] and sorted(h.active_at(60)) == ["R0"], shape
            assert sorted(h.active_at(250)) == ["R0", "R1", "R2"], shape
            assert sorted(h.active_at(350)) == ["R0", "R1", "R2", "R3"], shape
            assert sorted(h.active_at(450)) == ["R0", "R1", "R3"], shape            # R2 deleted
            assert sorted(h.active_at(550)) == ["R0", "R3", "R4"], shape            # R1 superseded by R4
            assert h.active_at(650)["R3"]["name"] == "renamed" and h.active_at(550)["R3"]["name"] == "R3"
            h.active_at(800)
            assert h.unknown_ops >= 1
    with tmp() as d:                                                  # fallback: records.json only
        folder = write_folder(d, [], records=[rec("R3", [0, 0, 1], lastSeenMs=300), rec("R4", [1, 0, 0], lastSeenMs=500)])
        h = L.RecordHistory(L.Folder(folder))
        assert h.source == "records.json" and sorted(h.active_at(350)) == ["R3"] and sorted(h.active_at(999)) == ["R3", "R4"]
    with tmp() as d:
        assert L.RecordHistory(L.Folder(write_folder(d, []))).source == "none"


def test_margin_uses_the_best_other_record_at_capture_time():
    rec = lambda rid, emb, **kw: dict({"id": rid, "name": rid, "embedding": emb, "lastSeenMs": 0}, **kw)  # noqa: E731
    r1, r2, r3 = rec("R1", [1, 0, 0]), rec("R2", [0, 1, 0]), rec("R3", [0, 0, 1])
    r1b = rec("R1b", [0.6, 0.8, 0], supersedes="R1", lineageId="R1")  # the same item, moved: never an "other"
    log = [{"t": 100, "op": "add", "record": r1}, {"t": 200, "op": "add", "record": r2}, {"t": 300, "op": "add", "record": r3},
           {"t": 400, "op": "delete", "record": r2}]
    emb = {"model": "m", "dim": 3, "record": "R1", "candidates": [
        {"kind": "projected", "rect": [0, 0, 1, 1], "sim": 0.0, "embedding": b64([0, 0, 1])},
        {"kind": "detector", "rect": [0, 0, 1, 1], "sim": 0.8, "embedding": b64([0.8, 0.6, 0])}]}
    alias = {"modelName": "m", "dimension": 3, "target": "R1", "candidates": [
        {"kind": "detector", "rectangle": [0, 0, 1, 1], "similarity": 0.8, "embedding": [0.8, 0.6, 0]}]}   # a JSON list
    with tmp() as d:
        folder = write_folder(d, [{"t": 1, "session": "s", "type": "session_start"}], records=[r1, r3], records_log=log,
                              files={"verify/q_1_emb.json": emb, "verify/q_2_emb.json": alias, "verify/bad_emb.json": {"dim": 3, "candidates": [{"embedding": "AAAA"}]}})
        ds = L.Dataset([folder])
        cap = lambda t, f="verify/q_1_emb.json": {"session": "s", "query": "q", "attempt": 1, "record": "R1", "folder": 0, "t": t, "emb_file": f}  # noqa: E731
        a = L.margin_inputs(ds, cap(350))                             # R2 and R3 are active: R2 is the closest other
        assert a["ok"] and close(a["best_sim"], 0.8, 1e-6) and close(a["other_sim"], 0.6, 1e-6) and a["other_record"] == "R2"
        assert a["n_others"] == 2 and a["source"] == "records_log.jsonl" and a["logged_diff"] < 1e-6
        b = L.margin_inputs(ds, cap(450))                             # after R2 was deleted only R3 is left
        assert close(b["other_sim"], 0.0, 1e-6) and b["other_record"] == "R3" and b["n_others"] == 1
        c = L.margin_inputs(ds, cap(150))                             # before any other record existed
        assert c["ok"] and c["other_sim"] is None and c["n_others"] == 0
        assert close(L.margin_inputs(ds, cap(350, "verify/q_2_emb.json"))["other_sim"], 0.6, 1e-6)   # alias keys, list embedding
        assert L.margin_inputs(ds, cap(350, None))["ok"]              # found by the <query>_<attempt> convention
        assert L.margin_inputs(ds, dict(cap(350, None), query="zz"))["reason"] == "no embedding file"
        assert L.margin_inputs(ds, cap(350, "verify/bad_emb.json"))["reason"] == "no candidate embedding"
    with tmp() as d:                                                  # the item's own newer record is not an "other"
        folder = write_folder(d, [{"t": 1, "session": "s", "type": "session_start"}], records=[],
                              records_log=log[:1] + [{"t": 150, "op": "add", "record": dict(r1b, supersedes=None)}],
                              files={"verify/q_1_emb.json": emb})
        got = L.margin_inputs(L.Dataset([folder]), {"session": "s", "query": "q", "attempt": 1, "record": "R1", "folder": 0, "t": 200})
        assert got["ok"] and got["n_others"] == 0
    assert L.decode_embedding(b64([1, 2, 3]), 3).tolist() == [1.0, 2.0, 3.0]
    assert L.decode_embedding(b64([1, 2, 3]), 4) is None and L.decode_embedding("AAA=") is None and L.decode_embedding(None) is None


def test_margin_rule_through_the_pipeline():
    """Two look-alike records. A look-alike capture passes the threshold but matches the other
    record better, so rule 2 rejects it and rule 1 does not."""
    u1, u2 = [1.0, 0.0], [0.8, 0.6]
    recs = [{"id": "R1", "name": "one", "embedding": u1, "lastSeenMs": 0}, {"id": "R2", "name": "two", "embedding": u2, "lastSeenMs": 0}]
    log = [{"t": 1, "session": "s0", "op": "add", "record": r} for r in recs]
    clock, ev, files, k = counter(), [], {}, [0]

    def cap(session, split, label, vec):
        k[0] += 1
        qid = "m%d" % k[0]
        sim = L.cosine(np.array(vec, np.float32), np.array(u1, np.float32))
        rel = "verify/q%s_1_emb.json" % qid
        files[rel] = {"model": "m", "dim": 2, "record": "rR1", "candidates": [{"kind": "projected", "rect": [0, 0, 1, 1], "sim": sim, "embedding": b64(vec)}]}
        out = query_events(clock, qid, session, None, None, sim, False, 2, "home=H;exp=E2;split=" + split, "BENCHMARK", end=False,
                           condition_label=(label, "default"), extra_verify={"embFile": rel, "record": "R1"})
        return out
    for split, session in (("dev", "d1"), ("test", "t1")):
        for _ in range(10):
            ev += cap(session, split, "target", [0.8, -0.6])          # 0.80 to R1, 0.28 to R2: margin +0.52
            ev += cap(session, split, "lookalike", [0.8, 0.6])        # 0.80 to R1, 1.00 to R2: margin -0.20
            ev += cap(session, split, "empty", [0.0, -1.0])           # 0.00 to R1
    with tmp() as d:
        A, _, _ = analyse([write_folder(d, ev, records=recs, records_log=log, files=files)], threshold=0.75)
        E2, N = A.E2, A.N
        assert E2["margin"] == 0.0 and E2["sim_mismatch"] == 0 and dict(E2["margin_sources"]) == {"records_log.jsonl": 60}
        assert N.get("ETwoSubTprRate") == "100\\% (10/10)" and N.get("ETwoMarTprRate") == "100\\% (10/10)"
        assert N.get("ETwoSubFprLookalikeRate") == "100\\% (10/10)" and N.get("ETwoMarFprLookalikeRate") == "0\\% (0/10)"
        assert N.get("ETwoMarFprEmptyRate") == "0\\% (0/10)" and N.get("ETwoMarginSubsetN") == "30" and N.get("ETwoMargin") == "0.00"
        r = E2["conditions"]["default"]
        assert r["sub"] == (10, 10) and r["mar"] == (10, 10)


# ----------------------------------------------------------------------------- T2

def test_cost_latency_and_energy_per_verification():
    with tmp() as d:
        tag = "home=H;exp=E4"
        clock, ev = counter(0), [event("s", 1, "session_start", tag, "FREE")]
        for i in range(6):
            ev.append(event("s", next(clock), "register", tag, "FREE", record="r%d" % i, recordBytes=50000 + 1000 * i, cropBytes=1,
                            latency={"convertMs": 50.0 + i, "detectMs": 70.0, "embedMs": 20.0}))
        clock = counter(1_000_000)
        ev += query_events(clock, "p", "s", None, None, 0.9, False, 4, tag, "FREE", end=False)       # t = 1 001 000 / 1 002 000
        for t in (1_010_000, 1_020_000, 1_030_000, 1_040_000):
            ev.append(event("s", t, "verify", tag, "FREE", query="qp", attempt=t, bestSimilarity=0.9, inferenceCalls=4,
                            latency={"convertMs": 50.0, "detectMs": 80.0, "embedMs": 30.0, "totalMs": 170.0}))
        ev.append(event("o", 1_015_000, "verify", "home=H;exp=E3", "PAIRED", query="other", attempt=1, bestSimilarity=0.9))   # another experiment
        folder = write_folder(os.path.join(d, "data"), ev, map_json={"mapId": "m", "nodes": [], "fixPoints": [], "rooms": []})
        pdir = os.path.join(d, "power")
        os.makedirs(pdir)

        def run(name, mode, mw, seconds, start=None, fg="edu.cs6301.findback/.MainActivity", verifications=None):
            doc = {"tool": "power_measure.py", "mode": mode, "label": name, "synthetic": False, "duration_s": seconds,
                   "all_rails": {"energy_j": mw * seconds / 1000.0, "mean_mw": mw}, "groups": {"cpu": {"mean_mw": mw / 2}},
                   "phone_start_ms": start, "phone_end_ms": None if start is None else start + seconds * 1000,
                   "verifications": verifications, "state_before": {"foreground": fg}, "state_after": {"foreground": fg}}
            json.dump(doc, open(os.path.join(pdir, name + ".json"), "w"))
        run("t1", "tracking_only", 1000.0, 100)
        run("t2", "tracking_only", 1100.0, 100)
        run("v1", "tracking_verify", 1150.0, 100, start=1_000_000)   # 115 J; five verify events tagged E4 fall inside
        run("x1", "tracking_verify", 9999.0, 100, fg="com.android.launcher/.Home")     # wrong app: left out
        run("x2", "tracking_only", 1.0, 100, fg=None)                                  # screen off, app unknown: left out
        run("o1", "other", 5000.0, 100)
        A, _, _ = analyse([folder], [pdir])
        N, C = A.N, A.C
        assert N.get("CostRegConvertMedianMs") == "52" and N.get("CostRegTotalMedianMs") == "142" and N.get("CostRegN") == "6"
        assert N.get("CostRegTotalPNinetyFiveMs") == PF.NSMALL and N.get("CostVerTotalMedianMs") == "170"
        assert N.get("CostStorageItemKB") == "52" and N.get("BytesPerItem") == "52500"
        assert N.get("CostCaptureSecondsN") == "1" and N.get("CostCaptureSecondsMedian") == PF.NSMALL
        assert C["capture_s"] == [4.17]                               # msSinceNear 4000 ms + latency.totalMs 170 ms
        assert N.get("CostPowerTrackingMw") == "1050" and N.get("CostPowerTrackingRuns") == "2"
        assert N.get("CostPowerTrackingRangeMw") == "1000--1100"
        assert N.get("CostPowerVerifyMw") == "1150" and N.get("CostPowerVerifyRuns") == "1" and N.get("CostMapFiles") == "1"
        assert sum("left out" in w for w in C["warnings"]) == 2
        # Two and one runs: too few to tell the modes apart, so no energy figure.
        assert C["energy_j"] is None and N.get("CostEnergyPerVerificationJ") == PF.NSMALL and N.get("EnergyPerVerifymJ") == PF.NSMALL

        run("t3", "tracking_only", 1050.0, 100)
        run("v2", "tracking_verify", 1150.0, 100, verifications=5)   # no events in its window: the entered count is used
        run("v3", "tracking_verify", 1150.0, 100, verifications=5)
        B, _, _ = analyse([folder], [pdir])
        # Three against three, every verifying run above every tracking run.
        # (115 J - 1.05 W * 100 s) / 5 verifications = 2 J in each run. The E3 capture inside v1's window is not counted.
        assert B.C["energy_n"] == 15 and close(B.C["energy_j"], 2.0) and B.N.get("CostEnergyPerVerificationJ") == "2.00"
        assert B.N.get("EnergyPerVerifymJ") == "2000" and B.N.get("CostPowerVerifyRangeMw") == "1150--1150"

        run("v3", "tracking_verify", 1090.0, 100, verifications=5)   # now one verifying run is below a tracking run
        D, _, _ = analyse([folder], [pdir])
        assert D.C["energy_j"] is None and D.N.get("CostEnergyPerVerificationJ") == PF.UNRESOLVED
        assert any("not resolved" in w for w in D.C["warnings"])


def pb_varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def pb_field(number, value):
    if isinstance(value, int):
        return pb_varint(number << 3) + pb_varint(value)
    return pb_varint(number << 3 | 2) + pb_varint(len(value)) + value


def fake_trace(samples, descriptors=((0, "S1S_VDD_CAM", "Multimedia"), (1, "S4M_VDD_CPUCL0", "CPU(LITTLE)"))):
    packets = [pb_field(6, pb_field(1, pb_field(1, 6) + pb_field(2, 5_000_000_000)) +       # BOOTTIME 5 s
                        pb_field(1, pb_field(1, 1) + pb_field(2, 1_700_000_005_000_000_000)))]   # REALTIME
    for j, row in enumerate(samples):
        rails = b"".join(pb_field(1, pb_field(1, i) + pb_field(2, name.encode()) + pb_field(3, sub.encode()))
                         for i, name, sub in descriptors) if j == 0 else b""
        rails += b"".join(pb_field(2, pb_field(1, i) + pb_field(2, t) + pb_field(3, e)) for i, t, e in row)
        packets.append(pb_field(8, 123) + pb_field(40, rails))
    return b"".join(pb_field(1, p) for p in packets)


def test_power_trace_parser():
    trace = fake_trace([[(0, 10_000, 1_000_000), (1, 10_001, 50_000_000)],
                        [(0, 15_000, 1_600_000), (1, 15_001, 52_000_000)],
                        [(0, 20_000, 2_000_000), (1, 20_001, 56_000_000)]])
    run = PM.summarise(PM.parse_trace(trace))
    rails = {r["name"]: r for r in run["rails"]}
    assert close(rails["S1S_VDD_CAM"]["energy_j"], 1.0) and close(rails["S1S_VDD_CAM"]["mean_mw"], 100.0)   # 1 J in 10 s
    assert close(rails["S4M_VDD_CPUCL0"]["energy_j"], 6.0) and close(rails["S4M_VDD_CPUCL0"]["mean_mw"], 600.0)
    assert rails["S1S_VDD_CAM"]["group"] == "camera" and rails["S4M_VDD_CPUCL0"]["group"] == "cpu"
    assert close(run["all_rails"]["energy_j"], 7.0) and close(run["all_rails"]["mean_mw"], 700.0) and close(run["duration_s"], 10.0)
    assert close(run["phone_start_ms"], 1_700_000_005_000 - 5_000 + 10_000)   # boot-relative samples on the phone's clock
    assert [PM.group_of(n) for n in ("S2S_VDD_G3D", "S7M_VDD_TPU", "VSYS_PWR_DISPLAY", "L22M_DISP", "S4S_VDD2H_MEM", "VSYS_PWR_MODEM")] == \
        ["gpu", "tpu", "display", "display", "memory", "other"]
    try:
        PM.summarise(PM.parse_trace(fake_trace([[(0, 1, 500)], [(0, 2, 400)]])))    # a counter that runs backwards
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        PM.summarise(PM.parse_trace(b""))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- synthetic safeguard

def run_script(name, *args):
    p = subprocess.run([sys.executable, os.path.join(HERE, name)] + list(args), capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def synthetic_set(d, *extra):
    out = os.path.join(d, "synth")
    rc, _, err = run_script("synth_generate.py", "--out", out, "--seed", "7", *extra)
    assert rc == 0, err
    return [os.path.join(out, "site_A"), os.path.join(out, "site_B")], os.path.join(out, "power_runs")


def test_each_synthetic_mark_is_detected_on_its_own():
    with tmp() as d:
        clock = counter()
        base = [event("abc", 1, "session_start", TAG3)] + query_events(clock, "A", "abc", "present", 0.1, 0.9, False, 3, TAG3)
        assert not L.Dataset([write_folder(os.path.join(d, "real"), base)]).any_synthetic
        flagged = [dict(e, synthetic=True) if i == 2 else e for i, e in enumerate(base)]
        renamed = [dict(e, session="SYNTH001") for e in base]
        tagged = [dict(e, studyTag="synthetic=1;" + TAG3) if "studyTag" in e else e for e in base]
        cases = {"event field": write_folder(os.path.join(d, "c1"), flagged),
                 "session id": write_folder(os.path.join(d, "c2"), renamed),
                 "study tag": write_folder(os.path.join(d, "c3"), tagged),
                 "records": write_folder(os.path.join(d, "c4"), base, records=[{"id": "x", "synthetic": True}]),
                 "record log": write_folder(os.path.join(d, "c5"), base, records_log=[{"t": 1, "op": "add", "record": {"id": "x", "synthetic": True}}])}
        marked = write_folder(os.path.join(d, "c6"), base)
        open(os.path.join(marked, L.SYNTHETIC_MARKER_FILE), "w").write("x")
        cases["marker file"] = marked
        for name, folder in cases.items():
            assert L.Dataset([folder]).any_synthetic, name
            assert set(L.Dataset([folder]).synthetic_sessions) == {"SYNTH001" if name == "session id" else "abc"}, name   # the session is named
            assert L.Dataset([os.path.join(d, "real"), folder]).any_synthetic, name   # one synthetic folder taints the run
        pdir = os.path.join(d, "p")
        os.makedirs(pdir)
        json.dump({"tool": "power_measure.py", "mode": "tracking_only", "synthetic": True, "all_rails": {"mean_mw": 1.0, "energy_j": 1.0}},
                  open(os.path.join(pdir, "r.json"), "w"))
        assert PF.load_inputs([os.path.join(d, "real")], [pdir])[2] is True
        assert PF.load_inputs([os.path.join(d, "real")], [])[2] is False


def test_paper_path_detection():
    assert PF.is_paper_path(os.path.join(PROJECT, "paper"))
    assert PF.is_paper_path(os.path.join(PROJECT, "paper", "generated"))
    assert PF.is_paper_path("/somewhere/else/Paper/figures")
    assert not PF.is_paper_path(os.path.join(tempfile.gettempdir(), "findback_out"))
    assert not PF.is_paper_path(os.path.join(tempfile.gettempdir(), "paperwork"))


def table_cells(text):
    for line in text.splitlines():
        if line.startswith("%") or "&" not in line:
            continue
        for cell in line.rstrip("\\ ").split("&"):
            yield cell.strip()


def test_synthetic_output_is_refused_under_paper_and_marked_everywhere():
    with tmp() as d:
        folders, power = synthetic_set(d, "--items", "8", "--rounds", "2", "--days", "1")
        paper = os.path.join(d, "x", "paper")
        rc, out, err = run_script("paper_figures.py", *folders, "--power-dir", power, "--out", paper)
        assert rc == 2 and "REFUSED" in err and out == "" and not os.path.exists(paper)
        rc, _, err = run_script("paper_figures.py", *folders, "--out", os.path.join(d, "ok"), "--debug-json", os.path.join(paper, "v.json"))
        assert rc == 2 and not os.path.exists(os.path.join(d, "ok"))
        # Only the power runs are synthetic: still refused.
        real = handbuilt_e3(os.path.join(d, "real"))
        rc, _, err = run_script("paper_figures.py", real, "--power-dir", power, "--out", paper)
        assert rc == 2 and "synthetic power run" in err and not os.path.exists(paper)

        for out_dir, flags in ((paper, ["--allow-synthetic"]), (os.path.join(d, "scratch_out"), [])):
            rc, out, err = run_script("paper_figures.py", *folders, "--power-dir", power, "--out", out_dir, *flags)
            assert rc == 0, err
            results = open(os.path.join(out_dir, "generated", "results.tex")).read()
            macros = re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}% ", results, re.M)
            assert [m[0] for m in macros] == list(PF.MACROS)
            assert {m[1] for m in macros} == {PF.SYNTH}                # every macro, constants included
            tables = sorted(glob.glob(os.path.join(out_dir, "generated", "table_*.tex")))
            assert len(tables) == 12
            for path in tables:
                text = open(path).read()
                assert text.startswith("% SYNTHETIC DATA")
                for cell in table_cells(text):
                    numeric = re.fullmatch(r"[-+]?[\d.,]+\s*(\\%)?(\s*\(\d+/\d+\))?", cell)
                    assert not numeric or cell in ("0.25", "0.50"), (os.path.basename(path), cell)   # the two fixed budgets
                assert not re.search(r"\d\s*\\%", text.replace("95\\% CI", "")), os.path.basename(path)   # no percentage, labels included
            for name in ("table_dataset.tex", "table_cost.tex"):
                assert "\\caption{\\textcolor{red}{SYNTHETIC DATA.}" in open(os.path.join(out_dir, "generated", name)).read()
            pdfs = sorted(glob.glob(os.path.join(out_dir, "figures", "*.pdf")))
            assert len(pdfs) == 7 and os.path.basename(pdfs[-1]) == "F4_what_if.pdf"
            for path in pdfs:
                assert b"SYNTHETIC DATA" in open(path, "rb").read()   # the PDF keywords
            prov = open(os.path.join(out_dir, "generated", "provenance.txt")).read()
            assert prov.startswith("SYNTHETIC DATA") and "==" not in prov       # no report of a synthetic run is kept


def test_synthetic_figures_carry_the_watermark():
    with tmp() as d:
        folders, power = synthetic_set(d, "--items", "8", "--rounds", "2", "--days", "1")
        A, ds, synthetic = analyse(folders, [power])
        assert synthetic and len(ds.synthetic_sessions) == len(ds.sessions())
        assert all(A.N.get(name) == PF.SYNTH for name in PF.MACROS)
        plt = PF.setup_matplotlib()
        figs = PF.build_figures(plt, A.E1, A.E2, A.PS, A.W, True)
        for name, fig in figs.items():
            marks = [t for t in fig.findobj(lambda o: getattr(o, "get_gid", lambda: None)() == "fb-synthetic")]
            assert len(marks) == len(fig.axes) + 1, name               # one across each axes, one banner
            assert all("SYNTHETIC" in m.get_text() for m in marks)
            plt.close(fig)
        clean = PF.build_figures(plt, A.E1, A.E2, A.PS, A.W, False)
        for fig in clean.values():
            assert not fig.findobj(lambda o: getattr(o, "get_gid", lambda: None)() == "fb-synthetic")
            plt.close(fig)


def test_synthetic_generator_refuses_real_locations():
    with tmp() as d:
        for bad in (os.path.join(PROJECT, "paper", "x"), os.path.join(PROJECT, "findback_data"), os.path.join(d, "backups", "y")):
            rc, _, err = run_script("synth_generate.py", "--out", bad)
            assert rc == 2 and "refusing" in err, bad
        real = handbuilt_e3(os.path.join(d, "looks_real"))
        before = open(os.path.join(real, "events.jsonl")).read()
        rc, _, err = run_script("synth_generate.py", "--out", real)   # holds an events.jsonl without the marker
        assert rc == 2 and open(os.path.join(real, "events.jsonl")).read() == before


def test_synthetic_variants_run_end_to_end():
    """Every fallback path: v0.3-shaped events with only a tag, no split, no record log, no embeddings; a tiny set."""
    with tmp() as d:
        for i, extra in enumerate((["--variant", "minimal", "--sites", "1"], ["--items", "4", "--rounds", "1", "--days", "1"])):
            out = os.path.join(d, "s%d" % i)
            rc, _, err = run_script("synth_generate.py", "--out", out, *extra)
            assert rc == 0, err
            folders = sorted(glob.glob(os.path.join(out, "site_*")))
            rc, report, err = run_script("paper_figures.py", *folders, "--out", os.path.join(d, "o%d" % i))
            assert rc == 0, err
            assert report.startswith("*** SYNTHETIC INPUT")
            if i == 0:
                assert "records.json only" in report and "leave-one-session-out" in report and "no dev/test split" in report
            else:
                assert "n too small" in report


# ----------------------------------------------------------------------------- macros and files

def test_macro_catalogue_and_results_file():
    names = list(PF.MACROS)
    assert len(names) == len(set(names)) and all(n.isalpha() for n in names)
    with tmp() as d:
        rc, out, err = run_script("paper_figures.py", handbuilt_e3(os.path.join(d, "in")), "--out", os.path.join(d, "out"))
        assert rc == 0, err
        assert "leave-one-session-out" in out and "SYNTHETIC" not in out
        text = open(os.path.join(d, "out", "generated", "results.tex")).read()
        defined = re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}", text, re.M)
        assert defined == names                                        # every macro, once, whatever the data
        assert "\\newcommand{\\EThreeMemoryWrongLeadRate}{73\\% (8/11)}" in text
        assert "\\newcommand{\\EFiveQueries}{\\textcolor{red}{TBD}}" in text
        assert "SYNTHETIC" not in text
        assert "leave-one-session-out" in open(os.path.join(d, "out", "generated", "provenance.txt")).read()
        if shutil.which("pdflatex"):                                   # the generated TeX compiles
            tables = sorted(glob.glob(os.path.join(d, "out", "generated", "table_*.tex")))
            body = "\n".join(("\\input{%s}" if os.path.basename(t) in ("table_dataset.tex", "table_cost.tex")
                              else "\\begin{table}\\tiny\\input{%s}\\end{table}") % t for t in tables)
            uses = " ".join("\\%s{}" % n for n in names)
            doc = ("\\documentclass{article}\\usepackage{xcolor,booktabs}\\usepackage[margin=1cm,landscape]{geometry}\n"
                   "\\input{%s}\n\\begin{document}\n%s\n%s\n\\end{document}\n") % (os.path.join(d, "out", "generated", "results.tex"), body, uses)
            open(os.path.join(d, "t.tex"), "w").write(doc)
            p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"], cwd=d, capture_output=True, text=True)
            assert p.returncode == 0, p.stdout[-1500:]


def tex_code(path):
    """A TeX file without its comments."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in open(path).read().splitlines())


def test_paper_macro_names_are_all_defined():
    """Every result macro the paper uses is one results.tex defines, and nothing else defines it."""
    paper = os.path.join(PROJECT, "paper", "findback.tex")
    fallback = os.path.join(PROJECT, "paper", "macros_fallback.tex")
    if not (os.path.exists(paper) and os.path.exists(fallback)):
        return "skipped: no paper/findback.tex or paper/macros_fallback.tex"
    body, fall = tex_code(paper), tex_code(fallback)
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    fallbacks = re.findall(r"\\providecommand\{\\([A-Za-z]+)\}", fall)
    app_constants = {"FixAgreeSamples", "FixAgreeWindowS"}            # copied from the app's source, not measured
    # 1. A fallback the paper uses must be a macro of results.tex; otherwise it would print TBD for ever.
    missing = sorted(w for w in fallbacks if w in used and w not in PF.MACROS and w not in app_constants)
    assert not missing, missing
    # 2. A results.tex macro the paper uses must have a fallback, so the paper compiles without results.tex.
    no_fallback = sorted(w for w in used if w in PF.MACROS and w not in fallbacks)
    assert not no_fallback, no_fallback
    # 3. Nothing the paper uses looks like a result macro without being one: a name the paper uses as
    #    \Name{} is defined by results.tex, by the paper itself, or is an app constant.
    own = set(re.findall(r"\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)", body))
    unknown = sorted(w for w in set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", body)) if w not in PF.MACROS and w not in own
                     and w not in app_constants and w not in ("TeX", "LaTeX"))
    assert not unknown, unknown
    # 4. Every generated file the paper reads is one the pipeline writes.
    with tmp() as d:
        A, _, _ = analyse([write_folder(d, [event("s", 1, "session_start", "")])])
        tables = PF.build_tables(A.N, A.D, A.E1, A.E2, A.PS, A.C, [])
        figures = PF.build_figures(PF.setup_matplotlib(), A.E1, A.E2, A.PS, A.W, False)
    for name in re.findall(r"generated/([A-Za-z0-9_]+\.tex)", body):
        assert name == "results.tex" or name in tables, name
    for name in re.findall(r"figures/([A-Za-z0-9_]+)(?:\.pdf)?\}", body):
        assert name + ".pdf" in figures, name
    m = re.search(r"\\providecommand\{\\RPresentCm\}\{([\d.]+)\}", fall)
    assert m and float(m.group(1)) == M.R_PRESENT_M * 100
    stale = sorted(w for w in fallbacks if w not in used and w not in PF.MACROS and w not in app_constants)
    n_used = len([w for w in used if w in PF.MACROS])
    return "%d result macros used by the paper%s" % (n_used, "; unused fallbacks that results.tex no longer defines: " + ", ".join(stale) if stale else "")


def test_figures_for_small_and_empty_data():
    plt = PF.setup_matplotlib()
    with tmp() as d:
        for folder, want in ((handbuilt_e3(os.path.join(d, "a")), "n too small (n = 11 queries)"),
                             (write_folder(os.path.join(d, "b"), [event("s", 1, "session_start", "")]), "TBD: no paired E3 queries yet")):
            A, _, _ = analyse([folder])
            figs = PF.build_figures(plt, A.E1, A.E2, A.PS, A.W, False)
            assert list(figs) == ["F1_marker_error_cdf.pdf", "F1_error_vs_walked.pdf", "F2_roc.pdf", "F2_conditions.pdf",
                                  "F3_policy_E3.pdf", "F3_policy_E5.pdf", "F4_what_if.pdf"]
            w_notes = [t.get_text() for t in figs["F4_what_if.pdf"].findobj(lambda o: getattr(o, "get_gid", lambda: None)() == "fb-note")]
            assert w_notes == [("n too small (3 captures at the marker, 8 not; 20 of each needed)" if "11" in want
                                else "TBD: no paired E3 queries yet")], w_notes
            assert not figs["F4_what_if.pdf"].axes[0].lines            # no model curve without its measured inputs
            assert "model" in figs["F4_what_if.pdf"].axes[0].get_xlabel() and "model" in figs["F4_what_if.pdf"].axes[0].get_ylabel()
            notes = [t.get_text() for t in figs["F3_policy_E3.pdf"].findobj(lambda o: getattr(o, "get_gid", lambda: None)() == "fb-note")]
            assert notes == [want, want], notes
            assert not figs["F3_policy_E3.pdf"].axes[0].lines          # no curve from eleven points
            for fig in figs.values():
                assert fig.get_size_inches()[0] == PF.COL_W
                assert all(not ax.get_title() for ax in fig.axes)      # no titles inside a figure
                assert all(ax.get_xlabel() or ax.get_ylabel() for ax in fig.axes)
                plt.close(fig)
    assert PF.POLICY_STYLE["Memory"]["color"] != PF.POLICY_STYLE["Always"]["color"]
    assert len({(s["color"], s.get("marker"), s.get("ls")) for s in PF.POLICY_STYLE.values()}) == len(PF.POLICY_STYLE)


# ----------------------------------------------------------------------------- evaluate.py

def run_evaluate(*args):
    p = subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py")] + list(args), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


def test_evaluate_counts_displaced_and_replaced_as_not_present():
    with tmp() as d:
        out = run_evaluate(handbuilt_e3(d), "--exclude-session", "u0", "--exclude-session", "fr", "--csv", os.path.join(d, "q.csv"))
        assert "== Queries with ground truth: 13 (present 6, absent 3, displaced 2, replaced 2) ==" in out
        assert "Paired (have a fresh capture): 11; without capture: 2" in out
        assert "Policy said VERIFY on 7/11 paired queries" in out
        rows = {l.split()[0]: l.split()[1:] for l in out.splitlines() if l.split() and l.split()[0] in ("memory_only", "always_verify", "selective")}
        # evaluate.py keeps treating every `present` as present (D and E included), so these differ from HB_EXPECT.
        assert rows["memory_only"] == ["11", "5", "6", "0", "0", "0.00", "0.00"]
        assert rows["always_verify"] == ["11", "3", "3", "2", "3", "1.00", "3.55"]
        assert rows["selective"] == ["11", "3", "3", "2", "3", "0.64", "2.45"]
        assert "== Marker error (n=5): median 30 cm, max 50 cm ==" in out          # A C D E M; the displaced taps are not in it
        assert "== Displacement of displaced items (n=2): median 50 cm, max 80 cm ==" in out
        assert "NOTE: 2 'present' queries have a measured marker error above 30 cm." in out
        assert "tau 0.75   TPR  0.60   FPR  0.50" in out                           # 3 of 5 present, 3 of 6 not present
        table = list(csv.DictReader(open(os.path.join(d, "q.csv"))))
        assert [r["truth"] for r in table if r["displacement_m"]] == ["displaced", "displaced"]
        assert all(not r["marker_error_m"] for r in table if r["truth"] != "present")


def test_evaluate_output_on_pilot_data_starts_with_the_saved_text():
    base = os.path.join(HERE, "testdata", "evaluate_baseline")
    cases = {"findback_data__plain.txt": ["findback_data"],
             "findback_data__excluded.txt": ["findback_data", "--exclude-session", "cab39b87", "--exclude-session", "1debdfe5"],
             "backup__plain.txt": ["backups/phone_files_2026-11-16"],
             "backup__excluded.txt": ["backups/phone_files_2026-11-16", "--exclude-session", "cab39b87", "--exclude-session", "1debdfe5"],
             "backup__dev_session.txt": ["backups/phone_files_2026-11-16", "--dev-session", "ff5fbb39", "--exclude-session", "cab39b87",
                                         "--exclude-session", "1debdfe5"]}
    checked = 0
    for name, args in cases.items():
        folder = os.path.join(PROJECT, args[0])
        if not os.path.isdir(folder) or not os.path.exists(os.path.join(base, name)):
            continue
        saved = open(os.path.join(base, name)).read()
        now = run_evaluate(folder, *args[1:])
        assert now.startswith(saved), name
        assert now == saved or now[len(saved):].startswith("\nNOTE: "), name       # nothing but the note is added
        checked += 1
    if not checked:
        return "skipped: pilot data or saved baselines not found"
    assert M.R_PRESENT_M == 0.30 and "R_PRESENT_M = 0.30" in open(os.path.join(HERE, "evaluate.py")).read()


# ----------------------------------------------------------------------------- trial sheet

def test_trial_sheet_mix_and_determinism():
    ids = ["id%02d" % i for i in range(20)]
    actions, order, loops = TS.draw(ids, 1, 7)
    count = lambda a: sum(v == a for v in actions.values())  # noqa: E731
    assert count("leave") == 10 and count("remove") == 4 and count("move") == 3 and count("replace") + count("swap") == 3
    assert sorted(order) == ids and len(loops) == 4 and loops <= set(ids)
    assert TS.draw(list(reversed(ids)), 1, 7) == (actions, order, loops)          # the order of the file does not matter
    assert TS.draw(ids, 1, 8)[0] != actions and TS.draw(ids, 2, 7)[0] != actions   # seed and round both matter
    odd = TS.draw(ids[:7], 1, 7)[0]                                                # 3.5 / 1.4 / 1.05 / 1.05 -> 7 in all
    assert len(odd) == 7 and sum(v == "leave" for v in odd.values()) in (3, 4)
    assert TS.counts_for(20, {"leave": 1, "remove": 0, "move": 0, "replace_or_swap": 0}, TS.stream(1, 1, "c"))["leave"] == 20


def test_trial_sheet_is_independent_of_the_records_and_of_the_order():
    ids = ["id%02d" % i for i in range(20)]
    leave = {i: 0 for i in ids}
    position_sum = {"leave": 0.0, "other": 0.0}
    loop_and_leave = loop_total = loop_position = 0
    reps = 600
    for seed in range(reps):
        actions, order, loops = TS.draw(ids, 3, seed)
        for i in ids:
            leave[i] += actions[i] == "leave"
        for pos, i in enumerate(order):
            position_sum["leave" if actions[i] == "leave" else "other"] += pos
        loop_total += len(loops)
        loop_and_leave += sum(actions[i] == "leave" for i in loops)
        loop_position += sum(pos for pos, i in enumerate(order) if i in loops)
    # Every item is left alone half of the time (binomial sd 0.02).
    assert all(0.42 <= v / reps <= 0.58 for v in leave.values()), leave
    # Left-alone and changed items are queried equally early (mean position 9.5).
    assert abs(position_sum["leave"] / (10 * reps) - 9.5) < 0.3 and abs(position_sum["other"] / (10 * reps) - 9.5) < 0.3
    # A long loop is as likely before a left-alone item as before a changed one.
    assert abs(loop_and_leave / loop_total - 0.5) < 0.04
    # ... and falls anywhere in the query order (mean position 9.5), not at its start or end.
    assert abs(loop_position / loop_total - 9.5) < 0.5
    with tmp() as d:                                                    # other fields of a record change nothing
        plain = [{"id": i, "name": "n" + i} for i in ids]
        rich = [{"id": i, "name": "n" + i, "mobility": "often" if k % 2 else "rare", "moveCount": k, "lastSeenMs": k * 1000,
                 "roomId": "r%d" % (k % 3), "embedding": [k, 1.0]} for k, i in reversed(list(enumerate(ids)))]
        outs = []
        for name, recs in (("plain", plain), ("rich", rich)):
            folder = write_folder(os.path.join(d, name), [], records=recs)
            assert TS.main([folder, "--round", "2", "--seed", "5", "--out-dir", os.path.join(d, name + "_out")]) == 0
            rows = list(csv.DictReader(open(os.path.join(d, name + "_out", "trial_sheet_round2_seed5.csv"))))
            outs.append([(r["order"], r["record"], r["action"], r["long_loop_first"], r["scripted_truth"]) for r in rows])
            md = open(os.path.join(d, name + "_out", "trial_sheet_round2_seed5.md")).read()
            assert "Walk a long loop first" in md and "Put every item back" in md
        assert outs[0] == outs[1] and len(outs[0]) == 20
        truth = {"leave": "present", "remove": "absent", "move": "displaced", "replace": "replaced", "swap": "replaced"}
        assert all(row[4] == truth[row[2]] for row in outs[0])


# ----------------------------------------------------------------------------- runner

def main(argv):
    pattern = argv[argv.index("-k") + 1] if "-k" in argv else ""
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f) and pattern in n]
    failed = 0
    for name, fn in tests:
        try:
            note = fn()
            skipped = bool(note) and note.startswith("skipped")
            print("%-4s %s%s" % ("SKIP" if skipped else "ok", name, ("  (" + note + ")") if note else ""))
        except Exception:  # noqa: BLE001
            import traceback
            failed += 1
            print("FAIL %s" % name)
            traceback.print_exc()
    print("\n%d tests, %d failed" % (len(tests), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
