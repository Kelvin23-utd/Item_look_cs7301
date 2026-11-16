#!/usr/bin/env python3
"""Figures, tables and numbers of the FindBack paper (STUDY_SPEC_v0.4 section 4).

Reads data folders pulled from phones, keeps the events logged under a study tag, and writes

  <out>/figures/F1_marker_error_cdf.pdf  F1_error_vs_walked.pdf      (E1)
                F2_roc.pdf  F2_conditions.pdf                        (E2)
                F3_policy_E3.pdf  F3_policy_E5.pdf                   (E3, E5; measured)
                F4_what_if.pdf                                       (a model fed two measured shares)
  <out>/generated/results.tex      one macro per number, every macro always defined
                  table_dataset.tex  table_cost.tex            T1 and T2 as complete table floats
                  table_dataset_by_experiment.tex  table_power_by_group.tex  table_marker_error.tex
                  table_refix_jumps.tex  table_rules.tex  table_policies_E3.tex  table_policies_E5.tex
                  table_ablation_E3.tex  table_ablation_E5.tex  table_confirm_E3.tex   (tabular only)
                  provenance.txt   what went in, and which protocol each number used

<out> defaults to <project>/paper. A number with no data is a red TBD; a quantity that needs
more data than there is says "n too small"; every rate carries its counts.

Synthetic data. If any input session, record folder or power run is synthetic, everything
written is marked: every macro and table cell expands to a red SYNTHETIC and every figure is
watermarked. Writing such output under a directory named `paper` is refused unless
--allow-synthetic is given.

Usage:
  python3 analysis/paper_figures.py DATA_FOLDER [DATA_FOLDER ...] [--power-dir analysis/power_runs]
  python3 analysis/paper_figures.py findback_data backups/phone_files_2026-11-16
Options are listed by --help. Definitions and choices are documented in analysis/README.md.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import fb_load as L  # noqa: E402
import fb_metrics as M  # noqa: E402

TBD = "\\textcolor{red}{TBD}"
SYNTH = "\\textcolor{red}{SYNTHETIC}"
NSMALL = "\\textcolor{red}{n too small}"
UNRESOLVED = "\\textcolor{red}{not resolved}"
MIN_POWER_RUNS = 3  # per mode, for an energy-per-verification figure (see compute_cost)

EXP_WORD = {"E1": "EOne", "E2": "ETwo", "E3": "EThree", "E5": "EFive"}
POLICY_EXPS = ("E3", "E5")
FIXED_POLICIES = OrderedDict([("Memory", "Memory only"), ("Always", "Always verify"),
                              ("Heuristic", "On-device rule (logged)"), ("Oracle", "Oracle")])
BUDGETS = OrderedDict([("Quarter", 0.25), ("Half", 0.50)])
E1_GROUPS = OrderedDict([("Session", "Same-session anchor"), ("MapSame", "Map, item in the fix point's room"),
                         ("MapOther", "Map, item in another room"), ("MapAll", "Map, all")])
NEG_TYPES = OrderedDict([("Empty", "empty"), ("Other", "other"), ("Lookalike", "lookalike")])
STAGES = OrderedDict([("Convert", "convertMs"), ("Detect", "detectMs"), ("Embed", "embedMs"), ("Total", "totalMs")])
POWER_GROUPS = ("cpu", "gpu", "tpu", "camera", "display", "memory", "other")
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
APP_PACKAGE = "edu.cs6301.findback"
# The swept policies of the measured policy figure and the rows of the ablation table. A fitted
# curve is the logistic risk score on those signals; the age-only curve is a plain threshold on
# the memory age (oldest first), with nothing fitted.
CURVES = OrderedDict([
    ("All", {"label": "All signals", "signals": list(M.SIGNALS), "fitted": True}),
    ("World", {"label": "World signals only", "signals": M.WORLD_SIGNALS, "fitted": True}),
    ("Frame", {"label": "Frame signals only", "signals": M.FRAME_SIGNALS, "fitted": True}),
    ("Age", {"label": "Age only (threshold, not fitted)", "signals": ["memory age"], "fitted": False}),
])
# Ground-truth classes of the verification-behaviour table (spec section 4, item 2).
CONFIRM_CLASSES = OrderedDict([
    ("AtMarker", "Item at the marker"), ("Stale", "Item not at the marker"),
    ("Absent", "absent"), ("Displaced", "displaced"), ("Replaced", "replaced"),
    ("PresentFar", "present, marker error over %.0f cm" % (M.R_PRESENT_M * 100)),
])
# The what-if model (spec section 4, item 3).
WHATIF_BASE_RATES = (0.1, 0.3, 0.5)
WHATIF_HEADLINE_BASE = 0.3
WHATIF_AUCS = OrderedDict([("Fifty", 0.5), ("Seventy", 0.7), ("Eighty", 0.8), ("Ninety", 0.9), ("Hundred", 1.0)])
WHATIF_GRID = np.round(np.linspace(0.5, 1.0, 101), 3)
FRAME_EXPS = ("E1", "E3")  # experiments whose present queries measure frame staleness


# ----------------------------------------------------------------------------- macro catalogue

def macro_catalogue():
    """Every macro results.tex defines, in order, with what it means. The set never depends on
    the data: a macro without data is TBD."""
    m = OrderedDict()

    def add(name, doc):
        assert name.isalpha() and name not in m, name
        m[name] = doc

    add("DataSites", "T1: sites (distinct home= values)")
    add("DataSessions", "T1: sessions with a study tag")
    add("DataRooms", "T1: rooms that hold a registered or queried item")
    add("DataItems", "T1: items (record lineages) registered or queried")
    add("DataRegistrations", "T1: registrations")
    add("DataQueries", "T1: finished queries with a ground truth, all five values")
    for t in L.GROUND_TRUTHS:
        add("Data" + t.capitalize(), "T1: queries with ground truth " + t)
    add("DataCaptures", "T1: captures (verify events)")
    add("DataBenchmarkCaptures", "T1: captures with a benchmark label")

    add("EOneQueries", "E1: queries with a usable ground truth")
    add("EOneMeasured", "E1: present queries with a measured marker error")
    for g, doc in E1_GROUPS.items():
        add("EOne%sN" % g, "E1 marker error, %s: measurements" % doc)
        add("EOne%sMedianCm" % g, "E1 marker error, %s: median, cm" % doc)
        add("EOne%sPNinetyCm" % g, "E1 marker error, %s: 90th percentile, cm" % doc)
    add("EOneMapRoomUnknownN", "E1: map-source measurements whose room relation to the fix point is unknown")
    add("EOneWalkN", "E1: map-source measurements with metres walked since the fix")
    add("EOneWalkSpearman", "E1: Spearman correlation of marker error with metres walked since the fix")
    for t in ("Image", "Spot"):
        add("EOneJump%sN" % t, "E1 re-fix jumps (kind refresh or manual, applied), fix type %s: count" % t)
        add("EOneJump%sMedianCm" % t, "E1 re-fix jumps, %s: median, cm" % t)
        add("EOneJump%sMaxCm" % t, "E1 re-fix jumps, %s: maximum, cm" % t)
        add("EOneJump%sMedianDeg" % t, "E1 re-fix jumps, %s: median, degrees" % t)
        add("EOneJump%sMaxDeg" % t, "E1 re-fix jumps, %s: maximum, degrees" % t)
    add("EOneDisagreements", "E1: re-fixes that asked the user (jump over the app's limit)")
    add("EOneDisagreementsAccepted", "E1: of those, accepted")
    add("EOneRoomNamedRate", "E1: share of queries for which the app named a room for the item (not a correctness check)")
    add("NavCrossRoomQueries", "E1 and E3: queries with a usable ground truth that started in another room than the item's (userRoom differs from itemRoom, both known)")
    add("NavCrossRoomNearRate", "E1 and E3: of those, the share that reached NEAR")
    add("NavCrossRoomSecondsMedian", "E1 and E3: median seconds from the start of such a query to NEAR, over those that reached it")
    add("NavCrossRoomSecondsN", "E1 and E3: queries behind that median")

    add("FrameStaleProtocol", "frame staleness: how the model was kept off its own test queries")
    add("FrameStaleTrainQueries", "frame staleness: present queries the model was trained on (smallest fold for leave-one-session-out)")
    add("FrameStaleQueries", "frame staleness: present queries of E1 and E3 the AUC is computed on")
    add("FrameStaleSessions", "frame staleness: sessions those queries come from (bootstrap clusters)")
    add("FrameStaleRate", "frame staleness: of those, the share with a marker error over R_PRESENT_M")
    add("FrameStaleAuc", "frame staleness: AUC of the out-of-sample frame-signals score for 'marker error over R_PRESENT_M' among present queries")
    add("FrameStaleAucCI", "frame staleness: 95% interval of that AUC, bootstrap over sessions")

    add("ETwoProtocol", "E2: how the threshold was chosen and where it was evaluated")
    add("ETwoCaptures", "E2: labelled captures")
    add("ETwoDevCaptures", "E2: labelled captures used only to choose the threshold and margin")
    add("ETwoEvalCaptures", "E2: labelled captures the reported numbers are computed on")
    add("ETwoUnlabelled", "E2: captures without a usable label (dismissed)")
    for n, lab in [("Target", "target")] + list(NEG_TYPES.items()):
        add("ETwo%sN" % n, "E2: evaluated captures labelled " + lab)
    add("ETwoThreshold", "E2: similarity threshold tau")
    add("ETwoThresholdSource", "E2: where tau comes from")
    add("ETwoMargin", "E2: margin m of rule 2, chosen on the dev split")
    add("ETwoClusters", "E2: clusters behind the bootstrap intervals")
    add("ETwoClusterUnit", "E2: what a cluster is")
    for n, lab in list(NEG_TYPES.items()) + [("All", "all negatives")]:
        add("ETwoAuc" + n, "E2: ROC AUC, target against " + lab)
        add("ETwoAuc%sCI" % n, "E2: 95% bootstrap interval of that AUC")
    add("ETwoAucAllLabelled", "E2: ROC AUC, target against all negatives, over all labelled captures of both splits (nothing is tuned for an AUC)")
    for rule, doc in (("Thr", "rule 1 (threshold), all evaluated captures"),
                      ("Sub", "rule 1 (threshold), captures with embeddings only"),
                      ("Mar", "rule 2 (threshold and margin), captures with embeddings only")):
        add("ETwo%sTprRate" % rule, "E2 %s: true-positive rate" % doc)
        for n, lab in list(NEG_TYPES.items()) + [("All", "all negatives pooled")]:
            add("ETwo%sFpr%sRate" % (rule, n), "E2 %s: false-positive rate on %s" % (doc, lab))
    add("ETwoMarginSubsetN", "E2: evaluated captures with a usable embedding file")
    for c in L.CONDITIONS:
        add("ETwoTpr%sRate" % c.capitalize(), "E2: true-positive rate of rule 1 under condition " + c)

    for exp in POLICY_EXPS:
        x = EXP_WORD[exp]
        add(x + "Queries", exp + ": queries with a usable ground truth")
        add(x + "Paired", exp + ": queries replayed under every policy")
        add(x + "Sessions", exp + ": sessions those queries come from (bootstrap clusters)")
        add(x + "NoCapture", exp + ": usable queries left out because they have no capture")
        add(x + "StaleRate", exp + ": replayed queries whose item is not at the marker")
        add(x + "Protocol", exp + ": how the learned policy was kept off its own test queries")
        add(x + "TrainQueries", exp + ": queries the learned policy was trained on (smallest fold for leave-one-session-out)")
        add(x + "Threshold", exp + ": similarity threshold used to judge a capture confirmed")
        add(x + "ThresholdSource", exp + ": where that threshold comes from")
        for p, doc in FIXED_POLICIES.items():
            add(x + p + "Captures", "%s %s: captures per query" % (exp, doc))
            add(x + p + "Calls", "%s %s: inference calls per query" % (exp, doc))
            add(x + p + "WrongLeadRate", "%s %s: wrong-lead rate" % (exp, doc))
            add(x + p + "WrongLeadCI", "%s %s: 95%% interval of the wrong-lead rate, bootstrap over sessions" % (exp, doc))
            add(x + p + "LostRate", "%s %s: lost-retrieval rate, per replayed query" % (exp, doc))
            add(x + p + "LostCI", "%s %s: 95%% interval of the lost-retrieval rate" % (exp, doc))
            add(x + p + "LostOfAtMarkerRate", "%s %s: lost retrievals among the queries whose item is at the marker" % (exp, doc))
        add(x + "RiskAuc", exp + ": AUC of the out-of-sample risk score for 'not at the marker'")
        add(x + "RiskAucCI", exp + ": 95% interval of that AUC")
        for b, v in BUDGETS.items():
            add(x + "Learned" + b + "WrongLead", "%s learned policy at %.2f captures per query: wrong-lead rate" % (exp, v))
            add(x + "Learned" + b + "WrongLeadCI", "%s learned policy at %.2f captures per query: 95%% interval" % (exp, v))
            add(x + "Learned" + b + "Lost", "%s learned policy at %.2f captures per query: lost-retrieval rate" % (exp, v))
            add(x + "Learned" + b + "LostCI", "%s learned policy at %.2f captures per query: 95%% interval" % (exp, v))
            add(x + "Random" + b + "WrongLead", "%s random capture with p = %.2f: expected wrong-lead rate" % (exp, v))
            add(x + "Random" + b + "Lost", "%s random capture with p = %.2f: expected lost-retrieval rate" % (exp, v))
        add(x + "LearnedAtHeuristicWrongLead", exp + ": learned policy's wrong-lead rate at the heuristic's captures per query")
        for key, doc in (("Age", "threshold on memory age, not fitted"), ("World", "learned policy, world signals only"),
                         ("Frame", "learned policy, frame signals only")):
            add(x + key + "Auc", "%s %s: AUC of its score for 'not at the marker' (the all-signals AUC is %sRiskAuc)" % (exp, doc, x))
            add(x + key + "AucCI", "%s %s: 95%% interval of that AUC" % (exp, doc))
        op = exp + " frame-signals policy at the operating point chosen on the dev split (fewest dev captures within %d points of always verifying)" % round(100 * M.WL_TOLERANCE)
        add(x + "FrameOpCaptures", op + ": captures per query on the test split")
        add(x + "FrameOpWrongLeadRate", op + ": wrong-lead rate on the test split")
        add(x + "FrameOpWrongLeadCI", op + ": 95% interval of the wrong-lead rate")
        add(x + "FrameOpLostRate", op + ": lost-retrieval rate on the test split")
        add(x + "FrameOpLostCI", op + ": 95% interval of the lost-retrieval rate")
        add(x + "FrameOpDevCaptures", op + ": captures per query at that point on the dev split (in-sample; how the point was chosen, not a result)")
        add(x + "FrameOpDevQueries", op + ": dev queries it was chosen on")

    add("EThreeConfirmQueries", "E3 verification behaviour: paired queries it is measured on (both splits)")
    add("EThreeConfirmSessions", "E3 verification behaviour: sessions (bootstrap clusters)")
    for key, doc in CONFIRM_CLASSES.items():
        add("EThreeConfirm%sRate" % key, "E3 verification behaviour: share of captures that confirm. %s" % doc)
        add("EThreeConfirm%sCI" % key, "E3 verification behaviour: 95%% interval of that share. %s" % doc)

    add("WhatIfBaseRatePct", "what-if MODEL: the base rate of stale queries the headline numbers assume, percent (a constant)")
    add("WhatIfTolerancePoints", "what-if MODEL and dev operating point: 'within this many percentage points of always verifying' (a constant)")
    for name, a in WHATIF_AUCS.items():
        add("WhatIfCapturesAuc" + name, "what-if MODEL, not a measurement: captures per query needed with a predictor of AUC %.1f at base rate %.1f" % (a, WHATIF_HEADLINE_BASE))

    for s, doc in (("Reg", "registration"), ("Ver", "verification")):
        for st in STAGES:
            add("Cost%s%sMedianMs" % (s, st), "T2 %s latency, %s: median, ms" % (doc, st.lower()))
            add("Cost%s%sPNinetyFiveMs" % (s, st), "T2 %s latency, %s: 95th percentile, ms" % (doc, st.lower()))
        add("Cost%sN" % s, "T2: %ss with latency logged" % doc)
    add("CostVerCallsMedian", "T2: inference calls per verification, median")
    add("CostStorageItemKB", "T2: stored bytes per item (crop and record), median, kB")
    add("CostStorageN", "T2: registrations with stored bytes logged")
    add("CostMapFileKB", "T2: largest map.json among the data folders, kB")
    add("CostMapFiles", "T2: data folders with a map.json")
    add("CostCaptureSecondsMedian", "T2: seconds from reaching the item (NEAR) to the first capture's result (msSinceNear + latency.totalMs), median")
    add("CostCaptureSecondsPNinetyFive", "T2: the same, 95th percentile")
    add("CostCaptureSecondsN", "T2: queries behind it")
    add("CostPowerTrackingMw", "T2: mean power of all monitored rails, tracking only, mW")
    add("CostPowerTrackingRuns", "T2: runs behind it")
    add("CostPowerTrackingRangeMw", "T2: lowest and highest run, tracking only, mW")
    add("CostPowerVerifyMw", "T2: mean power of all monitored rails, tracking plus periodic verification, mW")
    add("CostPowerVerifyRuns", "T2: runs behind it")
    add("CostPowerVerifyRangeMw", "T2: lowest and highest run, tracking plus periodic verification, mW")
    add("CostEnergyPerVerificationJ", "T2: extra rail energy per verification, J; 'not resolved' unless every verifying run is above every tracking-only run")
    add("CostEnergyVerifications", "T2: verifications behind it")

    # The names paper/findback.tex uses (its macros_fallback.tex shows them as TBD until they are
    # defined here). They are bare numbers for use inside a sentence, so a percentage from fewer
    # than MIN_N_BARE_RATE cases expands to "n too small (k/n)" instead.
    for name, doc in PAPER_MACROS.items():
        add(name, "paper: " + doc)
    return m


MIN_N_BARE_RATE = 20
PAPER_MACROS = OrderedDict([
    ("NSites", "homes = DataSites"),
    ("NRooms", "rooms = DataRooms"),
    ("NItems", "items = DataItems"),
    ("NItemsPerSite", "items per home (a range when the homes differ)"),
    ("NQueriesPaired", "E3 queries replayed under every policy = EThreePaired; the wrong-lead percentages below are over these"),
    ("NBenchCaptures", "all labelled E2 captures = ETwoCaptures"),
    ("MarkerErrSessionMedianCm", "E1 median marker error, same-session anchor, cm"),
    ("MarkerErrMapMedianCm", "E1 median marker error, map source (all rooms), cm"),
    ("MarkerErrMapPninetyCm", "E1 90th percentile marker error, map source, cm"),
    ("CrossRoomNearPct", "E1 and E3: percent of the queries that started in another room and reached NEAR"),
    ("CrossRoomSecondsToNear", "E1 and E3: median seconds to NEAR for those queries"),
    ("FrameStalePct", "E1 and E3: percent of the evaluated present queries with a marker error over R_PRESENT_M"),
    ("FrameStaleAUC", "AUC of the frame signals for frame staleness = FrameStaleAuc"),
    ("RelockJumpMedianCm", "E1 median map jump of a re-lock at a spot (map_fix kind manual, fix type SPOT), cm"),
    ("RelockJumpMedianDeg", "the same, degrees"),
    ("VerifyAUC", "E2 ROC AUC, target against all other labels, over all NBenchCaptures labelled captures"),
    ("ThresholdTau", "similarity threshold tau = ETwoThreshold"),
    ("VerifyTPRPct", "E2 percent of target captures confirmed at tau (evaluated captures)"),
    ("VerifyFPRPct", "E2 percent of non-target captures confirmed at tau (evaluated captures, the three kinds pooled)"),
    ("LookalikeFPRThreshPct", "E2 percent of look-alike captures confirmed by the threshold rule (same captures as the margin figure when there is one)"),
    ("LookalikeFPRMarginPct", "E2 percent of look-alike captures confirmed by the threshold-and-margin rule"),
    ("WrongLeadMemoryPct", "E3 wrong leads, memory only, percent of replayed queries"),
    ("WrongLeadAlwaysPct", "E3 wrong leads, always verify, percent of replayed queries"),
    ("WrongLeadSelectivePct", "E3 wrong leads, the logged on-device heuristic (query_start.policyVerify), percent of replayed queries"),
    ("LostRetrievalAlwaysPct", "E3 always verify: percent of the queries whose item is at the marker that the capture did not confirm"),
    ("CapturesSelective", "E3 captures per query of the logged on-device heuristic"),
    ("CaptureSavingPct", "E3 percent fewer captures than always verify for the logged on-device heuristic"),
    ("OnDeviceWrongLeadPct", "E3 wrong leads of the logged on-device rule, percent (same value as WrongLeadSelectivePct)"),
    ("OnDeviceCaptures", "E3 captures per query of the logged on-device rule (same value as CapturesSelective)"),
    ("OnDeviceCaptureSavingPct", "E3 percent fewer captures than always verify for the logged on-device rule (same value as CaptureSavingPct)"),
    ("FramePolicyWrongLeadPct", "E3 test split: wrong leads of the frame-signals policy at the operating point chosen on dev, percent"),
    ("FramePolicyCaptures", "E3 test split: its captures per query"),
    ("FramePolicyCaptureSavingPct", "E3 test split: its percent fewer captures than always verify"),
    ("FramePolicyLostPct", "E3 test split: its lost retrievals, percent of replayed queries"),
    ("ConfirmAtMarkerPct", "E3 percent of captures that confirm when the item is at the marker (c_p of the what-if model)"),
    ("ConfirmStalePct", "E3 percent of captures that confirm when the item is not at the marker (c_s of the what-if model)"),
    ("RegisterLatencyMs", "median registration processing time (convert + detect + embed), ms"),
    ("VerifyLatencyMs", "median verification time (latency.totalMs), ms"),
    ("BytesPerItem", "median stored bytes per item (register.recordBytes)"),
    ("SecondsPerCapture", "median seconds from reaching the item to the first capture's result"),
    ("EnergyPerVerifymJ", "extra rail energy per verification, mJ"),
    ("RPresentCm", "the analysis constant R_PRESENT_M in cm (a constant, not a measurement)"),
])


MACROS = macro_catalogue()


class Numbers:
    """The values of the macros. A macro that is never set stays TBD; in synthetic mode every
    macro, whatever was computed, is SYNTHETIC."""

    def __init__(self, synthetic):
        self.synthetic = synthetic
        self._tex = {}
        self.raw = {}

    def set(self, name, tex, raw=None):
        if name not in MACROS:
            raise KeyError("macro %s is not in the catalogue" % name)
        if tex is not None:
            self._tex[name] = tex
        if raw is not None:
            self.raw[name] = raw

    def copy(self, name, source):
        """Give macro `name` the value of macro `source` (TBD if that one has none)."""
        if name not in MACROS or source not in MACROS:
            raise KeyError("%s or %s is not in the catalogue" % (name, source))
        if source in self._tex:
            self._tex[name] = self._tex[source]
        if source in self.raw:
            self.raw[name] = self.raw[source]

    def get(self, name):
        if name not in MACROS:
            raise KeyError(name)
        return SYNTH if self.synthetic else self._tex.get(name, TBD)

    def cell(self, tex):
        """A table cell that is not a macro: the same marking rules."""
        if self.synthetic:
            return SYNTH
        return TBD if tex is None else tex

    def results_tex(self, header):
        lines = ["% " + h for h in header]
        lines += ["% Needs \\usepackage{xcolor}. Rate macros expand to 'percent (k/n)'; other macros are bare",
                  "% numbers whose unit is in the macro name. Do not edit: rerun analysis/paper_figures.py.", ""]
        for name, doc in MACROS.items():
            lines.append("\\newcommand{\\%s}{%s}%% %s" % (name, self.get(name), doc))
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- TeX formatting

TEX_ESCAPES = {"\\": "\\textbackslash{}", "%": "\\%", "_": "\\_", "&": "\\&", "#": "\\#", "$": "\\$", "{": "\\{",
               "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}


def t_text(s):
    return None if s is None else "".join(TEX_ESCAPES.get(ch, ch) for ch in str(s))


def t_int(n, have_data=True):
    return None if n is None or not have_data else "%d" % n


def t_num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return "%.*f" % (digits, x)


def t_rate(k, n):
    """'12\\% (5/40)': a rate always carries its counts."""
    if not n:
        return None
    return "%.*f\\%% (%d/%d)" % (1 if n >= 200 else 0, 100.0 * k / n, k, n)


def t_pct(x, digits=1):
    """A rate that is an expectation or an interpolation, so it has no integer counts."""
    if x is None or math.isnan(x):
        return None
    return "%.*f\\%%" % (digits, 100.0 * x)


def t_bare_pct(k, n):
    """A percentage without its counts, for a sentence. Too few cases -> 'n too small (k/n)'."""
    if not n:
        return None
    if n < MIN_N_BARE_RATE:
        return "\\textcolor{red}{n too small (%d/%d)}" % (k, n)
    return "%.*f" % (1 if n >= 200 else 0, 100.0 * k / n)


def t_ci(boot, index=None, scale=1.0, digits=2, unit=""):
    if boot is None:
        return None
    if not boot["ok"]:
        return NSMALL
    lo = boot["lo"] if index is None else boot["lo"][index]
    hi = boot["hi"] if index is None else boot["hi"][index]
    if math.isnan(lo) or math.isnan(hi):
        return None
    return "[%.*f, %.*f]%s" % (digits, lo * scale, digits, hi * scale, unit)


def or_small(value_tex, n, min_n):
    """value_tex, or 'n too small' when there is data but fewer than min_n, or None without data."""
    if value_tex is not None:
        return value_tex
    return NSMALL if 0 < n < min_n else None


# ----------------------------------------------------------------------------- thresholds

class Tau:
    """How a capture is judged confirmed in a replay: one threshold for all captures, or the
    on-device threshold logged with each capture."""

    def __init__(self, value, source):
        self.value, self.source = value, source

    def of(self, capture):
        return self.value if self.value is not None else capture.get("threshold")

    def confirmed(self, capture):
        tau = self.of(capture)
        return None if tau is None or capture.get("sim") is None else capture["sim"] >= tau

    def describe(self, captures):
        """(value or None, text) for the captures it was applied to."""
        if self.value is not None:
            return self.value, self.source
        logged = sorted({c["threshold"] for c in captures if c.get("threshold") is not None})
        if len(logged) == 1:
            return logged[0], self.source
        if not logged:
            return None, self.source
        return None, "%s (varies: %s)" % (self.source, ", ".join("%.2f" % v for v in logged))


# ----------------------------------------------------------------------------- T1

def compute_dataset(events, queries, captures, regs):
    def blank():
        return {"sessions": set(), "rooms": set(), "items": set(), "registrations": 0, "truth": Counter(),
                "captures": 0, "labelled": 0, "unfinished": 0}
    groups = defaultdict(blank)

    def key(exp, home):
        return (exp or "?", home or "?")
    for e in events:
        groups[key(L.tag_exp(e["_tag"]), L.tag_home(e["_tag"]))]["sessions"].add(e.get("session"))
    for r in regs:
        g = groups[key(r["exp"], r["home"])]
        g["registrations"] += 1
        g["items"].add(r["lineage"])
        if r["room"] is not None:
            g["rooms"].add(r["room"])
    for q in queries:
        g = groups[key(q["exp"], q["home"])]
        if q["truth"] is None:
            g["unfinished"] += 1
        else:
            g["truth"][q["truth"]] += 1
        g["items"].add(q["lineage"])
        if q["item_room"] is not None:
            g["rooms"].add(q["item_room"])
    for c in captures:
        g = groups[key(c["exp"], c["home"])]
        g["captures"] += 1
        g["labelled"] += c["label"] in L.LABELS
    homes = sorted({k[1] for k in groups})
    total = blank()
    rooms, items = set(), set()
    for (exp, home), g in groups.items():
        total["sessions"] |= g["sessions"]
        rooms |= {(home, r) for r in g["rooms"]}
        items |= {(home, i) for i in g["items"]}
        total["registrations"] += g["registrations"]
        total["truth"] += g["truth"]
        total["captures"] += g["captures"]
        total["labelled"] += g["labelled"]
        total["unfinished"] += g["unfinished"]
    total["rooms"], total["items"] = rooms, items
    per_site = Counter(home for home, _ in items if home != "?")
    return {"groups": OrderedDict(sorted(groups.items())), "total": total,
            "sites": [h for h in homes if h != "?"], "no_home": "?" in homes,
            "items_per_site": dict(per_site), "have_data": bool(events)}


# ----------------------------------------------------------------------------- E1

def compute_e1(queries, fixes, ds):
    qs = [q for q in queries if q["exp"] == "E1" and q["truth"] in M.USABLE_TRUTHS]
    rows = []
    for q in qs:
        if q["truth"] != "present" or q["marker_error_m"] is None:
            continue
        src = (q["marker_source"] or "").upper()
        group = "Other"
        if src == "SESSION":
            group = "Session"
        elif src == "MAP":
            maps = ds.folders[q["folder"]].maps
            m = maps.get(q["map_id"]) or (next(iter(maps.values())) if len(maps) == 1 and q["map_id"] is None else None)
            fix_room = L.fix_point_room(m[0] if m else None, q["marker_fix_point"] or q["fix_point"])
            if fix_room is None or q["item_room"] is None:
                group = "MapUnknown"
            else:
                group = "MapSame" if fix_room == q["item_room"] else "MapOther"
        metres = q["marker_meters_since_fix"] if q["marker_meters_since_fix"] is not None else q["meters_since_fix"]
        rows.append({"err": q["marker_error_m"], "group": group, "metres": metres, "session": q["session"]})
    groups = {g: [r["err"] for r in rows if r["group"] == g] for g in ("Session", "MapSame", "MapOther", "MapUnknown", "Other")}
    groups["MapAll"] = groups["MapSame"] + groups["MapOther"] + groups["MapUnknown"]
    walk = [(r["metres"], r["err"], r["group"]) for r in rows if r["group"].startswith("Map") and r["metres"] is not None]
    rho = None
    if len(walk) >= M.MIN_N_CURVE:
        from scipy.stats import spearmanr
        x, y = [w[0] for w in walk], [w[1] for w in walk]
        if len(set(x)) > 1 and len(set(y)) > 1:
            rho = float(spearmanr(x, y)[0])
    e1_fixes = [f for f in fixes if f["exp"] == "E1" and (f["kind"] or "").lower() in ("refresh", "manual")]
    jumps = {}
    for t in ("IMAGE", "SPOT"):
        js = [f for f in e1_fixes if f["applied"] and f["jump_m"] is not None and (f["fix_type"] or "").upper() == t]
        jumps[t] = {"m": [f["jump_m"] for f in js], "deg": [f["jump_deg"] for f in js if f["jump_deg"] is not None]}
    asked = [f for f in e1_fixes if f["disagreement"]]
    named = [q for q in qs if q["item_room_name"] is not None]
    return {"n_queries": len(qs), "rows": rows, "groups": groups, "walk": walk, "rho": rho, "jumps": jumps,
            "n_asked": len(asked), "n_accepted": sum(f["accepted"] is True for f in asked),
            "n_fixes": len(e1_fixes), "n_named": len(named)}


# ----------------------------------------------------------------------------- E2

def compute_e2(captures, ds, cli_tau, cluster_unit):
    caps = [c for c in captures if c["exp"] == "E2"]
    lab = [c for c in caps if c["label"] in L.LABELS and c["sim"] is not None]
    R = {"n_events": len(caps), "n_labelled": len(lab), "n_unlabelled": len(caps) - len(lab),
         "cluster_unit": cluster_unit, "warnings": []}
    dev = [c for c in lab if c["split"] == "dev"]
    test = [c for c in lab if c["split"] == "test"]

    def pos_neg(rows):
        return [c for c in rows if c["label"] == "target"], [c for c in rows if c["label"] != "target"]
    split = bool(dev) and bool(test)
    ev = test if split else lab
    R["n_dev"] = len(dev) if split else 0
    R["n_no_split_ignored"] = (len(lab) - len(dev) - len(test)) if split else 0
    R["ev"] = ev
    dev_pos, dev_neg = pos_neg(dev)
    dev_ok = split and min(len(dev_pos), len(dev_neg)) >= M.MIN_ROC_PER_CLASS

    if cli_tau is not None:
        tau = Tau(cli_tau, "set on the command line")
    elif dev_ok:
        tau = Tau(M.youden_threshold([c["sim"] for c in dev_pos], [c["sim"] for c in dev_neg]),
                  "chosen on the E2 dev split (Youden's J)")
    else:
        tau = Tau(None, "the on-device threshold logged with each capture")
    R["tau"] = tau
    if dev_ok and cli_tau is None:
        j = np.mean([c["sim"] >= tau.value for c in dev_pos]) - np.mean([c["sim"] >= tau.value for c in dev_neg])
        if j <= 0:
            R["warnings"].append("the E2 dev split shows no separation between target and other captures (Youden's J = %.2f at the "
                                 "chosen threshold %.2f); the threshold is not meaningful" % (j, tau.value))
    if split:
        R["protocol"] = "threshold and margin chosen on the dev split, numbers from the test split" if dev_ok or cli_tau is not None \
            else "dev split too small to choose a threshold; a-priori threshold, numbers from the test split"
        if not dev_ok:
            R["warnings"].append("E2 dev split has %d target and %d non-target captures; %d of each are needed to choose a threshold"
                                 % (len(dev_pos), len(dev_neg), M.MIN_ROC_PER_CLASS))
    else:
        R["protocol"] = "no dev/test split: a-priori threshold, numbers from all labelled captures"
        if dev or test:
            R["warnings"].append("E2 has only a %s split; nothing was tuned on it" % ("dev" if dev else "test"))

    # Margin inputs (best candidate against the other records as of the capture time).
    sources, mismatch, reasons = Counter(), 0, Counter()
    for c in (dev if split else []) + ev:
        mi = L.margin_inputs(ds, c)
        c["_mi"] = mi
        if mi["ok"]:
            sources[mi["source"]] += 1
            c["_margin"] = float("nan") if mi["other_sim"] is None else mi["best_sim"] - mi["other_sim"]
            if mi["logged_diff"] is not None and mi["logged_diff"] > 1e-3:
                mismatch += 1
        else:
            reasons[mi["reason"]] += 1
    R["margin_sources"], R["margin_missing"], R["sim_mismatch"] = sources, reasons, mismatch

    def conf1(c):
        return bool(tau.confirmed(c))
    margin = None
    dev_m = [c for c in dev if c.get("_mi", {}).get("ok")] if split else []
    dmp, dmn = pos_neg(dev_m)
    if min(len(dmp), len(dmn)) >= M.MIN_ROC_PER_CLASS and all(tau.of(c) is not None for c in dev_m):
        margin = M.youden_margin([conf1(c) for c in dmp], [c["_margin"] for c in dmp],
                                 [conf1(c) for c in dmn], [c["_margin"] for c in dmn])
    R["margin"] = margin

    def conf2(c):
        return conf1(c) and (math.isnan(c["_margin"]) or c["_margin"] >= margin)

    pos, neg = pos_neg(ev)
    R["n_by_label"] = Counter(c["label"] for c in ev)
    judged = [c for c in ev if tau.of(c) is not None]
    R["n_no_threshold"] = len(ev) - len(judged)
    cluster = (lambda c: c["session"]) if cluster_unit == "session" else (lambda c: (c["folder"], c["record"]))
    R["roc"] = {}
    for name, label in list(NEG_TYPES.items()) + [("All", None)]:
        ng = [c for c in neg if label is None or c["label"] == label]
        entry = {"n_pos": len(pos), "n_neg": len(ng), "auc": None, "boot": None, "fpr": None, "tpr": None}
        if min(len(pos), len(ng)) >= M.MIN_ROC_PER_CLASS:
            ps = np.array([c["sim"] for c in pos])
            ns = np.array([c["sim"] for c in ng])
            entry["auc"] = M.auc(ps, ns)
            entry["fpr"], entry["tpr"], _ = M.roc_points(ps, ns)
            scores = np.r_[ps, ns]
            y = np.r_[np.ones(ps.size, bool), np.zeros(ns.size, bool)]
            ids = [cluster(c) for c in pos + ng]
            codes = np.unique(np.array([repr(i) for i in ids]), return_inverse=True)[1]
            entry["boot"] = M.cluster_bootstrap(lambda idx: M.auc(scores[idx][y[idx]], scores[idx][~y[idx]]), codes)
        R["roc"][name] = entry
    R["n_clusters"] = len({repr(cluster(c)) for c in ev})
    # An AUC has no tuned parameter, so the one quoted for "all labelled captures" may use both splits.
    all_pos, all_neg = pos_neg(lab)
    R["auc_all_labelled"] = (M.auc([c["sim"] for c in all_pos], [c["sim"] for c in all_neg])
                             if min(len(all_pos), len(all_neg)) >= M.MIN_ROC_PER_CLASS else None)
    R["n_all_pos"], R["n_all_neg"] = len(all_pos), len(all_neg)

    def rule_rates(rows, conf):
        p, n = pos_neg(rows)
        out = {"Tpr": (sum(conf(c) for c in p), len(p)), "FprAll": (sum(conf(c) for c in n), len(n))}
        for name, label in NEG_TYPES.items():
            ng = [c for c in n if c["label"] == label]
            out["Fpr" + name] = (sum(conf(c) for c in ng), len(ng))
        return out
    R["Thr"] = rule_rates(judged, conf1)
    sub = [c for c in judged if c["_mi"]["ok"]]
    R["n_margin_subset"] = len(sub)
    R["Sub"] = rule_rates(sub, conf1) if margin is not None else None
    R["Mar"] = rule_rates(sub, conf2) if margin is not None else None
    R["conditions"] = OrderedDict()
    for cond in L.CONDITIONS:
        p_all = [c for c in judged if c["label"] == "target" and c["condition"] == cond]
        p_sub = [c for c in p_all if c["_mi"]["ok"]]
        R["conditions"][cond] = {
            "thr": (sum(conf1(c) for c in p_all), len(p_all)),
            "sub": (sum(conf1(c) for c in p_sub), len(p_sub)) if margin is not None else None,
            "mar": (sum(conf2(c) for c in p_sub), len(p_sub)) if margin is not None else None,
        }
    R["n_condition_other"] = sum(1 for c in judged if c["label"] == "target" and c["condition"] not in L.CONDITIONS)
    return R


# ----------------------------------------------------------------------------- E3, E5

def compute_policy(exp, queries, tau):
    """Replay of every policy on the paired queries of one experiment."""
    allq = [q for q in queries if q["exp"] == exp]
    usable = [q for q in allq if q["truth"] in M.USABLE_TRUTHS]
    P = {"exp": exp, "n_all": len(allq), "n_usable": len(usable),
         "n_unfinished": sum(q["truth"] is None for q in allq), "n_skip": sum(q["truth"] == "skip" for q in allq),
         "by_truth": Counter(q["truth"] for q in usable), "warnings": [], "tau": tau}
    by_mode = Counter(q["capture_mode"] or "not logged" for q in usable)
    P["by_mode"] = by_mode
    in_mode = [q for q in usable if q["capture_mode"] in (None, "PAIRED")]
    P["n_mode_excluded"] = len(usable) - len(in_mode)
    if P["n_mode_excluded"]:
        P["warnings"].append("%s: %d usable queries were logged in FREE or BENCHMARK mode and are not replayed "
                             "(in those modes the user chooses when to capture)" % (exp, P["n_mode_excluded"]))
    if by_mode.get("not logged"):
        P["warnings"].append("%s: %d queries have no captureMode; they are replayed as PAIRED" % (exp, by_mode["not logged"]))
    paired = [q for q in in_mode if q["first_capture"] is not None and tau.confirmed(q["first_capture"]) is not None]
    P["no_capture"] = Counter((q["no_capture_reason"] or "no reason logged") for q in in_mode if q["first_capture"] is None)
    P["n_no_capture"] = sum(P["no_capture"].values())
    P["n_no_threshold"] = len(in_mode) - len(paired) - P["n_no_capture"]
    P["n_paired_all"] = len(paired)
    # Diagnostics of the 0.30 m rule.
    P["n_present_far"] = sum(q["truth"] == "present" and q["marker_error_m"] is not None and q["marker_error_m"] > M.R_PRESENT_M for q in paired)
    P["n_present_unmeasured"] = sum(q["truth"] == "present" and q["marker_error_m"] is None for q in paired)
    P["n_displaced_near"] = sum(q["truth"] == "displaced" and q["displacement_m"] is not None and q["displacement_m"] <= M.R_PRESENT_M for q in paired)

    n = len(paired)
    stale_all = np.array([not M.at_marker(q["truth"], q["marker_error_m"]) for q in paired], bool)
    conf_all = np.array([bool(tau.confirmed(q["first_capture"])) for q in paired], bool)
    calls_all = np.array([q["first_capture"]["inference_calls"] if q["first_capture"]["inference_calls"] is not None else np.nan for q in paired], float)
    sessions_all = np.array([str(q["session"]) for q in paired])
    splits_all = [q["split"] for q in paired]
    x_all = M.feature_matrix(paired) if n else np.zeros((0, len(M.FEATURES)))
    oos = M.out_of_sample_risk(x_all, stale_all, sessions_all, splits_all) if n else {
        "protocol": None, "eval": np.zeros(0, bool), "risk": None, "reason": "no paired queries", "folds": [], "n_train": 0, "warning": None}
    if oos["warning"]:
        P["warnings"].append("%s: %s" % (exp, oos["warning"]))
    ev = oos["eval"]
    P["protocol"], P["n_train"], P["learned_reason"] = oos["protocol"], oos["n_train"], oos["reason"]
    P["n_split_ignored"] = int(n - ev.sum()) - (oos["n_train"] if oos["protocol"] == "dev/test" else 0)
    stale, conf, calls, sessions = stale_all[ev], conf_all[ev], calls_all[ev], sessions_all[ev]
    heur = [q["policy_verify"] for q, keep in zip(paired, ev) if keep]
    risk = oos["risk"][ev] if oos["risk"] is not None else None
    ne = int(stale.size)
    P.update({"n": ne, "n_sessions": len(set(sessions.tolist())), "n_stale": int(stale.sum()),
              "stale": stale, "conf": conf, "risk": risk, "sessions": sessions})
    tau_caps = [q["first_capture"] for q, keep in zip(paired, ev) if keep]
    P["tau_value"], P["tau_text"] = tau.describe(tau_caps)
    P["policies"], P["random"], P["learned"] = OrderedDict(), None, None
    P["curves"] = OrderedDict((k, None) for k in CURVES)
    P["curve_reason"] = {k: None for k in CURVES}
    P["op"], P["op_reason"] = None, None
    P["mobility_auc"] = None
    P["confirm"] = confirm_shares(paired, stale_all, conf_all, sessions_all)
    if ne == 0:
        return P
    calls_known = not np.isnan(calls).any()
    masks = OrderedDict([("Memory", M.mask_memory_only(ne)), ("Always", M.mask_always_verify(ne)),
                         ("Heuristic", None if any(h is None for h in heur) else np.array(heur, bool)),
                         ("Oracle", M.mask_oracle(stale))])
    if masks["Heuristic"] is None:
        P["warnings"].append("%s: policyVerify is missing on some queries; the on-device rule is not replayed" % exp)

    def replayed(mask):
        counts = M.replay(stale, conf, mask, calls if calls_known else None)
        cap = mask.astype(float)
        wrong = np.where(mask, stale & conf, stale).astype(float)
        lost = (mask & ~stale & ~conf).astype(float)
        boot = M.cluster_bootstrap(lambda idx: np.array([cap[idx].mean(), wrong[idx].mean(), lost[idx].mean()]), sessions)
        return {"counts": counts, "rates": M.rates(counts), "boot": boot, "mask": mask}
    for name, mask in masks.items():
        P["policies"][name] = None if mask is None else replayed(mask)
    P["random"] = {b: M.random_policy_expected(stale, conf, p, calls if calls_known else None) for b, p in BUDGETS.items()}
    P["random"]["ends"] = (M.random_policy_expected(stale, conf, 0.0), M.random_policy_expected(stale, conf, 1.0))

    if ne < M.MIN_POLICY_QUERIES:
        small = "n too small: %d replayed queries, %d needed" % (ne, M.MIN_POLICY_QUERIES)
        P["curve_reason"] = {k: small for k in CURVES}
        P["learned_reason"] = P["op_reason"] = small
        return P
    both = min(int(stale.sum()), int((~stale).sum())) >= M.MIN_ROC_PER_CLASS

    def swept(risk):
        """Operating curve, AUC for 'not at the marker' and their intervals for one risk score."""
        x, wl, lost, thr = M.sweep_risk(stale, conf, risk)

        def stat(idx):
            xi, wi, li, _ = M.sweep_risk(stale[idx], conf[idx], risk[idx])
            return np.r_[np.interp(GRID, xi, wi), np.interp(GRID, xi, li), M.auc(risk[idx][stale[idx]], risk[idx][~stale[idx]])]
        boot = M.cluster_bootstrap(stat, sessions)
        g = GRID.size
        return {"x": x, "wl": wl, "lost": lost, "thr": thr, "risk": risk,
                "auc": M.auc(risk[stale], risk[~stale]) if both else None, "auc_small": not both,
                "boot": boot, "wl_band": (boot["lo"][:g], boot["hi"][:g]), "lost_band": (boot["lo"][g:2 * g], boot["hi"][g:2 * g]),
                "auc_boot": {"ok": boot["ok"], "lo": float(boot["lo"][-1]), "hi": float(boot["hi"][-1])},
                "area": M.curve_area(x, wl)}
    for name, spec in CURVES.items():
        if spec["fitted"]:
            o = oos if name == "All" else M.out_of_sample_risk(x_all, stale_all, sessions_all, splits_all,
                                                              columns=M.columns_for(spec["signals"]))
            if o["risk"] is None:
                P["curve_reason"][name] = o["reason"]
                continue
            P["curves"][name] = swept(o["risk"][ev])
        else:
            # A threshold on the logged age itself: the oldest memories are captured first. Nothing
            # is fitted, so there is no training set. A query without an age is captured last.
            ages = np.array([q["age_min"] if q["age_min"] is not None else -np.inf for q, keep in zip(paired, ev) if keep], float)
            if np.all(np.isinf(ages)):
                P["curve_reason"][name] = "memory age was not logged"
                continue
            P["n_age_missing"] = int(np.isinf(ages).sum())
            P["curves"][name] = swept(ages)
    P["learned"] = P["curves"]["All"]
    # Mobility class alone as a score (the ordinal the features use; nothing fitted). Spec section 4
    # asks for it only as a mark on the what-if axis when there is E5 data.
    mobility = x_all[ev][:, M.FEATURES.index("mobility")]
    P["mobility_auc"] = M.auc(mobility[stale], mobility[~stale]) if both else None
    P["constant_signals"] = [s for s in M.SIGNALS if all(np.ptp(x_all[:, c]) == 0 for c in M.columns_for([s]))]

    # The frame-signals policy at one operating point, fixed on the dev split before the test
    # split is looked at: the fewest dev captures whose dev wrong-lead rate is within
    # WL_TOLERANCE of always verifying.
    frame = P["curves"]["Frame"]
    if oos["protocol"] != "dev/test":
        P["op_reason"] = "needs a dev and a test split (there is none), so no operating point can be fixed beforehand"
    elif frame is None:
        P["op_reason"] = P["curve_reason"]["Frame"]
    else:
        cols = M.columns_for(M.FRAME_SIGNALS)
        dev, dev_risk = M.dev_risk_for_operating_point(x_all, stale_all, splits_all, columns=cols)
        if dev_risk is None:
            P["op_reason"] = "n too small: the dev split cannot train the frame-signals model"
        else:
            thr, dev_caps, dev_wl = M.choose_operating_threshold(stale_all[dev], conf_all[dev], dev_risk)
            dev_always = M.rates(M.replay(stale_all[dev], conf_all[dev], M.mask_always_verify(dev.size)))["wrong_lead_rate"]
            P["op"] = dict(replayed(M.mask_risk(frame["risk"], thr)), threshold=thr, dev_n=int(dev.size),
                           dev_captures=dev_caps, dev_wrong=dev_wl, dev_always_wrong=dev_always)
    return P


def confirm_shares(paired, stale, conf, sessions):
    """Measured verification behaviour (spec section 4, item 2): of the captures of all paired
    queries of an experiment (both splits; nothing here is fitted on them), the share that
    confirm, by where the item really was. Intervals: bootstrap over sessions."""
    truth = np.array([q["truth"] for q in paired])
    far = np.array([q["truth"] == "present" and q["marker_error_m"] is not None and q["marker_error_m"] > M.R_PRESENT_M
                    for q in paired], bool)
    member = OrderedDict([("AtMarker", ~stale), ("Stale", stale.copy()), ("Absent", truth == "absent"),
                          ("Displaced", truth == "displaced"), ("Replaced", truth == "replaced"), ("PresentFar", far)])
    out = {"n": int(stale.size), "n_sessions": len(set(sessions.tolist())),
           "counts": OrderedDict((k, (int((conf & m).sum()), int(m.sum()))) for k, m in member.items())}
    if stale.size == 0:
        out["boot"] = None
        return out
    masks = list(member.values())

    def stat(idx):
        c = conf[idx]
        return np.array([c[m[idx]].mean() if m[idx].any() else np.nan for m in masks])
    out["boot"] = M.cluster_bootstrap(stat, sessions)
    return out


def compute_frame_staleness(queries):
    """Measured frame staleness (spec section 4, item 1): among queries whose item was not moved
    (ground truth `present`) in E1 and E3, how well the frame signals predict a marker error over
    0.30 m. Out-of-sample by the same rules as the learned policy. A `present` query without a
    measurement counts as not stale, as in section 2. No capture is needed, so E1 counts."""
    rows = [q for q in queries if q["exp"] in FRAME_EXPS and q["truth"] == "present"]
    F = {"n_rows": len(rows), "by_exp": Counter(q["exp"] for q in rows),
         "n_unmeasured": sum(q["marker_error_m"] is None for q in rows), "auc": None, "boot": None, "auc_small": False}
    label = np.array([q["marker_error_m"] is not None and q["marker_error_m"] > M.R_PRESENT_M for q in rows], bool)
    F["n_stale_all"] = int(label.sum())
    sessions = np.array([str(q["session"]) for q in rows])
    if not rows:
        F.update({"protocol": None, "n_train": 0, "n": 0, "n_stale": 0, "n_sessions": 0, "reason": "no present queries in E1 or E3",
                  "n_split_ignored": 0})
        return F
    oos = M.out_of_sample_risk(M.feature_matrix(rows), label, sessions, [q["split"] for q in rows],
                               columns=M.columns_for(M.FRAME_SIGNALS))
    ev = oos["eval"]
    lab, ses = label[ev], sessions[ev]
    F.update({"protocol": oos["protocol"], "n_train": oos["n_train"], "n": int(ev.sum()), "n_stale": int(lab.sum()),
              "n_sessions": len(set(ses.tolist())), "reason": oos["reason"], "warning": oos["warning"],
              "n_split_ignored": int(len(rows) - ev.sum()) - (oos["n_train"] if oos["protocol"] == "dev/test" else 0)})
    if oos["risk"] is None:
        return F
    risk = oos["risk"][ev]
    if min(int(lab.sum()), int((~lab).sum())) < M.MIN_ROC_PER_CLASS:
        F["auc_small"] = True
        F["reason"] = "n too small: %d of the %d evaluated present queries have a marker error over %.0f cm; %d of each kind are needed" % (
            int(lab.sum()), int(ev.sum()), M.R_PRESENT_M * 100, M.MIN_ROC_PER_CLASS)
        return F
    F["auc"] = M.auc(risk[lab], risk[~lab])
    F["boot"] = M.cluster_bootstrap(lambda idx: M.auc(risk[idx][lab[idx]], risk[idx][~lab[idx]]), ses)
    return F


def compute_cross_room(queries):
    """Guidance across rooms (spec F1): of the E1 and E3 queries that start in another room than
    the item's, the share that reached NEAR and the time it took."""
    usable = [q for q in queries if q["exp"] in FRAME_EXPS and q["truth"] in M.USABLE_TRUTHS]
    known = [q for q in usable if q["user_room"] is not None and q["item_room"] is not None]
    cross = [q for q in known if q["user_room"] != q["item_room"]]
    judged = [q for q in cross if q["near_reached"] is not None]
    reached = [q for q in judged if q["near_reached"]]
    # Not in any number, only in the output: cross-room queries that were skipped or never finished.
    # A walk that failed is likely to end that way, so the share above can flatter the guidance.
    left_out = [q for q in queries if q["exp"] in FRAME_EXPS and q["truth"] not in M.USABLE_TRUTHS
                and q["user_room"] is not None and q["item_room"] is not None and q["user_room"] != q["item_room"]]
    return {"n_usable": len(usable), "n_room_unknown": len(usable) - len(known), "n_same_room": len(known) - len(cross),
            "n_cross": len(cross), "n_judged": len(judged), "n_reached": len(reached),
            "seconds": [q["time_to_near_ms"] / 1000.0 for q in reached if q["time_to_near_ms"] is not None],
            "n_left_out": len(left_out), "n_left_out_not_reached": sum(not q["near_reached"] for q in left_out)}


def compute_whatif(PS, FR):
    """The what-if curve for world staleness (spec section 4, item 3). A model, not a measurement:
    its only measured inputs are the two confirmation shares of the paired E3 queries."""
    counts = PS["E3"]["confirm"]["counts"]
    (kp, n_p), (ks, n_s) = counts["AtMarker"], counts["Stale"]
    W = {"state": None, "c_p": None, "c_s": None, "counts": ((kp, n_p), (ks, n_s)), "curves": None, "headline": None}
    if n_p == 0 and n_s == 0:
        W["state"] = "no data"
    elif min(n_p, n_s) < M.MIN_N_SHARE:
        W["state"] = "n too small"
    else:
        c_p, c_s = kp / n_p, ks / n_s
        W["c_p"], W["c_s"] = c_p, c_s
        W["curves"] = OrderedDict((b, np.array([M.whatif_captures_needed(b, a, c_s)[0] for a in WHATIF_GRID])) for b in WHATIF_BASE_RATES)
        W["headline"] = OrderedDict((name, M.whatif_captures_needed(WHATIF_HEADLINE_BASE, a, c_s)[0]) for name, a in WHATIF_AUCS.items())
    # Every AUC actually measured, to mark on the model's axis.
    # Every AUC actually measured, to mark on the model's axis: age only and all signals in E3,
    # the frame signals for frame staleness, and age, mobility and all signals of any E5 data.
    marks = []
    for exp in POLICY_EXPS:
        for key, label in (("Age", "age only"), ("All", "all signals")):
            c = PS[exp]["curves"][key]
            if c is not None and c["auc"] is not None:
                marks.append(("%s, %s" % (label, exp), c["auc"]))
        if exp == "E5" and PS[exp]["mobility_auc"] is not None:
            marks.append(("mobility only, %s" % exp, PS[exp]["mobility_auc"]))
    if FR["auc"] is not None:
        marks.append(("frame signals, frame staleness", FR["auc"]))
    W["marks"] = marks
    return W


# ----------------------------------------------------------------------------- T2

def load_power_runs(dirs):
    runs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                with open(p) as f:
                    r = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(r, dict) and r.get("tool") == "power_measure.py" and isinstance(r.get("all_rails"), dict):
                r["_path"] = p
                runs.append(r)
    return runs


def power_run_is_synthetic(r):
    return bool(r.get("synthetic")) or str(r.get("label", "")).upper().startswith("SYNTH")


def compute_cost(events, queries, captures, regs, ds, power_runs):
    C = {"warnings": []}
    C["reg"] = {s: [r["latency"].get(k) for r in regs if r["latency"].get(k) is not None] for s, k in STAGES.items() if s != "Total"}
    C["reg"]["Total"] = [sum(r["latency"][k] for k in ("convertMs", "detectMs", "embedMs")) for r in regs
                         if all(r["latency"].get(k) is not None for k in ("convertMs", "detectMs", "embedMs"))]
    C["ver"] = {s: [c["latency"].get(k) for c in captures if c["latency"].get(k) is not None] for s, k in STAGES.items()}
    C["calls"] = [c["inference_calls"] for c in captures if c["inference_calls"] is not None]
    C["storage"] = [r["record_bytes"] for r in regs if r["record_bytes"] is not None]
    used = {e["_folder"] for e in events}
    C["map_bytes"] = [(ds.folders[i].path, ds.folders[i].current_map_bytes) for i in sorted(used)
                      if ds.folders[i].current_map_bytes is not None]
    # From reaching the item (NEAR) to the result on screen. The app takes msSinceNear when Verify
    # is pressed, so the processing time of the capture (latency.totalMs) is added.
    C["capture_s"] = [(q["first_capture"]["ms_since_near"] + q["first_capture"]["latency"]["totalMs"]) / 1000.0 for q in queries
                      if q["first_capture"] is not None and q["first_capture"]["ms_since_near"] is not None
                      and q["first_capture"]["ms_since_near"] >= 0 and q["first_capture"]["latency"]["totalMs"] is not None]

    usable = {"tracking_only": [], "tracking_verify": []}
    for r in power_runs:
        if r.get("mode") not in usable:
            continue
        # A run counts only when the FindBack app was confirmed in the foreground before and after
        # it. "Unknown" is not enough: a phone that is dozing reports no foreground app.
        fg = [(r.get(k) or {}).get("foreground") for k in ("state_before", "state_after")]
        if not all(f is not None and APP_PACKAGE in f for f in fg):
            C["warnings"].append("power run %s left out: the foreground app before/after was %s, not %s"
                                 % (os.path.basename(r["_path"]), fg, APP_PACKAGE))
            continue
        usable[r["mode"]].append(r)
    C["power"] = {}
    for mode, runs in usable.items():
        C["power"][mode] = {
            "n": len(runs),
            "mw": float(np.mean([r["all_rails"]["mean_mw"] for r in runs])) if runs else None,
            "seconds": [r.get("duration_s") for r in runs],
            "range": (min(r["all_rails"]["mean_mw"] for r in runs), max(r["all_rails"]["mean_mw"] for r in runs)) if runs else None,
            "groups": {g: float(np.mean([(r.get("groups") or {}).get(g, {}).get("mean_mw", 0.0) for r in runs])) for g in POWER_GROUPS} if runs else None,
        }
    # Energy per verification is a small difference between two modes. It is reported only when
    # the data can tell the modes apart: at least MIN_POWER_RUNS runs of each, and every verifying
    # run above every tracking-only run (an exact one-sided rank test; p = 1/20 for 3 against 3).
    C["energy_j"], C["energy_n"], C["energy_state"] = None, 0, None
    track, ver = usable["tracking_only"], usable["tracking_verify"]
    if track or ver:
        if min(len(track), len(ver)) < MIN_POWER_RUNS:
            C["energy_state"] = "n too small"
            C["warnings"].append("energy per verification needs %d runs per mode (have %d tracking_only, %d tracking_verify)"
                                 % (MIN_POWER_RUNS, len(track), len(ver)))
        elif C["power"]["tracking_verify"]["range"][0] <= C["power"]["tracking_only"]["range"][1]:
            C["energy_state"] = "not resolved"
            C["warnings"].append("energy per verification is not resolved: verifying runs %.0f to %.0f mW overlap tracking-only runs "
                                 "%.0f to %.0f mW" % (C["power"]["tracking_verify"]["range"] + C["power"]["tracking_only"]["range"]))
        else:
            p_track = C["power"]["tracking_only"]["mw"] / 1000.0
            # Only captures logged under exp=E4 count: a session of another experiment (or another
            # phone) may overlap a run in time.
            verify_t = [e.get("t") for e in events if e.get("type") == "verify" and L.num(e.get("t")) is not None
                        and L.tag_exp(e["_tag"]) == "E4"]
            per_run, n_total = [], 0
            for r in ver:
                n_log = 0
                if r.get("phone_start_ms") is not None and r.get("phone_end_ms") is not None:
                    n_log = sum(r["phone_start_ms"] <= t <= r["phone_end_ms"] for t in verify_t)
                n_manual = r.get("verifications")
                if n_log and n_manual and n_log != n_manual:
                    C["warnings"].append("power run %s: %d verify events in the log during the run, %d entered by hand; the log count is used"
                                         % (os.path.basename(r["_path"]), n_log, n_manual))
                n_ver = n_log or n_manual
                if not n_ver:
                    C["warnings"].append("power run %s: no verification count (no verify event tagged exp=E4 inside the run, none entered)"
                                         % os.path.basename(r["_path"]))
                    continue
                per_run.append((r["all_rails"]["energy_j"] - p_track * r["duration_s"]) / n_ver)
                n_total += n_ver
            if len(per_run) == len(ver):
                C["energy_j"], C["energy_n"] = float(np.mean(per_run)), n_total
                C["energy_range"] = (min(per_run), max(per_run))
    return C


# ----------------------------------------------------------------------------- macros

def fill_macros(N, D, E1, E2, PS, C, FR, NAV, W):
    have = D["have_data"]
    tot = D["total"]
    N.set("DataSites", t_int(len(D["sites"]), have and bool(D["sites"])), len(D["sites"]))
    N.set("DataSessions", t_int(len(tot["sessions"]), have), len(tot["sessions"]))
    N.set("DataRooms", t_int(len(tot["rooms"]), have), len(tot["rooms"]))
    N.set("DataItems", t_int(len(tot["items"]), have), len(tot["items"]))
    N.set("DataRegistrations", t_int(tot["registrations"], have), tot["registrations"])
    N.set("DataQueries", t_int(sum(tot["truth"].values()), have), sum(tot["truth"].values()))
    for t in L.GROUND_TRUTHS:
        N.set("Data" + t.capitalize(), t_int(tot["truth"][t], have), tot["truth"][t])
    N.set("DataCaptures", t_int(tot["captures"], have), tot["captures"])
    N.set("DataBenchmarkCaptures", t_int(tot["labelled"], have), tot["labelled"])

    # E1
    h1 = E1["n_queries"] > 0
    N.set("EOneQueries", t_int(E1["n_queries"], h1), E1["n_queries"])
    N.set("EOneMeasured", t_int(len(E1["rows"]), h1), len(E1["rows"]))
    for g in E1_GROUPS:
        v = E1["groups"][g]
        N.set("EOne%sN" % g, t_int(len(v), h1), len(v))
        med, p90 = M.median(v), M.quantile(v, 0.9, M.MIN_N_P90)
        N.set("EOne%sMedianCm" % g, or_small(t_num(med * 100 if med is not None else None, 0), len(v), M.MIN_N_MEDIAN), med)
        N.set("EOne%sPNinetyCm" % g, or_small(t_num(p90 * 100 if p90 is not None else None, 0), len(v), M.MIN_N_P90), p90)
    N.set("EOneMapRoomUnknownN", t_int(len(E1["groups"]["MapUnknown"]), h1), len(E1["groups"]["MapUnknown"]))
    N.set("EOneWalkN", t_int(len(E1["walk"]), h1), len(E1["walk"]))
    N.set("EOneWalkSpearman", or_small(t_num(E1["rho"], 2), len(E1["walk"]), M.MIN_N_CURVE), E1["rho"])
    hf = E1["n_fixes"] > 0
    for t, key in (("Image", "IMAGE"), ("Spot", "SPOT")):
        m, deg = E1["jumps"][key]["m"], E1["jumps"][key]["deg"]
        N.set("EOneJump%sN" % t, t_int(len(m), hf), len(m))
        med = M.median(m)
        N.set("EOneJump%sMedianCm" % t, or_small(t_num(med * 100 if med is not None else None, 1), len(m), M.MIN_N_MEDIAN), med)
        N.set("EOneJump%sMaxCm" % t, t_num(max(m) * 100, 1) if m else None, max(m) if m else None)
        medd = M.median(deg)
        N.set("EOneJump%sMedianDeg" % t, or_small(t_num(medd, 2), len(deg), M.MIN_N_MEDIAN), medd)
        N.set("EOneJump%sMaxDeg" % t, t_num(max(deg), 2) if deg else None, max(deg) if deg else None)
    N.set("EOneDisagreements", t_int(E1["n_asked"], hf), E1["n_asked"])
    N.set("EOneDisagreementsAccepted", t_int(E1["n_accepted"], hf), E1["n_accepted"])
    N.set("EOneRoomNamedRate", t_rate(E1["n_named"], E1["n_queries"]), [E1["n_named"], E1["n_queries"]])

    # Guidance across rooms (E1 and E3)
    N.set("NavCrossRoomQueries", t_int(NAV["n_cross"], NAV["n_usable"] > 0), NAV["n_cross"])
    N.set("NavCrossRoomNearRate", t_rate(NAV["n_reached"], NAV["n_judged"]), [NAV["n_reached"], NAV["n_judged"]])
    med = M.median(NAV["seconds"])
    N.set("NavCrossRoomSecondsMedian", or_small(t_num(med, 1), len(NAV["seconds"]), M.MIN_N_MEDIAN), med)
    N.set("NavCrossRoomSecondsN", t_int(len(NAV["seconds"]), NAV["n_judged"] > 0), len(NAV["seconds"]))

    # Measured frame staleness (E1 and E3)
    if FR["n_rows"]:
        N.set("FrameStaleProtocol", t_text(FR["protocol"]), FR["protocol"])
        N.set("FrameStaleTrainQueries", t_int(FR["n_train"]), FR["n_train"])
        N.set("FrameStaleQueries", t_int(FR["n"]), FR["n"])
        N.set("FrameStaleSessions", t_int(FR["n_sessions"]), FR["n_sessions"])
        N.set("FrameStaleRate", t_rate(FR["n_stale"], FR["n"]), [FR["n_stale"], FR["n"]])
        small = FR["auc"] is None and "n too small" in (FR["reason"] or "")
        N.set("FrameStaleAuc", NSMALL if small else t_num(FR["auc"], 3), FR["auc"])
        N.set("FrameStaleAucCI", NSMALL if small else t_ci(FR["boot"], digits=3),
              None if FR["boot"] is None else [FR["boot"]["lo"], FR["boot"]["hi"]])

    # E2
    h2 = E2["n_labelled"] > 0
    if h2:
        N.set("ETwoProtocol", t_text(E2["protocol"]), E2["protocol"])
        tau_value, tau_text = E2["tau"].describe(E2["ev"])
        N.set("ETwoThreshold", t_num(tau_value, 2), tau_value)
        N.set("ETwoThresholdSource", t_text(tau_text), tau_text)
        N.set("ETwoMargin", t_num(E2["margin"], 2), E2["margin"])
        N.set("ETwoClusters", t_int(E2["n_clusters"]), E2["n_clusters"])
        N.set("ETwoClusterUnit", t_text(E2["cluster_unit"]), E2["cluster_unit"])
    N.set("ETwoCaptures", t_int(E2["n_labelled"], h2), E2["n_labelled"])
    N.set("ETwoDevCaptures", t_int(E2["n_dev"], h2), E2["n_dev"])
    N.set("ETwoEvalCaptures", t_int(len(E2["ev"]), h2), len(E2["ev"]))
    N.set("ETwoUnlabelled", t_int(E2["n_unlabelled"], E2["n_events"] > 0), E2["n_unlabelled"])
    for n, lab in [("Target", "target")] + list(NEG_TYPES.items()):
        N.set("ETwo%sN" % n, t_int(E2["n_by_label"][lab], h2), E2["n_by_label"][lab])
    for n in list(NEG_TYPES) + ["All"]:
        r = E2["roc"][n]
        small = h2 and min(r["n_pos"], r["n_neg"]) < M.MIN_ROC_PER_CLASS
        N.set("ETwoAuc" + n, NSMALL if small else t_num(r["auc"], 3), r["auc"])
        N.set("ETwoAuc%sCI" % n, NSMALL if small else t_ci(r["boot"], digits=3),
              None if r["boot"] is None else [r["boot"]["lo"], r["boot"]["hi"]])
    small_all = h2 and min(E2["n_all_pos"], E2["n_all_neg"]) < M.MIN_ROC_PER_CLASS
    N.set("ETwoAucAllLabelled", NSMALL if small_all else t_num(E2["auc_all_labelled"], 3), E2["auc_all_labelled"])
    for rule in ("Thr", "Sub", "Mar"):
        if E2[rule] is None:
            continue
        for k, (a, b) in E2[rule].items():
            N.set("ETwo%s%sRate" % (rule, k), t_rate(a, b), [a, b])
    N.set("ETwoMarginSubsetN", t_int(E2["n_margin_subset"], h2), E2["n_margin_subset"])
    for cond, r in E2["conditions"].items():
        N.set("ETwoTpr%sRate" % cond.capitalize(), t_rate(*r["thr"]), list(r["thr"]))

    # E3, E5
    for exp in POLICY_EXPS:
        P, x = PS[exp], EXP_WORD[exp]
        hu, hp = P["n_usable"] > 0, P["n"] > 0
        N.set(x + "Queries", t_int(P["n_usable"], hu), P["n_usable"])
        N.set(x + "Paired", t_int(P["n"], hu), P["n"])
        N.set(x + "Sessions", t_int(P["n_sessions"], hp), P["n_sessions"])
        N.set(x + "NoCapture", t_int(P["n_no_capture"], hu), P["n_no_capture"])
        N.set(x + "StaleRate", t_rate(P["n_stale"], P["n"]), [P["n_stale"], P["n"]])
        if hp:
            N.set(x + "Protocol", t_text(P["protocol"]), P["protocol"])
            N.set(x + "TrainQueries", t_int(P["n_train"]), P["n_train"])
            N.set(x + "Threshold", t_num(P["tau_value"], 2), P["tau_value"])
            N.set(x + "ThresholdSource", t_text(P["tau_text"]), P["tau_text"])
        for p in FIXED_POLICIES:
            r = P["policies"].get(p)
            if r is None:
                continue
            c, n = r["counts"], r["counts"]["n"]
            N.set(x + p + "Captures", t_num(r["rates"]["captures_per_query"], 2), r["rates"]["captures_per_query"])
            N.set(x + p + "Calls", t_num(r["rates"].get("calls_per_query"), 2), r["rates"].get("calls_per_query"))
            N.set(x + p + "WrongLeadRate", t_rate(c["wrong_lead"], n), [c["wrong_lead"], n])
            N.set(x + p + "LostRate", t_rate(c["lost_retrieval"], n), [c["lost_retrieval"], n])
            N.set(x + p + "LostOfAtMarkerRate", t_rate(c["lost_retrieval"], n - P["n_stale"]), [c["lost_retrieval"], n - P["n_stale"]])
            N.set(x + p + "WrongLeadCI", t_ci(r["boot"], 1, 100, 0, "\\%"), [float(r["boot"]["lo"][1]), float(r["boot"]["hi"][1])])
            N.set(x + p + "LostCI", t_ci(r["boot"], 2, 100, 0, "\\%"), [float(r["boot"]["lo"][2]), float(r["boot"]["hi"][2])])
        if P["random"] is not None:
            for b in BUDGETS:
                N.set(x + "Random" + b + "WrongLead", t_pct(P["random"][b]["wrong_lead_rate"]), P["random"][b]["wrong_lead_rate"])
                N.set(x + "Random" + b + "Lost", t_pct(P["random"][b]["lost_retrieval_rate"]), P["random"][b]["lost_retrieval_rate"])
        for key in ("Age", "World", "Frame"):
            cv = P["curves"][key]
            if cv is None:
                if hp and "n too small" in (P["curve_reason"][key] or ""):
                    N.set(x + key + "Auc", NSMALL)
                    N.set(x + key + "AucCI", NSMALL)
                continue
            N.set(x + key + "Auc", NSMALL if cv["auc_small"] else t_num(cv["auc"], 3), cv["auc"])
            N.set(x + key + "AucCI", NSMALL if cv["auc_small"] else t_ci(cv["auc_boot"], digits=3), [cv["auc_boot"]["lo"], cv["auc_boot"]["hi"]])
        op = P["op"]
        if op is None:
            if hp and "n too small" in (P["op_reason"] or ""):
                for name in ("Captures", "WrongLeadRate", "WrongLeadCI", "LostRate", "LostCI", "DevCaptures"):
                    N.set(x + "FrameOp" + name, NSMALL)
        else:
            c = op["counts"]
            N.set(x + "FrameOpCaptures", t_num(op["rates"]["captures_per_query"], 2), op["rates"]["captures_per_query"])
            N.set(x + "FrameOpWrongLeadRate", t_rate(c["wrong_lead"], c["n"]), [c["wrong_lead"], c["n"]])
            N.set(x + "FrameOpLostRate", t_rate(c["lost_retrieval"], c["n"]), [c["lost_retrieval"], c["n"]])
            N.set(x + "FrameOpWrongLeadCI", t_ci(op["boot"], 1, 100, 0, "\\%"))
            N.set(x + "FrameOpLostCI", t_ci(op["boot"], 2, 100, 0, "\\%"))
            N.set(x + "FrameOpDevCaptures", t_num(op["dev_captures"], 2), op["dev_captures"])
            N.set(x + "FrameOpDevQueries", t_int(op["dev_n"]), op["dev_n"])
        Lr = P["learned"]
        if Lr is None:
            if hp and P["learned_reason"] and "n too small" in P["learned_reason"]:
                for name in ["RiskAuc", "RiskAucCI", "LearnedAtHeuristicWrongLead"] + [
                        "Learned" + b + s for b in BUDGETS for s in ("WrongLead", "WrongLeadCI", "Lost", "LostCI")]:
                    N.set(x + name, NSMALL)
            continue
        N.set(x + "RiskAuc", NSMALL if Lr["auc_small"] else t_num(Lr["auc"], 3), Lr["auc"])
        N.set(x + "RiskAucCI", NSMALL if Lr["auc_small"] else t_ci(Lr["auc_boot"], digits=3), [Lr["auc_boot"]["lo"], Lr["auc_boot"]["hi"]])
        for b, v in BUDGETS.items():
            i = int(np.argmin(np.abs(GRID - v)))
            wl, lost = M.curve_at(Lr["x"], Lr["wl"], v), M.curve_at(Lr["x"], Lr["lost"], v)
            N.set(x + "Learned" + b + "WrongLead", t_pct(wl), wl)
            N.set(x + "Learned" + b + "Lost", t_pct(lost), lost)
            band = {"ok": Lr["boot"]["ok"], "lo": Lr["boot"]["lo"], "hi": Lr["boot"]["hi"]}
            N.set(x + "Learned" + b + "WrongLeadCI", t_ci(band, i, 100, 0, "\\%"))
            N.set(x + "Learned" + b + "LostCI", t_ci(band, GRID.size + i, 100, 0, "\\%"))
        h = P["policies"].get("Heuristic")
        if h is not None:
            v = M.curve_at(Lr["x"], Lr["wl"], h["rates"]["captures_per_query"])
            N.set(x + "LearnedAtHeuristicWrongLead", t_pct(v), v)

    # Measured verification behaviour (E3, all paired queries)
    cf = PS["E3"]["confirm"]
    N.set("EThreeConfirmQueries", t_int(cf["n"], cf["n"] > 0), cf["n"])
    N.set("EThreeConfirmSessions", t_int(cf["n_sessions"], cf["n"] > 0), cf["n_sessions"])
    for i, key in enumerate(CONFIRM_CLASSES):
        k, n = cf["counts"][key]
        N.set("EThreeConfirm%sRate" % key, t_rate(k, n), [k, n])
        if n:
            N.set("EThreeConfirm%sCI" % key, t_ci(cf["boot"], i, 100, 0, "\\%"))

    # The what-if model. Its two constants are always defined; its outputs need the measured shares.
    N.set("WhatIfBaseRatePct", "%.0f" % (100 * WHATIF_HEADLINE_BASE))
    N.set("WhatIfTolerancePoints", "%.0f" % (100 * M.WL_TOLERANCE))
    for name in WHATIF_AUCS:
        if W["state"] == "n too small":
            N.set("WhatIfCapturesAuc" + name, NSMALL)
        elif W["headline"] is not None:
            N.set("WhatIfCapturesAuc" + name, t_num(W["headline"][name], 2), W["headline"][name])

    # T2
    for s, key in (("Reg", "reg"), ("Ver", "ver")):
        n_total = len(C[key]["Total"])
        for st in STAGES:
            v = C[key][st]
            med, p95 = M.median(v), M.quantile(v, 0.95, M.MIN_N_P95)
            N.set("Cost%s%sMedianMs" % (s, st), or_small(t_num(med, 0), len(v), M.MIN_N_MEDIAN), med)
            N.set("Cost%s%sPNinetyFiveMs" % (s, st), or_small(t_num(p95, 0), len(v), M.MIN_N_P95), p95)
        N.set("Cost%sN" % s, t_int(n_total, n_total > 0), n_total)
    med = M.median(C["calls"])
    N.set("CostVerCallsMedian", or_small(t_num(med, 0), len(C["calls"]), M.MIN_N_MEDIAN), med)
    med = M.median(C["storage"])
    N.set("CostStorageItemKB", or_small(t_num(med / 1000.0 if med is not None else None, 0), len(C["storage"]), M.MIN_N_MEDIAN), med)
    N.set("CostStorageN", t_int(len(C["storage"]), bool(C["storage"])), len(C["storage"]))
    if C["map_bytes"]:
        biggest = max(b for _, b in C["map_bytes"])
        N.set("CostMapFileKB", t_num(biggest / 1000.0, 0), biggest)
        N.set("CostMapFiles", t_int(len(C["map_bytes"])), len(C["map_bytes"]))
    v = C["capture_s"]
    med, p95 = M.median(v), M.quantile(v, 0.95, M.MIN_N_P95)
    N.set("CostCaptureSecondsMedian", or_small(t_num(med, 1), len(v), M.MIN_N_MEDIAN), med)
    N.set("CostCaptureSecondsPNinetyFive", or_small(t_num(p95, 1), len(v), M.MIN_N_P95), p95)
    N.set("CostCaptureSecondsN", t_int(len(v), bool(v)), len(v))
    for name, mode in (("Tracking", "tracking_only"), ("Verify", "tracking_verify")):
        p = C["power"][mode]
        N.set("CostPower%sMw" % name, t_num(p["mw"], 0), p["mw"])
        N.set("CostPower%sRuns" % name, t_int(p["n"], p["n"] > 0), p["n"])
        if p["range"]:
            N.set("CostPower%sRangeMw" % name, "%.0f--%.0f" % p["range"], list(p["range"]))
    unresolved = {"n too small": NSMALL, "not resolved": UNRESOLVED}.get(C["energy_state"])
    N.set("CostEnergyPerVerificationJ", unresolved or t_num(C["energy_j"], 2), C["energy_j"])
    N.set("CostEnergyVerifications", t_int(C["energy_n"], C["energy_n"] > 0), C["energy_n"])

    # ---- the paper's own names: the same values as bare numbers
    for paper, mine in (("NSites", "DataSites"), ("NRooms", "DataRooms"), ("NItems", "DataItems"),
                        ("NBenchCaptures", "ETwoCaptures"), ("NQueriesPaired", "EThreePaired"),
                        ("MarkerErrSessionMedianCm", "EOneSessionMedianCm"), ("MarkerErrMapMedianCm", "EOneMapAllMedianCm"),
                        ("MarkerErrMapPninetyCm", "EOneMapAllPNinetyCm"), ("RelockJumpMedianCm", "EOneJumpSpotMedianCm"),
                        ("RelockJumpMedianDeg", "EOneJumpSpotMedianDeg"), ("VerifyAUC", "ETwoAucAllLabelled"),
                        ("ThresholdTau", "ETwoThreshold"), ("RegisterLatencyMs", "CostRegTotalMedianMs"),
                        ("VerifyLatencyMs", "CostVerTotalMedianMs"), ("SecondsPerCapture", "CostCaptureSecondsMedian"),
                        ("CapturesSelective", "EThreeHeuristicCaptures")):
        N.copy(paper, mine)
    per = sorted(set(D["items_per_site"].values()))
    if per:
        N.set("NItemsPerSite", "%d" % per[0] if len(per) == 1 else "%d--%d" % (per[0], per[-1]), per)
    N.set("CrossRoomNearPct", t_bare_pct(NAV["n_reached"], NAV["n_judged"]))
    N.copy("CrossRoomSecondsToNear", "NavCrossRoomSecondsMedian")
    if FR["n_rows"]:
        N.set("FrameStalePct", t_bare_pct(FR["n_stale"], FR["n"]))
    N.copy("FrameStaleAUC", "FrameStaleAuc")
    k, n = cf["counts"]["AtMarker"]
    N.set("ConfirmAtMarkerPct", t_bare_pct(k, n))
    k, n = cf["counts"]["Stale"]
    N.set("ConfirmStalePct", t_bare_pct(k, n))
    thr, sub, mar = E2["Thr"], E2["Sub"], E2["Mar"]
    N.set("VerifyTPRPct", t_bare_pct(*thr["Tpr"]))
    N.set("VerifyFPRPct", t_bare_pct(*thr["FprAll"]))
    N.set("LookalikeFPRThreshPct", t_bare_pct(*(sub if sub is not None else thr)["FprLookalike"]))
    if mar is not None:
        N.set("LookalikeFPRMarginPct", t_bare_pct(*mar["FprLookalike"]))
    pol = PS["E3"]["policies"]
    for paper, name, key in (("WrongLeadMemoryPct", "Memory", "wrong_lead"), ("WrongLeadAlwaysPct", "Always", "wrong_lead"),
                             ("WrongLeadSelectivePct", "Heuristic", "wrong_lead")):
        if pol.get(name) is not None:
            N.set(paper, t_bare_pct(pol[name]["counts"][key], pol[name]["counts"]["n"]))
    if pol.get("Always") is not None:
        c = pol["Always"]["counts"]
        N.set("LostRetrievalAlwaysPct", t_bare_pct(c["lost_retrieval"], c["n"] - PS["E3"]["n_stale"]))
    if pol.get("Heuristic") is not None:
        c = pol["Heuristic"]["counts"]
        N.set("CaptureSavingPct", t_bare_pct(c["n"] - c["captures"], c["n"]))
    # The same three numbers under names that say what they are: the logged on-device rule.
    for paper, old in (("OnDeviceWrongLeadPct", "WrongLeadSelectivePct"), ("OnDeviceCaptures", "CapturesSelective"),
                       ("OnDeviceCaptureSavingPct", "CaptureSavingPct")):
        N.copy(paper, old)
    op = PS["E3"]["op"]
    if op is not None:
        c = op["counts"]
        N.set("FramePolicyWrongLeadPct", t_bare_pct(c["wrong_lead"], c["n"]))
        N.set("FramePolicyLostPct", t_bare_pct(c["lost_retrieval"], c["n"]))
        N.set("FramePolicyCaptureSavingPct", t_bare_pct(c["n"] - c["captures"], c["n"]))
        N.copy("FramePolicyCaptures", "EThreeFrameOpCaptures")
    elif "n too small" in (PS["E3"]["op_reason"] or "") and PS["E3"]["n"]:
        for name in ("FramePolicyWrongLeadPct", "FramePolicyLostPct", "FramePolicyCaptureSavingPct", "FramePolicyCaptures"):
            N.set(name, NSMALL)
    med = M.median(C["storage"])
    N.set("BytesPerItem", or_small(t_num(med, 0), len(C["storage"]), M.MIN_N_MEDIAN))
    N.set("EnergyPerVerifymJ", unresolved or t_num(C["energy_j"] * 1000.0 if C["energy_j"] is not None else None, 0))
    N.set("RPresentCm", "%.0f" % (M.R_PRESENT_M * 100))


# ----------------------------------------------------------------------------- tables

def tabular(spec, head, rows, note, float_env=None):
    """A booktabs tabular. With float_env=(caption, label) it is wrapped in a complete table float,
    which is what the paper inputs for T1 and T2; otherwise only the tabular is written."""
    out = ["% " + n for n in note]
    if float_env:
        out += ["\\begin{table}[t]", "\\caption{%s}" % float_env[0], "\\label{%s}" % float_env[1], "\\centering\\small",
                "\\setlength{\\tabcolsep}{4pt}"]
    out += ["\\begin{tabular}{%s}" % spec, "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if float_env:
        out.append("\\end{table}")
    return "\n".join(out) + "\n"


def build_tables(N, D, E1, E2, PS, C, banner):
    T = {}
    mark = "\\textcolor{red}{SYNTHETIC DATA.} " if N.synthetic else ""
    note = lambda *lines: list(banner) + list(lines) + ["Needs booktabs and xcolor. Generated; do not edit."]  # noqa: E731

    # T1 as the paper inputs it: a complete single-column float of totals.
    rows = [
        ["Homes / rooms", "%s / %s" % (N.get("DataSites"), N.get("DataRooms"))],
        ["Items", N.get("DataItems")],
        ["Registrations", N.get("DataRegistrations")],
        ["Queries with a ground truth", N.get("DataQueries")],
    ] + [["\\quad " + t, N.get("Data" + t.capitalize())] for t in L.GROUND_TRUTHS] + [
        ["Captures", N.get("DataCaptures")],
        ["Labelled benchmark captures", N.get("DataBenchmarkCaptures")],
        ["Paired queries replayed (E3)", N.get("EThreePaired")],
    ]
    T["table_dataset.tex"] = tabular("lr", ["", "Count"], rows, note(
        "T1 dataset, totals over all events logged under a study tag. A complete float: paper/findback.tex inputs it",
        "in place of its fallback table. The breakdown by experiment and site is table_dataset_by_experiment.tex."),
        float_env=(mark + "Dataset. Only sessions logged under a study tag are counted.", "tab:dataset"))

    rows = []
    for (exp, home), g in D["groups"].items():
        rows.append([t_text(exp), t_text(home)] + [N.cell("%d" % v) for v in (
            len(g["sessions"]), len(g["rooms"]), len(g["items"]), g["registrations"], g["truth"]["present"],
            g["truth"]["displaced"], g["truth"]["absent"], g["truth"]["replaced"], g["truth"]["skip"],
            g["captures"], g["labelled"])])
    if rows:
        rows.append("\\midrule")
        rows.append(["Total", N.get("DataSites")] + [N.get(k) for k in (
            "DataSessions", "DataRooms", "DataItems", "DataRegistrations", "DataPresent", "DataDisplaced", "DataAbsent",
            "DataReplaced", "DataSkip", "DataCaptures", "DataBenchmarkCaptures")])
    else:
        rows.append(["\\multicolumn{13}{c}{%s}" % N.cell(None)])
    T["table_dataset_by_experiment.tex"] = tabular(
        "llrrrrrrrrrrr",
        ["Exp.", "Site", "Sessions", "Rooms", "Items", "Reg.", "Present", "Displaced", "Absent", "Replaced", "Skip", "Captures", "Labelled"],
        rows, note("T1 detail (tabular only, two columns wide). One row per experiment (exp=) and site (home=).",
                   "Rooms and items are distinct per site; the Total row counts each room and item once and shows the number of sites."))

    rows = []
    for s, label in (("Reg", "Register"), ("Ver", "Verify")):
        for st in STAGES:
            rows.append(["%s: %s (ms)" % (label, st.lower()), N.get("Cost%s%sMedianMs" % (s, st)),
                         N.get("Cost%s%sPNinetyFiveMs" % (s, st)), N.get("Cost%sN" % s)])
        rows.append("\\midrule")
    rows += [
        ["Calls per verification", N.get("CostVerCallsMedian"), "--", N.get("CostVerN")],
        ["Storage per item (kB)", N.get("CostStorageItemKB"), "--", N.get("CostStorageN")],
        ["Map file, largest (kB)", N.get("CostMapFileKB"), "--", N.get("CostMapFiles")],
        ["Capture time (s)", N.get("CostCaptureSecondsMedian"), N.get("CostCaptureSecondsPNinetyFive"), N.get("CostCaptureSecondsN")],
        "\\midrule",
        ["Power, tracking (mW)", N.get("CostPowerTrackingMw"), "--", N.get("CostPowerTrackingRuns")],
        ["Power, verifying (mW)", N.get("CostPowerVerifyMw"), "--", N.get("CostPowerVerifyRuns")],
        ["Energy per verification (J)", N.get("CostEnergyPerVerificationJ"), "--", N.get("CostEnergyVerifications")],
    ]
    T["table_cost.tex"] = tabular("lrrr", ["Quantity", "Median", "95th pct.", "$n$"], rows, note(
        "T2 cost, a complete float. Latencies: median and 95th percentile over logged events; the register total is the sum of its stages.",
        "Capture time: verify.msSinceNear + latency.totalMs of the first capture of a query (from reaching the item to the result).",
        "Power rows: the Median column holds the mean over runs of the summed monitored rails (not the phone's total power); n = runs.",
        "Energy per verification = (rail energy of a verifying run - tracking power x its duration) / verifications in it; n = verifications.",
        "It is given only with 3 runs per mode and every verifying run above every tracking-only run; otherwise 'n too small' or 'not resolved'."),
        float_env=(mark + "Cost. Power is the sum of the monitored rails, averaged over runs.", "tab:cost"))

    rows = []
    for g in POWER_GROUPS:
        cells = []
        for mode in ("tracking_only", "tracking_verify"):
            p = C["power"][mode]
            cells.append(N.cell(None if not p["n"] else "%.0f" % p["groups"][g]))
        rows.append([g] + cells)
    rows.append("\\midrule")
    rows.append(["all monitored rails", N.get("CostPowerTrackingMw"), N.get("CostPowerVerifyMw")])
    rows.append(["runs", N.get("CostPowerTrackingRuns"), N.get("CostPowerVerifyRuns")])
    T["table_power_by_group.tex"] = tabular("lrr", ["Rails (mW)", "Tracking only", "Tracking and verification"], rows, note(
        "T2 detail: mean rail power by group. Groups are sums of rails chosen by name (see analysis/power_measure.py)."))

    short = {"Session": "Same-session anchor", "MapSame": "Map, fix point's room", "MapOther": "Map, another room", "MapAll": "Map, all rooms"}
    rows = [[short[g], N.get("EOne%sN" % g), N.get("EOne%sMedianCm" % g), N.get("EOne%sPNinetyCm" % g)] for g in E1_GROUPS]
    T["table_marker_error.tex"] = tabular("lrrr", ["Position from", "$n$", "Median (cm)", "90th pct. (cm)"], rows, note(
        "F1 table: marker error of present items by position source, and by whether the item is in the fix point's room."))
    rows = [[label, N.get("EOneJump%sN" % t), N.get("EOneJump%sMedianCm" % t), N.get("EOneJump%sMaxCm" % t),
             N.get("EOneJump%sMedianDeg" % t), N.get("EOneJump%sMaxDeg" % t)] for t, label in (("Image", "Picture"), ("Spot", "Spot"))]
    T["table_refix_jumps.tex"] = tabular("lrrrrr", ["Fix point", "$n$", "Median (cm)", "Max (cm)", "Median ($^\\circ$)", "Max ($^\\circ$)"], rows, note(
        "F1 table: map jump on a re-fix while located (map_fix with kind refresh or manual, applied), by fix type."))

    rows = []
    for rule, label in (("Thr", "Threshold, all captures"), ("Sub", "Threshold, captures with embeddings"),
                        ("Mar", "Threshold and margin, captures with embeddings")):
        rows.append([label, N.get("ETwo%sTprRate" % rule)] + [N.get("ETwo%sFpr%sRate" % (rule, n)) for n in list(NEG_TYPES) + ["All"]])
    T["table_rules.tex"] = tabular("lrrrrr", ["Rule", "TPR (target)", "FPR, empty", "FPR, other", "FPR, look-alike", "FPR, all"], rows, note(
        "F2 table: the two decision rules at the threshold and margin of results.tex."))

    for exp in POLICY_EXPS:
        P, x = PS[exp], EXP_WORD[exp]
        rows = []
        for p, label in FIXED_POLICIES.items():
            rows.append([label, N.get(x + p + "Captures"), N.get(x + p + "Calls"), N.get(x + p + "WrongLeadRate"),
                         N.get(x + p + "WrongLeadCI"), N.get(x + p + "LostRate"), N.get(x + p + "LostCI")])
        rows.append("\\midrule")
        rows.append(["Frame signals, point chosen on dev", N.get(x + "FrameOpCaptures"), "--", N.get(x + "FrameOpWrongLeadRate"),
                     N.get(x + "FrameOpWrongLeadCI"), N.get(x + "FrameOpLostRate"), N.get(x + "FrameOpLostCI")])
        rows.append("\\midrule")
        for b, v in BUDGETS.items():
            rows.append(["Random, $p = %.2f$" % v, "%.2f" % v, "--", N.get(x + "Random" + b + "WrongLead"), "--",
                         N.get(x + "Random" + b + "Lost"), "--"])
            rows.append(["All signals, %.2f captures per query" % v, "%.2f" % v, "--", N.get(x + "Learned" + b + "WrongLead"),
                         N.get(x + "Learned" + b + "WrongLeadCI"), N.get(x + "Learned" + b + "Lost"), N.get(x + "Learned" + b + "LostCI")])
        T["table_policies_%s.tex" % exp] = tabular(
            "lrrrrrr", ["Policy", "Captures/query", "Calls/query", "Wrong leads", "95\\% CI", "Lost retrievals", "95\\% CI"], rows,
            note("F3 table, %s: policy replay on the paired queries. Rates are per replayed query, with counts." % exp,
                 "The on-device rule is the app's logged decision (query_start.policyVerify).",
                 "Frame signals, point chosen on dev: the fewest dev captures whose dev wrong-lead rate is within %d points of always verifying."
                 % round(100 * M.WL_TOLERANCE),
                 "Intervals: percentile bootstrap over sessions. Random and fixed-budget rows are expectations, so they have no counts."))

        # Ablation: age only, world signals only, frame signals only, all signals.
        rows = []
        for key in ("Age", "World", "Frame", "All"):
            cv = P["curves"][key]
            auc_name = x + ("RiskAuc" if key == "All" else key + "Auc")
            if cv is None:
                missing = N.cell(NSMALL if (P["n"] and "too small" in (P["curve_reason"][key] or "")) else None)
                rows.append([CURVES[key]["label"], missing, missing, missing])
            else:
                rows.append([CURVES[key]["label"], N.get(auc_name), N.get(auc_name + "CI"), N.cell("%.3f" % cv["area"])])
        T["table_ablation_%s.tex" % exp] = tabular(
            "lrrr", ["Signals", "AUC", "95\\% CI", "Area"], rows,
            note("F3 ablation, %s: which signals the capture decision may use. Same queries and protocol in every row." % exp,
                 "Age only is a threshold on the logged memory age, oldest first; nothing is fitted. The other rows are the learned risk score.",
                 "AUC: the score against 'not at the marker', out of sample, with a 95% bootstrap interval over sessions.",
                 "Area: area under the wrong-lead rate against captures per query (0 to 1); smaller is better; capturing at random gives",
                 "the mean of the memory-only and always-verify wrong-lead rates."))

    # Measured verification behaviour (E3).
    rows = []
    for i, (key, doc) in enumerate(CONFIRM_CLASSES.items()):
        if i == 2:
            rows.append("\\midrule")
        label = doc if i < 2 else "\\quad " + doc
        rows.append([label, N.get("EThreeConfirm%sRate" % key), N.get("EThreeConfirm%sCI" % key)])
    T["table_confirm_E3.tex"] = tabular(
        "lrr", ["Where the item really was", "Captures that confirm", "95\\% CI"], rows,
        note("Measured verification behaviour, E3: the share of captures that confirm, on all paired E3 queries (both splits).",
             "The four indented classes together are 'not at the marker'. Intervals: percentile bootstrap over sessions.",
             "The first two rows are c_p and c_s of the what-if model."))
    return T


# ----------------------------------------------------------------------------- figures

COL_W = 3.33  # ACM single column, inches
INK, MUTED, ALERT = "#000000", "#7F7F7F", "#C00000"
# One colour, marker and line style per policy, the same in every figure. Colours are Okabe-Ito;
# marker and line style carry the identity in greyscale.
POLICY_STYLE = OrderedDict([
    ("Memory", {"label": "Memory only", "color": "#000000", "marker": "s"}),
    ("Always", {"label": "Always verify", "color": "#0072B2", "marker": "o"}),
    ("Heuristic", {"label": "On-device rule (logged)", "color": "#D55E00", "marker": "^"}),
    ("Oracle", {"label": "Oracle", "color": "#009E73", "marker": "D"}),
    ("Random", {"label": "Random capture", "color": "#7F7F7F", "ls": ":"}),
    ("All", {"label": "All signals (learned)", "color": "#CC79A7", "ls": "-"}),
    ("Frame", {"label": "Frame signals only (learned)", "color": "#56B4E9", "ls": "--"}),
    ("Age", {"label": "Age-only threshold", "color": "#E69F00", "ls": "-."}),
    ("FrameOp", {"label": "Frame signals, point chosen on dev", "color": "#56B4E9", "marker": "P"}),
])
BAND_CURVE = "Frame"  # the one swept curve drawn with its 95% band; the others' intervals are in the ablation table
E1_STYLE = OrderedDict([
    ("Session", {"label": "Same-session anchor", "color": "#000000", "ls": "-", "marker": "s"}),
    ("MapSame", {"label": "Map, fix point's room", "color": "#0072B2", "ls": "--", "marker": "o"}),
    ("MapOther", {"label": "Map, another room", "color": "#D55E00", "ls": "-.", "marker": "^"}),
    ("MapUnknown", {"label": "Map, room unknown", "color": "#7F7F7F", "ls": ":", "marker": "x"}),
])
NEG_STYLE = OrderedDict([
    ("Empty", {"label": "nothing there", "color": "#0072B2", "ls": "-"}),
    ("Other", {"label": "another object", "color": "#009E73", "ls": "--"}),
    ("Lookalike", {"label": "a look-alike", "color": "#D55E00", "ls": "-."}),
])
RULE_STYLE = {"thr": {"label": "Threshold", "color": "#0072B2", "hatch": None},
              "mar": {"label": "Threshold and margin", "color": "#E69F00", "hatch": "///"}}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "lines.linewidth": 1.1, "lines.markersize": 4, "legend.frameon": False, "legend.handlelength": 2.4,
        "figure.constrained_layout.use": True, "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5,
    })
    return plt


def note(ax, text):
    """A message inside an axes that has nothing (or too little) to draw."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=ALERT, fontsize=8, gid="fb-note")


def watermark(fig):
    """Mark a figure as synthetic: once across every axes and once along the top edge."""
    for ax in fig.axes:
        ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, ha="center", va="center", rotation=25, fontsize=30,
                fontweight="bold", color=ALERT, alpha=0.35, zorder=1000, gid="fb-synthetic")
    # Along the right edge: legends sit above or below the axes, never there. The tight bounding
    # box keeps text that is outside the figure.
    fig.text(1.01, 0.5, "SYNTHETIC DATA: not a result", rotation=270, ha="left", va="center", fontsize=8, fontweight="bold",
             color=ALERT, gid="fb-synthetic")


def fig_e1_cdf(plt, E1):
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.set_xlabel("Marker error (cm)")
    ax.set_ylabel("Cumulative share of measurements")
    ax.set_ylim(0, 1.02)
    if not E1["rows"]:
        note(ax, "TBD: no marker-error measurements yet")
        ax.set_xlim(0, 100)
        return fig
    small, xmax = [], 0.0
    for g, st in E1_STYLE.items():
        v = np.sort(np.array(E1["groups"][g], float)) * 100.0
        if v.size == 0:
            continue
        if v.size < M.MIN_N_CURVE:
            small.append("%s: n too small (n = %d)" % (st["label"], v.size))
            continue
        ax.step(np.r_[0.0, v], np.r_[0.0, np.arange(1, v.size + 1) / v.size], where="post", color=st["color"], ls=st["ls"],
                label="%s (n = %d)" % (st["label"], v.size))
        xmax = max(xmax, float(v[-1]))
    ax.axvline(M.R_PRESENT_M * 100, color=MUTED, lw=0.5, ls=":")
    ax.set_xlim(0, max(xmax * 1.05, M.R_PRESENT_M * 100 * 1.3))
    ax.text(M.R_PRESENT_M * 100, 1.01, "%.0f cm" % (M.R_PRESENT_M * 100), transform=ax.get_xaxis_transform(), color=MUTED,
            fontsize=6.5, ha="center", va="bottom")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")
    if small:
        ax.text(0.98, 0.45 if ax.get_legend() else 0.5, "\n".join(small), transform=ax.transAxes, ha="right", va="center",
                color=ALERT, fontsize=6.5, gid="fb-note")
    return fig


def fig_e1_walk(plt, E1):
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.set_xlabel("Walked since the last map fix (m)")
    ax.set_ylabel("Marker error (cm)")
    walk = E1["walk"]
    if not walk:
        note(ax, "TBD: no map-frame measurements yet")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 100)
        return fig
    if len(walk) < M.MIN_N_CURVE:
        note(ax, "n too small (n = %d)" % len(walk))
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 100)
        return fig
    for g in ("MapSame", "MapOther", "MapUnknown"):
        pts = [(m, e * 100) for m, e, grp in walk if grp == g]
        if pts:
            st = E1_STYLE[g]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=10, marker=st["marker"], linewidths=0.6,
                       facecolors="none" if st["marker"] != "x" else st["color"], edgecolors=st["color"],
                       label="%s (n = %d)" % (st["label"], len(pts)))
    ax.axhline(M.R_PRESENT_M * 100, color=MUTED, lw=0.5, ls=":")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    return fig


def fig_e2_roc(plt, E2):
    fig, ax = plt.subplots(figsize=(COL_W, 3.7))
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (item at the marker)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.5, ls=":")
    if E2["n_labelled"] == 0:
        note(ax, "TBD: no labelled captures yet")
        return fig
    small = []
    for n, st in NEG_STYLE.items():
        r = E2["roc"][n]
        if r["fpr"] is None:
            small.append("%s: n too small (%d target, %d %s)" % (st["label"], r["n_pos"], r["n_neg"], NEG_TYPES[n]))
            continue
        # Straight segments between the operating points: where a positive and a negative tie,
        # the segment is diagonal, which is what the AUC (ties count one half) integrates.
        ax.plot(r["fpr"], r["tpr"], color=st["color"], ls=st["ls"],
                label="%s (AUC %.2f, n = %d)" % (st["label"], r["auc"], r["n_neg"]))
        for rule, face in (("Thr", st["color"]), ("Mar", "white")):
            rr = E2[rule]
            if rr is None or not rr["Tpr"][1] or not rr["Fpr" + n][1]:
                continue
            ax.plot(rr["Fpr" + n][0] / rr["Fpr" + n][1], rr["Tpr"][0] / rr["Tpr"][1], marker="o" if rule == "Thr" else "s",
                    ms=4.5, mfc=face, mec=st["color"], mew=0.9, ls="none", zorder=5)
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    if handles:
        handles.append(Line2D([], [], marker="o", ls="none", mfc=INK, mec=INK, ms=4.5))
        labels.append("threshold rule")
        if E2["Mar"] is not None:
            handles.append(Line2D([], [], marker="s", ls="none", mfc="white", mec=INK, ms=4.5))
            labels.append("threshold and margin")
        fig.legend(handles, labels, loc="outside lower center", ncol=2, columnspacing=1.0,
                   title="Negatives: at the marker is", title_fontsize=6.5)
    if small:
        ax.text(0.5, 0.5 if not handles else 0.62, "\n".join(small), transform=ax.transAxes, ha="center", va="center",
                color=ALERT, fontsize=6.5, gid="fb-note")
    return fig


def fig_e2_conditions(plt, E2):
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    conds = list(L.CONDITIONS)
    ax.set_xlabel("True-positive rate (%)")
    ax.set_xlim(0, 100)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.set_ylim(len(conds) - 0.4, -0.6)
    ax.set_ylabel("Capture condition")
    ax.tick_params(axis="y", length=0)
    if E2["n_labelled"] == 0 or not any(E2["conditions"][c]["thr"][1] for c in conds):
        note(ax, "TBD: no labelled target captures yet")
        return fig
    two = E2["Mar"] is not None
    h = 0.38 if two else 0.6
    for i, c in enumerate(conds):
        for j, rule in enumerate(("thr", "mar") if two else ("thr",)):
            k, n = E2["conditions"][c]["sub" if (two and rule == "thr") else rule]
            y = i + (j - 0.5) * h if two else i
            st = RULE_STYLE[rule]
            if n == 0:
                if j == 0:
                    ax.text(1, i, "no captures", va="center", ha="left", fontsize=6, color=MUTED)
                continue
            if n < M.MIN_N_BAR:
                ax.text(1, y, "n too small (%d/%d)" % (k, n), va="center", ha="left", fontsize=6, color=ALERT, gid="fb-note")
                continue
            ax.barh(y, 100.0 * k / n, height=h * 0.92, color=st["color"], hatch=st["hatch"], edgecolor="white", linewidth=0.4)
            inside = 100.0 * k / n > 30
            ax.text(100.0 * k / n + (-1.5 if inside else 1.5), y, "%d/%d" % (k, n), va="center", ha="right" if inside else "left",
                    fontsize=6, color="white" if inside and rule == "thr" else INK)
    if two:
        from matplotlib.patches import Patch
        fig.legend([Patch(facecolor=RULE_STYLE[r]["color"], hatch=RULE_STYLE[r]["hatch"], edgecolor="white") for r in ("thr", "mar")],
                   [RULE_STYLE[r]["label"] for r in ("thr", "mar")], loc="outside upper center", ncol=2)
    return fig


def fig_policy(plt, P):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 4.5), sharex=True)
    a1.set_ylabel("Wrong leads (% of queries)")
    a2.set_ylabel("Lost retrievals (% of queries)")
    a2.set_xlabel("Captures per query")
    a1.set_xlim(-0.04, 1.04)
    for ax in (a1, a2):
        ax.set_ylim(0, 50)
    if P["n"] == 0:
        for ax in (a1, a2):
            note(ax, "TBD: no paired %s queries yet" % P["exp"])
        return fig
    if P["n"] < M.MIN_POLICY_QUERIES:
        for ax in (a1, a2):
            note(ax, "n too small (n = %d queries)" % P["n"])
        return fig
    ends = P["random"]["ends"]
    st = POLICY_STYLE["Random"]
    a1.plot([0, 1], [100 * ends[0]["wrong_lead_rate"], 100 * ends[1]["wrong_lead_rate"]], color=st["color"], ls=st["ls"], label=st["label"])
    a2.plot([0, 1], [100 * ends[0]["lost_retrieval_rate"], 100 * ends[1]["lost_retrieval_rate"]], color=st["color"], ls=st["ls"])
    top = [100 * ends[0]["wrong_lead_rate"], 100 * ends[1]["lost_retrieval_rate"]]
    for key in ("All", "Frame", "Age"):
        cv, st = P["curves"][key], POLICY_STYLE[key]
        if cv is None:
            # A curve that cannot be drawn keeps its legend entry, which says why (the full reason is in provenance.txt).
            reason = P["curve_reason"][key] or ""
            a1.plot([], [], color=st["color"], ls=st["ls"], gid="fb-missing",
                    label="%s: %s" % (st["label"], "n too small" if "too small" in reason else "not available"))
            continue
        a1.plot(cv["x"], 100 * cv["wl"], color=st["color"], ls=st["ls"], label=st["label"], zorder=3)
        a2.plot(cv["x"], 100 * cv["lost"], color=st["color"], ls=st["ls"], zorder=3)
        if key == BAND_CURVE and cv["boot"]["ok"]:
            a1.fill_between(GRID, 100 * cv["wl_band"][0], 100 * cv["wl_band"][1], color=st["color"], alpha=0.25, lw=0, zorder=1)
            a2.fill_between(GRID, 100 * cv["lost_band"][0], 100 * cv["lost_band"][1], color=st["color"], alpha=0.25, lw=0, zorder=1)
            top[0] = max(top[0], float(np.nanmax(100 * cv["wl_band"][1])))
            top[1] = max(top[1], float(np.nanmax(100 * cv["lost_band"][1])))
    points = [(name, r) for name, r in P["policies"].items() if r is not None]
    if P["op"] is not None:
        points.append(("FrameOp", P["op"]))
    for name, r in points:
        st = POLICY_STYLE[name]
        x = r["rates"]["captures_per_query"]
        hollow = name == "FrameOp"
        for i_ax, (ax, key, i) in enumerate(((a1, "wrong_lead_rate", 1), (a2, "lost_retrieval_rate", 2))):
            y = 100 * r["rates"][key]
            b = r["boot"]
            kw = {}
            if b["ok"] and not (math.isnan(b["lo"][i]) or math.isnan(b["hi"][i])):
                kw["yerr"] = [[max(y - 100 * b["lo"][i], 0)], [max(100 * b["hi"][i] - y, 0)]]
                if name in ("Heuristic", "Oracle", "FrameOp") and not (math.isnan(b["lo"][0]) or math.isnan(b["hi"][0])):
                    kw["xerr"] = [[max(x - b["lo"][0], 0)], [max(b["hi"][0] - x, 0)]]
                top[i_ax] = max(top[i_ax], 100 * b["hi"][i])
            ax.errorbar([x], [y], marker=st["marker"], color=st["color"], mfc="white" if hollow else st["color"],
                        mec=st["color"] if hollow else "white", mew=1.0 if hollow else 0.5, ms=6 if hollow else 5.5, ls="none",
                        elinewidth=0.7, capsize=1.5, zorder=7 if hollow else 6, label=st["label"] if ax is a1 else None, clip_on=False, **kw)
            top[i_ax] = max(top[i_ax], y)
    a1.set_ylim(0, max(top[0] * 1.12, 5))
    a2.set_ylim(0, max(top[1] * 1.12, 5))
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, columnspacing=1.0)
    return fig


def fig_whatif(plt, W):
    """The what-if model of spec section 4, item 3. Both axis labels say 'model'."""
    fig, ax = plt.subplots(figsize=(COL_W, 3.3))
    ax.set_xlabel("AUC of the staleness predictor (model)")
    ax.set_ylabel("Captures per query needed\n(model)")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.02)
    (kp, n_p), (ks, n_s) = W["counts"]
    if W["state"] == "no data":
        note(ax, "TBD: no paired E3 queries yet")
        return fig
    if W["state"] == "n too small":
        note(ax, "n too small (%d captures at the marker, %d not; %d of each needed)" % (n_p, n_s, M.MIN_N_SHARE))
        return fig
    for b, ls in zip(WHATIF_BASE_RATES, (":", "-", "--")):
        ax.plot(WHATIF_GRID, W["curves"][b], color=INK, ls=ls, label="model, %.0f%% of queries stale" % (100 * b))
    # Every AUC actually measured, as a marker on the AUC axis. One below 0.5 sits at the left edge.
    style = {"age only": (POLICY_STYLE["Age"]["color"], "v"), "all signals": (POLICY_STYLE["All"]["color"], "o"),
             "frame signals": (POLICY_STYLE["Frame"]["color"], "s"), "mobility only": ("#009E73", "^")}
    for label, auc in W["marks"]:
        color, marker = style[label.split(",")[0]]
        ax.plot([min(max(auc, 0.5), 1.0)], [0], marker=marker, ms=5.5, mfc=color if "E5" not in label else "white", mec=color, mew=0.9,
                ls="none", clip_on=False, zorder=10, label="measured AUC: %s (%.2f)" % (label, auc))
    fig.legend(loc="outside lower center", ncol=1, title_fontsize=6.5,
               title="Model: wrong leads within %d points of always verifying.\nMeasured input: %.0f%% of captures confirm a stale lead (%d/%d)."
               % (round(100 * M.WL_TOLERANCE), 100 * W["c_s"], ks, n_s))
    return fig


def build_figures(plt, E1, E2, PS, W, synthetic):
    figs = OrderedDict([
        ("F1_marker_error_cdf.pdf", fig_e1_cdf(plt, E1)),
        ("F1_error_vs_walked.pdf", fig_e1_walk(plt, E1)),
        ("F2_roc.pdf", fig_e2_roc(plt, E2)),
        ("F2_conditions.pdf", fig_e2_conditions(plt, E2)),
    ])
    for exp in POLICY_EXPS:
        figs["F3_policy_%s.pdf" % exp] = fig_policy(plt, PS[exp])
    figs["F4_what_if.pdf"] = fig_whatif(plt, W)
    if synthetic:
        for f in figs.values():
            watermark(f)
    return figs


# ----------------------------------------------------------------------------- report

def build_report(ds, a, events, queries, captures, stats, D, E1, E2, PS, C, FR, NAV, W, synthetic):
    r = []
    add = r.append
    if synthetic:
        add("*** SYNTHETIC INPUT. Everything below is test output of the pipeline, not a result. ***")
    add("== Inputs ==")
    for f in ds.folders:
        add("  %s: %d events%s, %d records, history %s, %d map file(s)%s" % (
            shown(f.path), len(f.events), "" if f.has_events else " (no events.jsonl)", len(f.records),
            "records_log.jsonl" if f.records_log is not None else ("records.json only" if f.has_records else "none"),
            len(f.maps), ("; %d unreadable line(s)" % f.bad_lines) if f.bad_lines else ""))
    add("  %d distinct events (%d duplicates across folders dropped), %d sessions" % (len(ds.events), ds.duplicates, len(ds.sessions())))
    tags = ds.session_tags()
    untagged = sorted(str(s) for s, t in tags.items() if t == [""])
    partly = sorted(str(s) for s, t in tags.items() if "" in t and len(t) > 1)
    add("  sessions with a study tag: %d; without (left out of everything): %d%s" % (
        len(tags) - len(untagged), len(untagged), (" [" + ", ".join(untagged) + "]") if untagged else ""))
    if partly:
        add("  sessions tagged part of the time (only their tagged events count): " + ", ".join(partly))
    if a.exclude_session:
        add("  excluded by --exclude-session: " + ", ".join(a.exclude_session))
    add("  events kept: %d;  queries: %d;  captures: %d" % (len(events), len(queries), len(captures)))
    if any(stats.values()):
        add("  join problems: " + ", ".join("%s=%d" % kv for kv in stats.items() if kv[1]))
    clash = sum(1 for q in queries if q["marker_displaced_flag"] is not None and q["truth"] in ("present", "displaced")
                and q["marker_displaced_flag"] != (q["truth"] == "displaced"))
    if clash:
        add("  WARNING: groundTruth and markerError.displaced disagree in %d queries; groundTruth is used" % clash)
    bad_tags = sorted({e["_tag_raw"] for e in events if e["_tag"].get("_malformed") or not L.tag_exp(e["_tag"])})
    if bad_tags:
        add("  WARNING: tags without a usable exp= (not used by any experiment): " + "; ".join(repr(t) for t in bad_tags))
    if D["no_home"] and D["have_data"]:
        add("  WARNING: some tagged events have no home=; they are listed under site '?'")

    add("== T1 dataset ==")
    for (exp, home), g in D["groups"].items():
        add("  exp %-5s site %-3s sessions %2d rooms %2d items %3d reg %3d  truth %s  unfinished %d  captures %d  labelled %d" % (
            exp, home, len(g["sessions"]), len(g["rooms"]), len(g["items"]), g["registrations"],
            dict(g["truth"]), g["unfinished"], g["captures"], g["labelled"]))
    if not D["groups"]:
        add("  no tagged events: every number is TBD")

    add("== F1 (E1) ==")
    add("  usable E1 queries %d; marker-error measurements %d" % (E1["n_queries"], len(E1["rows"])))
    for g in list(E1_GROUPS) + ["MapUnknown", "Other"]:
        if E1["groups"][g]:
            add("  %-11s n=%d" % (g, len(E1["groups"][g])))
    if len(E1["walk"]) >= M.MIN_N_CURVE and E1["rho"] is None:
        add("  correlation of error with metres walked: undefined (one of the two does not vary); the macro stays TBD")
    add("  re-fix events (refresh/manual): %d; IMAGE jumps %d, SPOT jumps %d; disagreements asked %d" % (
        E1["n_fixes"], len(E1["jumps"]["IMAGE"]["m"]), len(E1["jumps"]["SPOT"]["m"]), E1["n_asked"]))
    add("  queries that start in another room (E1 and E3): %d of %d usable (%d in the same room, %d with a room unknown); "
        "nearReached logged on %d, reached NEAR %d, with a time %d" % (
            NAV["n_cross"], NAV["n_usable"], NAV["n_same_room"], NAV["n_room_unknown"], NAV["n_judged"], NAV["n_reached"], len(NAV["seconds"])))
    if NAV["n_left_out"]:
        add("  NOTE: %d more cross-room queries were skipped or never finished and are in none of these numbers; %d of them did not "
            "reach NEAR (or did not log it). If those were failed walks, the share above is too high" % (
                NAV["n_left_out"], NAV["n_left_out_not_reached"]))

    add("== Frame staleness (present queries of E1 and E3) ==")
    add("  present queries %d %s; marker error over %.0f cm: %d; without a measurement (counted as not stale): %d" % (
        FR["n_rows"], dict(FR["by_exp"]), M.R_PRESENT_M * 100, FR["n_stale_all"], FR["n_unmeasured"]))
    if FR["n_rows"]:
        if FR["protocol"] == "dev/test":
            add("  protocol dev/test: trained on %d dev queries, scored on %d test queries (%d stale, %d sessions); %d without a split left out"
                % (FR["n_train"], FR["n"], FR["n_stale"], FR["n_sessions"], FR["n_split_ignored"]))
        elif FR["protocol"]:
            add("  protocol %s: %d queries scored (%d stale, %d sessions), each session by a model trained on the others (smallest training fold %d)"
                % (FR["protocol"], FR["n"], FR["n_stale"], FR["n_sessions"], FR["n_train"]))
        if FR["auc"] is None:
            add("  AUC of the frame signals: not available (%s)" % FR["reason"])
        else:
            add("  AUC of the frame signals: computed; interval %s" % ("yes" if FR["boot"]["ok"] else "no (%d sessions, %d needed)" % (FR["n_sessions"], M.MIN_BOOT_CLUSTERS)))
        if FR.get("warning"):
            add("  WARNING: " + FR["warning"])

    add("== F2 (E2) ==")
    add("  captures %d, labelled %d, without a usable label %d" % (E2["n_events"], E2["n_labelled"], E2["n_unlabelled"]))
    if E2["n_labelled"]:
        tv, tt = E2["tau"].describe(E2["ev"])
        add("  protocol: " + E2["protocol"])
        add("  threshold: %s (%s); margin: %s" % ("%.2f" % tv if tv is not None else "per capture", tt,
                                                   "%.2f" % E2["margin"] if E2["margin"] is not None else "not available"))
        add("  evaluated captures %d (dev %d); by label %s" % (len(E2["ev"]), E2["n_dev"], dict(E2["n_by_label"])))
        add("  bootstrap clusters: %d (%s); intervals need %d" % (E2["n_clusters"], E2["cluster_unit"], M.MIN_BOOT_CLUSTERS))
        add("  margin inputs: record history %s; unusable %s; logged similarity differs from the embedding file in %d capture(s)" % (
            dict(E2["margin_sources"]) or "none", dict(E2["margin_missing"]) or "none", E2["sim_mismatch"]))
        if E2["n_condition_other"]:
            add("  target captures with no or an unknown condition: %d" % E2["n_condition_other"])
        if E2["n_no_threshold"]:
            add("  WARNING: %d evaluated captures have no threshold (none logged, none chosen); they are in the ROC but not in the rates"
                % E2["n_no_threshold"])
    for w in E2["warnings"]:
        add("  WARNING: " + w)

    for exp in POLICY_EXPS:
        P = PS[exp]
        add("== F3 (%s) ==" % exp)
        add("  queries %d: usable %d %s, skip %d, unfinished %d" % (P["n_all"], P["n_usable"], dict(P["by_truth"]), P["n_skip"], P["n_unfinished"]))
        if not P["n_usable"]:
            continue
        add("  capture mode of usable queries: %s" % dict(P["by_mode"]))
        add("  left out: no capture %d %s; no threshold %d; wrong capture mode %d" % (
            P["n_no_capture"], dict(P["no_capture"]), P["n_no_threshold"], P["n_mode_excluded"]))
        add("  paired queries %d; learned-policy protocol: %s" % (P["n_paired_all"], P["protocol"]))
        if P["protocol"] == "dev/test":
            add("    trained on %d dev queries, every policy replayed on the %d test queries (%d sessions); %d without a split left out"
                % (P["n_train"], P["n"], P["n_sessions"], P["n_split_ignored"]))
        elif P["protocol"]:
            add("    every policy replayed on all %d queries (%d sessions); each session scored by a model trained on the others (smallest training fold %d)"
                % (P["n"], P["n_sessions"], P["n_train"]))
        add("  threshold: %s (%s)" % ("%.2f" % P["tau_value"] if P["tau_value"] is not None else "per capture", P["tau_text"]))
        add("  0.30 m rule: %d present with a marker error above it (counted as not at the marker), %d present without a measurement "
            "(counted as at the marker), %d displaced within it (still not at the marker)" % (
                P["n_present_far"], P["n_present_unmeasured"], P["n_displaced_near"]))
        for key, spec in CURVES.items():
            if P["curves"][key] is None:
                add("  curve '%s': not available (%s)" % (spec["label"], P["curve_reason"][key] or "no replayed queries"))
            elif P["curves"][key]["auc_small"]:
                add("  curve '%s': drawn; its AUC needs %d queries of each kind and stays 'n too small'" % (spec["label"], M.MIN_ROC_PER_CLASS))
        if P.get("constant_signals"):
            add("  signals with no variation in this data (they cannot contribute): " + ", ".join(P["constant_signals"]))
        if P.get("n_age_missing"):
            add("  queries without a memory age (captured last by the age-only policy): %d" % P["n_age_missing"])
        if P["op"] is None:
            add("  frame-signals operating point: not available (%s)" % (P["op_reason"] or "no replayed queries"))
        else:
            add("  frame-signals operating point: chosen on %d dev queries (in-sample scores, used only to fix the threshold), "
                "then applied unchanged to the %d test queries" % (P["op"]["dev_n"], P["n"]))
            if not math.isfinite(P["op"]["threshold"]):
                add("    on the dev split no capture at all was already within %d points of always verifying, so the chosen policy never captures"
                    % round(100 * M.WL_TOLERANCE))
        cs = P["confirm"]
        add("  verification behaviour, all %d paired queries (%d sessions): captures at the marker %d, not at the marker %d %s; a share needs %d" % (
            cs["n"], cs["n_sessions"], cs["counts"]["AtMarker"][1], cs["counts"]["Stale"][1],
            {k: cs["counts"][k][1] for k in ("Absent", "Displaced", "Replaced", "PresentFar")}, M.MIN_N_SHARE))
        add("  bootstrap: %d sessions; intervals need %d" % (P["n_sessions"], M.MIN_BOOT_CLUSTERS))
        for w in P["warnings"]:
            add("  WARNING: " + w)

    add("== F4 what-if model ==")
    (kp, n_p), (ks, n_s) = W["counts"]
    if W["state"] is None:
        add("  measured inputs from E3: confirmations at the marker %d of %d, not at the marker %d of %d; curves for stale shares %s"
            % (kp, n_p, ks, n_s, ", ".join("%.1f" % b for b in WHATIF_BASE_RATES)))
        flat = [b for b in WHATIF_BASE_RATES if b * (1 - W["c_s"]) <= M.WL_TOLERANCE]
        if flat:
            add("  NOTE: for stale share %s memory alone is already within %d points of always verifying, so that curve is zero at every AUC"
                % (", ".join("%.1f" % b for b in flat), round(100 * M.WL_TOLERANCE)))
    else:
        add("  not drawn (%s): captures at the marker %d, not at the marker %d; %d of each needed" % (W["state"], n_p, n_s, M.MIN_N_SHARE))
    add("  measured AUCs marked on its axis: %s" % ("; ".join(label for label, _ in W["marks"]) or "none"))

    add("== T2 cost ==")
    add("  registrations with latency %d; verifications with latency %d; first captures with msSinceNear and totalMs %d; map files %d" % (
        len(C["reg"]["Total"]), len(C["ver"]["Total"]), len(C["capture_s"]), len(C["map_bytes"])))
    add("  power runs used: tracking_only %d, tracking_verify %d; verifications behind the energy figure: %d" % (
        C["power"]["tracking_only"]["n"], C["power"]["tracking_verify"]["n"], C["energy_n"]))
    for w in C["warnings"]:
        add("  WARNING: " + w)
    if synthetic:
        add("*** SYNTHETIC INPUT. Not a result. ***")
    return r


# ----------------------------------------------------------------------------- main

def is_paper_path(path):
    """True for the project's paper/ directory and for any path with a component named 'paper'."""
    real = os.path.realpath(path)
    project_paper = os.path.realpath(os.path.join(PROJECT, "paper"))
    if real == project_paper or real.startswith(project_paper + os.sep):
        return True
    return any(part.lower() == "paper" for part in real.split(os.sep))


def shown(path):
    """A path as written into generated files: relative to the project when inside it."""
    real = os.path.realpath(path)
    root = os.path.realpath(PROJECT)
    return os.path.relpath(real, root) if real.startswith(root + os.sep) else real


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def load_inputs(data, power_dirs=()):
    """(dataset, power runs, synthetic?) for the given data folders and power-run folders."""
    ds = L.Dataset(list(data))
    power_runs = load_power_runs(power_dirs)
    synthetic = ds.any_synthetic or any(power_run_is_synthetic(r) for r in power_runs)
    return ds, power_runs, synthetic


class Analysis:
    pass


def analyse(ds, power_runs, synthetic, exclude_sessions=(), threshold=None, e2_cluster="session"):
    """Everything the outputs are built from. Used by main() and by the tests."""
    A = Analysis()
    A.events = ds.tagged(exclude_sessions)
    A.queries, A.captures, A.stats = L.build_queries(A.events)
    regs = L.registrations_from(A.events)
    fixes = L.fixes_from(A.events)
    A.D = compute_dataset(A.events, A.queries, A.captures, regs)
    A.E1 = compute_e1(A.queries, fixes, ds)
    A.E2 = compute_e2(A.captures, ds, threshold, e2_cluster)
    # The replay judges a capture with the same threshold as F2: the one chosen on the E2 dev split
    # (or given on the command line), else the on-device threshold logged with each capture.
    A.PS = {exp: compute_policy(exp, A.queries, A.E2["tau"]) for exp in POLICY_EXPS}
    A.C = compute_cost(A.events, A.queries, A.captures, regs, ds, power_runs)
    A.FR = compute_frame_staleness(A.queries)
    A.NAV = compute_cross_room(A.queries)
    A.W = compute_whatif(A.PS, A.FR)
    A.N = Numbers(synthetic)
    fill_macros(A.N, A.D, A.E1, A.E2, A.PS, A.C, A.FR, A.NAV, A.W)
    return A


def main(argv=None):
    ap = argparse.ArgumentParser(description="Figures, tables and numbers of the FindBack paper.")
    ap.add_argument("data", nargs="+", help="data folders pulled from phones")
    ap.add_argument("--out", default=os.path.join(PROJECT, "paper"), help="output root (default: %(default)s)")
    ap.add_argument("--power-dir", action="append", default=[], help="folder of power_measure.py runs (repeatable)")
    ap.add_argument("--exclude-session", action="append", default=[])
    ap.add_argument("--threshold", type=float, help="use this similarity threshold instead of the dev-chosen or logged one")
    ap.add_argument("--e2-cluster", choices=("session", "item"), default="session",
                    help="unit resampled for the E2 bootstrap intervals (default: session, as the spec says)")
    ap.add_argument("--allow-synthetic", action="store_true", help="allow marked synthetic output under paper/")
    ap.add_argument("--debug-json", help="also dump the raw values of the macros here (never under paper/)")
    a = ap.parse_args(argv)

    for p in a.data:
        if not os.path.isdir(p):
            ap.error("not a folder: %s" % p)
    ds, power_runs, synthetic = load_inputs(a.data, a.power_dir)
    if not any(f.has_events for f in ds.folders):
        ap.error("no events.jsonl in the given folders")

    if synthetic and is_paper_path(a.out) and not a.allow_synthetic:
        print("REFUSED: the input contains synthetic data and the output directory is under paper/:", file=sys.stderr)
        print("  " + os.path.realpath(a.out), file=sys.stderr)
        for s in sorted(map(str, ds.synthetic_sessions))[:5]:
            print("  synthetic session %s (%s)" % (s, "; ".join(sorted(ds.synthetic_sessions[s]))), file=sys.stderr)
        if len(ds.synthetic_sessions) > 5:
            print("  ... and %d more" % (len(ds.synthetic_sessions) - 5), file=sys.stderr)
        for r in power_runs:
            if power_run_is_synthetic(r):
                print("  synthetic power run %s" % r["_path"], file=sys.stderr)
        print("Nothing was written. Use another --out, or --allow-synthetic to write output in which every number "
              "is replaced by SYNTHETIC and every figure is watermarked.", file=sys.stderr)
        return 2
    if a.debug_json and is_paper_path(a.debug_json):
        print("REFUSED: --debug-json must not point under paper/.", file=sys.stderr)
        return 2

    A = analyse(ds, power_runs, synthetic, a.exclude_session, a.threshold, a.e2_cluster)
    events, queries, captures, stats = A.events, A.queries, A.captures, A.stats
    D, E1, E2, PS, C, N = A.D, A.E1, A.E2, A.PS, A.C, A.N
    report = build_report(ds, a, events, queries, captures, stats, D, E1, E2, PS, C, A.FR, A.NAV, A.W, synthetic)
    banner = ["SYNTHETIC DATA: every number is replaced by SYNTHETIC. Not a result."] if synthetic else []
    header = banner + ["FindBack paper numbers. Generated by analysis/paper_figures.py from:"] + [
        "  " + shown(p) for p in a.data] + ["  power runs: " + (", ".join(shown(p) for p in a.power_dir) or "none")]
    tables = build_tables(N, D, E1, E2, PS, C, banner)

    plt = setup_matplotlib()
    figs = build_figures(plt, E1, E2, PS, A.W, synthetic)

    fig_dir, gen_dir = os.path.join(a.out, "figures"), os.path.join(a.out, "generated")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(os.path.join(fig_dir, name), format="pdf", bbox_inches="tight", pad_inches=0.02,
                    metadata={"CreationDate": None, "Title": name, "Creator": "analysis/paper_figures.py",
                              "Keywords": "SYNTHETIC DATA, not a result" if synthetic else "FindBack"})
        plt.close(fig)
    with open(os.path.join(gen_dir, "results.tex"), "w") as f:
        f.write(N.results_tex(header))
    for name, text in tables.items():
        with open(os.path.join(gen_dir, name), "w") as f:
            f.write(text)
    with open(os.path.join(gen_dir, "provenance.txt"), "w") as f:
        if synthetic:  # no numbers of a synthetic run are kept next to the paper
            f.write("\n".join(banner + header[1:]) + "\n")
        else:
            f.write("\n".join(header + [""] + report) + "\n")
    if a.debug_json:
        with open(a.debug_json, "w") as f:
            json.dump({"synthetic": synthetic, "WARNING": "SYNTHETIC test output, not a result" if synthetic else None,
                       "values": to_jsonable(N.raw)}, f, indent=1)

    print("\n".join(report))
    print("\nWrote %d figures to %s and results.tex, %d tables and provenance.txt to %s" % (len(figs), fig_dir, len(tables), gen_dir))
    if synthetic:
        print("Every macro and table cell is SYNTHETIC and every figure is watermarked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
