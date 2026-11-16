#!/usr/bin/env python3
"""Metric code for the FindBack paper (numpy, scipy, scikit-learn).

Everything here is a pure function of arrays or plain rows, so each definition of
STUDY_SPEC_v0.4 section 2 can be tested on a hand-built log (analysis/test_analysis.py).

Definitions (spec section 2):
  at the marker   ground truth `present` with a measured marker error <= R_PRESENT_M, or
                  `present` with no measurement. `displaced`, `absent`, `replaced` are never
                  at the marker. `skip` (and a missing ground truth) is not usable.
  stale           a usable query whose item is not at the marker.
  system answer   LAST_SEEN_HERE (no capture), HERE_NOW (capture confirmed),
                  NOT_CONFIRMED (capture did not confirm).
  correct lead    LAST_SEEN_HERE or HERE_NOW, item at the marker
  wrong lead      LAST_SEEN_HERE or HERE_NOW, item not at the marker
  correct warning NOT_CONFIRMED, item not at the marker
  lost retrieval  NOT_CONFIRMED, item at the marker
All four rates are per replayed query, so they sum to 1.
"""
import math
import warnings
from collections import OrderedDict

import numpy as np
from scipy.stats import norm, rankdata

R_PRESENT_M = 0.30
USABLE_TRUTHS = ("present", "displaced", "absent", "replaced")

LAST_SEEN_HERE, HERE_NOW, NOT_CONFIRMED = "LAST_SEEN_HERE", "HERE_NOW", "NOT_CONFIRMED"
OUTCOMES = ("correct_lead", "wrong_lead", "correct_warning", "lost_retrieval")

# Minimum sample sizes. Below them the pipeline writes "n too small" instead of a number or a curve.
MIN_N_MEDIAN = 5          # median of a latency or an error
MIN_N_P90 = 10            # 90th percentile
MIN_N_P95 = 20            # 95th percentile
MIN_N_CURVE = 10          # a CDF curve or a scatter trend
MIN_N_BAR = 5             # a rate drawn as a bar (the count is always printed)
MIN_ROC_PER_CLASS = 10    # positives and negatives for a ROC curve or an AUC
MIN_POLICY_QUERIES = 20   # replayed queries for the policy figure
MIN_TRAIN_QUERIES = 20    # training queries for the learned policy, in every fold
MIN_TRAIN_PER_CLASS = 5   # of each class (stale / not stale), in every fold
MIN_LOSO_SESSIONS = 3     # sessions for leave-one-session-out
MIN_BOOT_CLUSTERS = 4     # sessions for a clustered bootstrap interval
MIN_N_SHARE = 20          # cases behind each of the two measured shares that feed the what-if model

# "Within 5 percentage points of always verifying": the tolerance on the wrong-lead rate used
# for the operating point chosen on the dev split and for the what-if curve (spec section 4).
WL_TOLERANCE = 0.05

N_BOOT = 2000
BOOT_SEED = 20261116
BOOT_ALPHA = 0.05
BOOT_MIN_VALID = 0.9      # an interval needs the statistic defined in this share of resamples


# ----------------------------------------------------------------------------- section 2

def at_marker(truth, marker_error_m=None):
    """True / False per the definition above; None when the ground truth is not usable."""
    if truth not in USABLE_TRUTHS:
        return None
    if truth != "present":
        return False
    if marker_error_m is None:
        return True
    return marker_error_m <= R_PRESENT_M


def system_answer(captured, confirmed):
    if not captured:
        return LAST_SEEN_HERE
    return HERE_NOW if confirmed else NOT_CONFIRMED


def outcome(answer, is_at_marker):
    lead = answer in (LAST_SEEN_HERE, HERE_NOW)
    if lead:
        return "correct_lead" if is_at_marker else "wrong_lead"
    return "lost_retrieval" if is_at_marker else "correct_warning"


def confirmed_by_threshold(sim, tau):
    """Rule 1: a capture confirms when its best similarity is at least tau."""
    return sim is not None and sim >= tau


def confirmed_by_margin(sim, tau, margin_value, m):
    """Rule 2: rule 1 and (best similarity - best similarity to another record) >= m.
    `margin_value` is None when there is no other record to confuse the item with."""
    if not confirmed_by_threshold(sim, tau):
        return False
    return margin_value is None or margin_value >= m


# ----------------------------------------------------------------------------- policy replay

def replay(stale, confirmed, capture, calls=None):
    """Counts of the four outcomes when the queries in boolean mask `capture` use their capture.

    stale[i]      the item of query i is not at the marker
    confirmed[i]  the capture of query i confirms (used only where capture[i])
    calls[i]      inference calls of that capture (optional)
    """
    stale = np.asarray(stale, bool)
    confirmed = np.asarray(confirmed, bool)
    capture = np.asarray(capture, bool)
    lead = ~capture | confirmed
    out = {
        "n": int(stale.size),
        "captures": int(capture.sum()),
        "correct_lead": int((lead & ~stale).sum()),
        "wrong_lead": int((lead & stale).sum()),
        "correct_warning": int((~lead & stale).sum()),
        "lost_retrieval": int((~lead & ~stale).sum()),
    }
    if calls is not None:
        out["calls"] = float(np.asarray(calls, float)[capture].sum())
    return out


def rates(counts):
    """Per-query rates from replay() counts (nan when there is no query)."""
    n = counts["n"]
    d = float(n) if n else float("nan")
    out = {k + "_rate": counts[k] / d for k in OUTCOMES}
    out["captures_per_query"] = counts["captures"] / d
    if "calls" in counts:
        out["calls_per_query"] = counts["calls"] / d
    return out


def mask_memory_only(n):
    return np.zeros(n, bool)


def mask_always_verify(n):
    return np.ones(n, bool)


def mask_oracle(stale):
    """The oracle captures exactly the stale queries."""
    return np.asarray(stale, bool).copy()


def mask_risk(risk, threshold):
    """The learned policy captures when the risk score is at least the threshold."""
    return np.asarray(risk, float) >= threshold


def random_policy_expected(stale, confirmed, p, calls=None):
    """Expected per-query rates when each query is captured independently with probability p.

    This is exact, not simulated: every rate is linear in p, so the random baseline is the
    straight line from memory only (p = 0) to always verify (p = 1).
    """
    stale = np.asarray(stale, bool)
    confirmed = np.asarray(confirmed, bool)
    n = stale.size
    if n == 0:
        return {k: float("nan") for k in ("captures_per_query", "wrong_lead_rate", "lost_retrieval_rate",
                                           "correct_lead_rate", "correct_warning_rate", "calls_per_query")}
    wl0, wl1 = stale.mean(), (stale & confirmed).mean()
    lr1 = (~stale & ~confirmed).mean()
    out = {
        "captures_per_query": float(p),
        "wrong_lead_rate": float((1 - p) * wl0 + p * wl1),
        "lost_retrieval_rate": float(p * lr1),
        "correct_lead_rate": float((1 - p) * (~stale).mean() + p * (~stale & confirmed).mean()),
        "correct_warning_rate": float(p * (stale & ~confirmed).mean()),
    }
    if calls is not None:
        out["calls_per_query"] = float(p * np.asarray(calls, float).mean())
    return out


def sweep_risk(stale, confirmed, risk):
    """Operating points of the learned policy over every threshold.

    Queries are captured in order of decreasing risk; queries with equal risk are captured
    together (a threshold cannot split a tie, and no random tie-break is used). Returns arrays
    (captures_per_query, wrong_lead_rate, lost_retrieval_rate, threshold), starting at zero
    captures (threshold +inf: memory only) and ending at one capture per query (always verify).
    """
    stale = np.asarray(stale, bool)
    confirmed = np.asarray(confirmed, bool)
    risk = np.asarray(risk, float)
    n = stale.size
    if n == 0:
        z = np.array([], float)
        return z, z, z, z
    order = np.argsort(-risk, kind="stable")
    r = risk[order]
    s = stale[order]
    c = confirmed[order]
    # Capturing a query changes its wrong-lead count from `stale` to `stale & confirmed`.
    wl_drop = np.cumsum(s & ~c)
    lost = np.cumsum(~s & ~c)
    ends = np.flatnonzero(np.r_[r[1:] != r[:-1], True])  # last index of every tie group
    k = ends + 1
    caps = np.r_[0, k] / n
    wrong = np.r_[s.sum(), s.sum() - wl_drop[ends]] / n
    lost_rate = np.r_[0, lost[ends]] / n
    thr = np.r_[np.inf, r[ends]]
    return caps, wrong, lost_rate, thr


def curve_at(x, y, x0):
    """Linear interpolation of an operating curve at capture rate x0. A point between two
    thresholds is reached by choosing between them at random, and every rate is linear in
    that choice, so the interpolated value is the expected rate of a real policy."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0:
        return float("nan")
    return float(np.interp(x0, x, y))


def curve_area(x, y):
    """Area under an operating curve over capture rates 0..1 (trapezoids). For the wrong-lead
    curve a smaller area is better; the random baseline's area is the mean of its end points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


# ----------------------------------------------------------------------------- ROC

def auc(pos, neg):
    """Area under the ROC curve = P(score of a positive > score of a negative), ties count 1/2."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = rankdata(np.r_[pos, neg])
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def roc_points(pos, neg):
    """(fpr, tpr, threshold) for the rule 'confirm when score >= threshold', from (0, 0) to (1, 1)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    thr = np.unique(np.r_[pos, neg])[::-1]
    tpr = np.array([(pos >= t).mean() for t in thr]) if pos.size else np.full(thr.size, np.nan)
    fpr = np.array([(neg >= t).mean() for t in thr]) if neg.size else np.full(thr.size, np.nan)
    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[np.inf, thr]


THRESHOLD_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
MARGIN_GRID = np.round(np.arange(0.0, 0.5001, 0.01), 2)


def youden_threshold(pos, neg, grid=THRESHOLD_GRID):
    """Threshold of `grid` maximising TPR - FPR (Youden's J). Ties go to the higher, more
    conservative threshold, as analysis/evaluate.py does. None when a class is empty."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return None
    best_j, best_t = None, None
    for t in grid:  # ascending, so ">=" keeps the highest threshold among equals
        j = (pos >= t).mean() - (neg >= t).mean()
        if best_j is None or j >= best_j - 1e-12:
            best_j = j if best_j is None else max(j, best_j)
            best_t = float(t)
    return best_t


def youden_margin(pos_confirmed, pos_margin, neg_confirmed, neg_margin, grid=MARGIN_GRID):
    """Margin of `grid` (m >= 0) maximising Youden's J of rule 2, given which captures rule 1
    confirms. Ties go to the smaller margin. Note that m = 0 is already a constraint: the
    target must be at least as similar to the crop as every other record. A margin value of
    nan means 'no other record' and always passes."""
    pos_ok, pos_margin = np.asarray(pos_confirmed, bool), np.asarray(pos_margin, float)
    neg_ok, neg_margin = np.asarray(neg_confirmed, bool), np.asarray(neg_margin, float)
    if pos_ok.size == 0 or neg_ok.size == 0:
        return None
    best_j, best_m = None, None
    for m in grid:
        tp = (pos_ok & (np.isnan(pos_margin) | (pos_margin >= m))).mean()
        fp = (neg_ok & (np.isnan(neg_margin) | (neg_margin >= m))).mean()
        j = tp - fp
        if best_j is None or j > best_j + 1e-12:
            best_j, best_m = j, float(m)
    return best_m


# ----------------------------------------------------------------------------- bootstrap

def cluster_bootstrap(stat, clusters, n_boot=N_BOOT, seed=BOOT_SEED, alpha=BOOT_ALPHA,
                      min_clusters=MIN_BOOT_CLUSTERS):
    """Percentile bootstrap interval with whole clusters (sessions) resampled.

    stat(idx) -> a float or a 1-D array, computed on the rows `idx` (an integer array that
    repeats the rows of a cluster drawn more than once); nan where it is undefined.
    clusters: one cluster id per row.

    Every resample draws as many clusters as the data has, with replacement, and keeps all
    rows of each drawn cluster. Returns a dict:
      lo, hi      interval bounds (float or array); nan where unavailable
      n_clusters  number of clusters
      n_boot      resamples drawn
      valid       share of resamples in which the statistic was defined (float or array)
      ok          False when there are fewer than `min_clusters` clusters (bounds are nan)
    An entry whose statistic is undefined in more than 1 - BOOT_MIN_VALID of the resamples
    gets nan bounds.
    """
    clusters = np.asarray(clusters)
    if clusters.size == 0:
        return {"lo": float("nan"), "hi": float("nan"), "n_clusters": 0, "n_boot": 0, "valid": 0.0, "ok": False}
    uniq, inv = np.unique(clusters, return_inverse=True)
    k = uniq.size
    full = np.asarray(stat(np.arange(clusters.size)), float)
    scalar = full.ndim == 0
    shape = np.atleast_1d(full).shape
    if k < min_clusters:
        out = {"lo": np.full(shape, np.nan), "hi": np.full(shape, np.nan), "n_clusters": int(k), "n_boot": 0,
               "valid": np.zeros(shape), "ok": False}
    else:
        members = [np.flatnonzero(inv == c) for c in range(k)]
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot,) + shape)
        for b in range(n_boot):
            pick = rng.integers(0, k, size=k)
            idx = np.concatenate([members[c] for c in pick])
            draws[b] = np.atleast_1d(np.asarray(stat(idx), float))
        valid = np.mean(~np.isnan(draws), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-nan columns
            lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
            hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
        bad = valid < BOOT_MIN_VALID
        out = {"lo": np.where(bad, np.nan, lo), "hi": np.where(bad, np.nan, hi), "n_clusters": int(k),
               "n_boot": int(n_boot), "valid": valid, "ok": True}
    if scalar:
        for key in ("lo", "hi", "valid"):
            out[key] = float(out[key][0])
    return out


# ----------------------------------------------------------------------------- learned policy

# The pre-capture signals of spec section 2 and the feature columns built from each. Nothing
# else may enter the risk score.
SIGNALS = OrderedDict([
    ("memory age", ["log_age_min"]),
    ("mobility class", ["mobility"]),
    ("earlier moves", ["move_count"]),
    ("position source", ["source_is_map"]),
    ("metres since fix", ["log_m_since_fix"]),
    ("tracking losses", ["losses_since_fix"]),
    ("time since fix", ["log_min_since_fix"]),
    ("fix type", ["fix_is_image", "fix_is_spot"]),
])
FEATURES = [c for cols in SIGNALS.values() for c in cols]
# World staleness: the item may have moved. Frame staleness: the marker may be drawn in the
# wrong place although the item has not moved.
WORLD_SIGNALS = ["memory age", "mobility class", "earlier moves"]
FRAME_SIGNALS = ["position source", "metres since fix", "tracking losses", "time since fix", "fix type"]
assert sorted(WORLD_SIGNALS + FRAME_SIGNALS) == sorted(SIGNALS)
MOBILITY_ORDINAL = {"rare": 0.0, "some": 1.0, "often": 2.0}
LOGREG_C = 1.0


def features_of(q):
    """Feature vector of one query row (fb_load.build_queries), in FEATURES order.

    Missing values: mobility -> 'some' (the app's default); move count -> 0 (spec S7); the
    drift proxies (metres, losses, time since the fix) -> 0 when the phone was not located,
    in which case both fix-type indicators are 0 as well, so 'not located' is its own level.
    Ages, metres and minutes enter as log(1 + x).
    """
    def nz(x):
        return x if x is not None else 0.0
    fix = (q.get("fix_type") or "").upper() if isinstance(q.get("fix_type"), str) else ""
    src = (q.get("position_source") or "").upper() if isinstance(q.get("position_source"), str) else ""
    row = {
        "log_age_min": math.log1p(max(nz(q.get("age_min")), 0.0)),
        "mobility": MOBILITY_ORDINAL.get(q.get("mobility"), 1.0),
        "move_count": nz(q.get("move_count")),
        "source_is_map": 1.0 if src in ("MAP", "REFERENCE") else 0.0,
        "log_m_since_fix": math.log1p(max(nz(q.get("meters_since_fix")), 0.0)),
        "losses_since_fix": nz(q.get("losses_since_fix")),
        "log_min_since_fix": math.log1p(max(nz(q.get("ms_since_fix")), 0.0) / 60000.0),
        "fix_is_image": 1.0 if fix == "IMAGE" else 0.0,
        "fix_is_spot": 1.0 if fix == "SPOT" else 0.0,
    }
    return [row[c] for c in FEATURES]


def feature_matrix(queries):
    return np.array([features_of(q) for q in queries], float).reshape(len(queries), len(FEATURES))


def columns_for(signals):
    """Column indices of FEATURES for a list of signal names."""
    return [FEATURES.index(c) for s in signals for c in SIGNALS[s]]


def fit_predict(x_train, y_train, x_eval, c=LOGREG_C):
    """P(stale) for x_eval from an L2-regularised logistic regression fitted on the training
    rows only. Features are standardised with the training mean and standard deviation; a
    column that is constant in training carries no information and is set to 0."""
    from sklearn.linear_model import LogisticRegression
    x_train = np.asarray(x_train, float)
    x_eval = np.asarray(x_eval, float)
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    keep = sd > 1e-12
    sd = np.where(keep, sd, 1.0)
    z_train = np.where(keep, (x_train - mu) / sd, 0.0)
    z_eval = np.where(keep, (x_eval - mu) / sd, 0.0)
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000)
    model.fit(z_train, np.asarray(y_train, int))
    return model.predict_proba(z_eval)[:, list(model.classes_).index(1)]


def _train_ok(y):
    y = np.asarray(y, bool)
    return y.size >= MIN_TRAIN_QUERIES and min(int(y.sum()), int((~y).sum())) >= MIN_TRAIN_PER_CLASS


def out_of_sample_risk(x, stale, sessions, splits, columns=None, fit=fit_predict):
    """Risk scores that never come from a model trained on the scored query.

    Protocol 'dev/test': when the rows hold both split 'dev' and split 'test', the model is
    fitted on the dev rows and scores the test rows. Rows with no split are then left out.
    Protocol 'leave-one-session-out': otherwise, each session is scored by a model fitted on
    all other sessions.

    Returns a dict:
      protocol   'dev/test' | 'leave-one-session-out' | None
      eval       boolean mask of the rows that are scored (and on which every policy is replayed)
      risk       array, nan outside `eval`; None when the learned policy is unavailable
      reason     why unavailable, else None
      folds      list of (train_index_array, eval_index_array), for audit and tests
      n_train    training rows (dev/test), or the smallest fold's (leave-one-session-out)
      warning    set when a session holds both dev and test queries (the tag changed without
                 an app restart): no query is shared, but the two splits are then not
                 independent sessions
    """
    x = np.asarray(x, float)
    stale = np.asarray(stale, bool)
    sessions = np.asarray(sessions)
    splits = np.asarray([s if s in ("dev", "test") else "" for s in splits])
    n = stale.size
    if columns is not None:
        x = x[:, list(columns)]
    out = {"protocol": None, "eval": np.zeros(n, bool), "risk": None, "reason": None, "folds": [], "n_train": 0,
           "warning": None}
    has_dev, has_test = bool((splits == "dev").any()), bool((splits == "test").any())
    if has_dev and has_test:
        out["protocol"] = "dev/test"
        train = np.flatnonzero(splits == "dev")
        ev = np.flatnonzero(splits == "test")
        out["eval"][ev] = True
        out["folds"] = [(train, ev)]
        out["n_train"] = int(train.size)
        shared = set(sessions[train].tolist()) & set(sessions[ev].tolist())
        if shared:
            out["warning"] = "sessions with both dev and test queries: " + ", ".join(sorted(map(str, shared)))
        if not _train_ok(stale[train]):
            out["reason"] = "n too small: dev has %d queries, %d stale" % (train.size, int(stale[train].sum()))
            return out
        risk = np.full(n, np.nan)
        risk[ev] = fit(x[train], stale[train], x[ev])
        out["risk"] = risk
        return out

    out["protocol"] = "leave-one-session-out"
    out["eval"][:] = True
    uniq = list(dict.fromkeys(sessions.tolist()))
    if len(uniq) < MIN_LOSO_SESSIONS:
        out["reason"] = "n too small: %d session(s), leave-one-session-out needs %d" % (len(uniq), MIN_LOSO_SESSIONS)
        return out
    folds = [(np.flatnonzero(sessions != s), np.flatnonzero(sessions == s)) for s in uniq]
    out["folds"] = folds
    out["n_train"] = int(min(tr.size for tr, _ in folds))
    for tr, _ in folds:
        if not _train_ok(stale[tr]):
            out["reason"] = "n too small: a training fold has %d queries, %d stale" % (tr.size, int(stale[tr].sum()))
            return out
    risk = np.full(n, np.nan)
    for tr, ev in folds:
        risk[ev] = fit(x[tr], stale[tr], x[ev])
    out["risk"] = risk
    return out


def dev_risk_for_operating_point(x, stale, splits, columns=None, fit=fit_predict):
    """Scores of the dev rows from the model fitted on those same dev rows.

    These are in-sample, so they are never reported as performance. Their only use is to choose
    a risk threshold before the test rows are looked at (choose_operating_threshold). Returns
    (dev_index_array, risk_of_dev_rows), or (dev_index_array, None) when the dev rows cannot
    train a model. The model is the one out_of_sample_risk fits under the dev/test protocol.
    """
    x = np.asarray(x, float)
    stale = np.asarray(stale, bool)
    dev = np.flatnonzero(np.asarray([s == "dev" for s in splits], bool))
    if columns is not None:
        x = x[:, list(columns)]
    if not _train_ok(stale[dev]):
        return dev, None
    return dev, fit(x[dev], stale[dev], x[dev])


def choose_operating_threshold(stale, confirmed, risk, tolerance=WL_TOLERANCE):
    """The risk threshold with the fewest captures whose wrong-lead rate, on these (dev) queries,
    is within `tolerance` of always verifying. A policy then captures when risk >= threshold.

    Returns (threshold, captures_per_query, wrong_lead_rate) on the queries given. The threshold
    is +inf when no capture at all is already within the tolerance. Always verifying satisfies
    the condition, so there is always an answer.
    """
    x, wl, _, thr = sweep_risk(stale, confirmed, risk)
    if x.size == 0:
        return float("inf"), float("nan"), float("nan")
    ok = np.flatnonzero(wl <= wl[-1] + tolerance + 1e-12)   # wl never rises as captures are added
    i = int(ok[0])
    return float(thr[i]), float(x[i]), float(wl[i])


# ----------------------------------------------------------------------------- what-if model

def binormal_fpr(tpr, auc):
    """False-positive rate at a given true-positive rate on the equal-variance binormal ROC
    curve with area `auc` (0.5 <= auc <= 1): TPR = Phi(d + Phi^-1(FPR)), d = sqrt(2) Phi^-1(auc).
    AUC 0.5 is the diagonal; AUC 1 is the corner, with FPR 0 at every TPR."""
    tpr = float(tpr)
    if tpr <= 0.0:
        return 0.0
    if auc >= 1.0:
        return 0.0
    if tpr >= 1.0:
        return 1.0
    d = math.sqrt(2.0) * float(norm.ppf(auc))
    return float(norm.cdf(norm.ppf(tpr) - d))


def whatif_rates(b, tpr, fpr, c_p, c_s):
    """The model of spec section 4 ("How the selective policy is evaluated", item 3).

    b    base rate of stale queries
    tpr  share of stale queries the predictor sends to a capture; fpr: the same for fresh ones
    c_p  share of captures that confirm when the item is at the marker (measured)
    c_s  share of captures that confirm when it is not (measured)
    """
    return {
        "wrong_lead_rate": b * (1.0 - tpr * (1.0 - c_s)),
        "lost_retrieval_rate": (1.0 - b) * fpr * (1.0 - c_p),
        "captures_per_query": b * tpr + (1.0 - b) * fpr,
    }


def whatif_captures_needed(b, auc, c_s, tolerance=WL_TOLERANCE):
    """Fewest captures per query that bring the model's wrong-lead rate within `tolerance` of
    always verifying, for a predictor with the given AUC. Returns (captures, tpr, fpr).

    Always verifying leaves b * c_s wrong leads, so the condition is
    b * (1 - c_s) * (1 - TPR) <= tolerance, i.e. TPR >= 1 - tolerance / (b * (1 - c_s)).
    Captures grow with TPR along the curve, so the answer is at that TPR. When memory alone is
    already within the tolerance, no capture is needed.
    """
    gain = b * (1.0 - c_s)                     # wrong leads that verifying every stale query removes
    if gain <= tolerance:
        return 0.0, 0.0, 0.0
    tpr = 1.0 - tolerance / gain
    fpr = binormal_fpr(tpr, auc)
    return whatif_rates(b, tpr, fpr, 0.0, c_s)["captures_per_query"], tpr, fpr


# ----------------------------------------------------------------------------- summaries

def quantile(values, q, min_n):
    """Linear-interpolated quantile, or None when there are fewer than min_n values."""
    v = [x for x in values if x is not None]
    if len(v) < min_n:
        return None
    return float(np.quantile(np.asarray(v, float), q))


def median(values, min_n=MIN_N_MEDIAN):
    return quantile(values, 0.5, min_n)
