#!/usr/bin/env python3
"""Offline evaluator for FindBack pilot data (stdlib only).

Replays the three study conditions on the same logged queries:
  memory_only    - claim the remembered spot is current; no capture.
  always_verify  - use the (first) fresh capture for every query.
  selective      - use the fresh capture only when the logged policy said VERIFY.
A capture "confirms" when its best similarity >= threshold; otherwise the outcome is
"uncertain" (insufficient evidence), never "moved".

v0.3 (home map) adds three sections at the end of the report: marker error by position source,
map-fix jumps (the drift measurement), and queries by guidance mode. Position source "REFERENCE"
(v0.2 picture offset) and "MAP" (v0.3 home map) are both accepted.

v0.4 ground truth: `displaced` and `replaced` are counted as "not present", exactly like `absent`
(a claim on them is WRONG, a withheld claim is unc|abs). The tap after `displaced` measures the
item's displacement, so it is kept out of the marker-error statistics and reported on its own
line. This summary still treats every `present` as present; the paper pipeline
(paper_figures.py) also applies the 0.30 m rule of the study spec, and a note at the very end
says how many queries that changes.

Usage:
  python3 analysis/evaluate.py findback_data
  python3 analysis/evaluate.py findback_data --threshold 0.7
  python3 analysis/evaluate.py findback_data --dev-session ab12cd34   # pick threshold on dev, report on rest
  python3 analysis/evaluate.py findback_data --exclude-session cab39b87 --csv queries.csv
"""
import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict

NOT_PRESENT = ("absent", "displaced", "replaced")  # usable ground truths other than "present"
R_PRESENT_M = 0.30  # study spec section 2; used only for the note at the end (see the docstring)


def load_events(root):
    with open(os.path.join(root, "events.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_queries(events, excluded):
    q = defaultdict(lambda: {"verifies": []})
    for e in events:
        if e.get("session") in excluded:
            continue
        t = e["type"]
        if t == "query_start":
            q[e["query"]].update(start=e, session=e["session"])
        elif t == "verify":
            q[e["query"]]["verifies"].append(e)
        elif t == "query_end":
            q[e["query"]]["end"] = e
    rows = []
    for qid, d in q.items():
        if "start" not in d or "end" not in d:
            continue
        truth = d["end"]["groundTruth"]
        if truth != "present" and truth not in NOT_PRESENT:
            continue
        measured = (d["end"].get("markerError") or {}).get("errorM")
        first = sorted(d["verifies"], key=lambda v: v["attempt"])[0] if d["verifies"] else None
        rows.append(
            {
                "query": qid,
                "session": d["session"],
                "record": d["start"]["record"],
                "name": d["start"].get("name", ""),
                "truth": truth,
                "policy_verify": bool(d["start"]["policyVerify"]),
                "position_source": d["start"]["positionSource"],
                "age_min": d["start"]["ageMin"],
                "max_sim_others": d["start"]["maxSimOthers"],
                "sim": first["bestSimilarity"] if first else None,
                "sim_projected": next((s["sim"] for s in first["scores"] if s["kind"] == "projected"), None) if first else None,
                "inference_calls": first["inferenceCalls"] if first else 0,
                "verify_ms": first["latency"]["totalMs"] if first else None,
                "marker_error_m": measured if truth == "present" else None,
                # v0.3 columns (None for older logs).
                "marker_source": (d["end"].get("markerError") or {}).get("positionSource"),
                "marker_horizontal_m": (d["end"].get("markerError") or {}).get("horizontalErrorM"),
                "guidance_mode": d["start"].get("guidanceMode"),
                "near_reached": d["end"].get("nearReached"),
                "time_to_near_ms": d["end"].get("timeToNearMs"),
                "item_room": d["start"].get("itemRoomName"),
                "user_room": d["start"].get("userRoomName"),
                "route_m": d["start"].get("routeM"),
                "straight_m": d["start"].get("straightM"),
                "fix_type": d["start"].get("fixType"),
                "meters_since_fix": d["start"].get("metersSinceFix"),
                "tracking_losses_since_fix": d["start"].get("trackingLossesSinceFix"),
                # v0.4 column: the tap after `displaced` is the item's displacement, not a marker error.
                "displacement_m": measured if truth == "displaced" else None,
            }
        )
    return rows


def decide(row, condition, tau):
    """Return (claims_current, captures, inference_calls)."""
    if condition == "memory_only":
        return True, 0, 0
    use_capture = condition == "always_verify" or (condition == "selective" and row["policy_verify"])
    if not use_capture:
        return True, 0, 0
    return row["sim"] >= tau, 1, row["inference_calls"]


def evaluate(rows, condition, tau):
    m = Counter()
    for r in rows:
        claim, caps, calls = decide(r, condition, tau)
        m["n"] += 1
        m["captures"] += caps
        m["calls"] += calls
        if claim and r["truth"] == "present":
            m["correct_confirm"] += 1
        elif claim:  # absent, displaced or replaced: the item is not at the marker
            m["wrong_confirm"] += 1
        elif r["truth"] == "present":
            m["uncertain_present"] += 1
        else:
            m["uncertain_absent"] += 1
    return m


def fmt_table(rows_by_cond):
    head = f"{'condition':<15}{'n':>4}{'correct':>9}{'WRONG':>7}{'unc|pres':>10}{'unc|abs':>9}{'capt/q':>8}{'calls/q':>9}"
    lines = [head, "-" * len(head)]
    for cond, m in rows_by_cond.items():
        n = max(m["n"], 1)
        lines.append(
            f"{cond:<15}{m['n']:>4}{m['correct_confirm']:>9}{m['wrong_confirm']:>7}"
            f"{m['uncertain_present']:>10}{m['uncertain_absent']:>9}{m['captures'] / n:>8.2f}{m['calls'] / n:>9.2f}"
        )
    return "\n".join(lines)


def sweep(rows):
    pres = [r["sim"] for r in rows if r["truth"] == "present"]
    absn = [r["sim"] for r in rows if r["truth"] != "present"]
    out = []
    for i in range(6, 20):
        tau = i * 0.05
        tpr = sum(s >= tau for s in pres) / len(pres) if pres else float("nan")
        fpr = sum(s >= tau for s in absn) / len(absn) if absn else float("nan")
        out.append((tau, tpr, fpr))
    return out


def pick_threshold(rows):
    """Youden's J on dev rows; ties go to the higher (more conservative) threshold."""
    best = None
    for tau, tpr, fpr in sweep(rows):
        if tpr != tpr or fpr != fpr:  # NaN: need both classes
            continue
        j = tpr - fpr
        if best is None or j >= best[0]:
            best = (j, tau)
    return best[1] if best else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.1f}" if xs else "-"


def system_summary(events, excluded):
    ev = [e for e in events if e.get("session") not in excluded]
    regs = [e for e in ev if e["type"] == "register"]
    vers = [e for e in ev if e["type"] == "verify"]
    print("== System summary ==")
    print(f"sessions: {len({e['session'] for e in ev})}   registrations: {len(regs)}   "
          f"cancelled: {sum(e['type'] == 'register_cancel' for e in ev)}   "
          f"rejected hits: {sum(e['type'] == 'hit_rejected' for e in ev)}")
    if regs:
        print("crop source:", dict(Counter(e["boxSource"] for e in regs)))
        print("hit type:   ", dict(Counter(e["hitType"] for e in regs)))
        print("name source:", dict(Counter(e.get("nameSource", "n/a") for e in regs)))
        print(f"with room reference: {sum(bool(e['refAvailable']) for e in regs)}/{len(regs)}")
        print("register latency ms (median): convert", med([e["latency"]["convertMs"] for e in regs]),
              " detect", med([e["latency"]["detectMs"] for e in regs]),
              " embed", med([e["latency"]["embedMs"] for e in regs]))
        print("stored bytes/object (median):", med([e["recordBytes"] for e in regs]))
    if vers:
        print("verify latency ms (median): total", med([e["latency"]["totalMs"] for e in vers]),
              " detect", med([e["latency"]["detectMs"] for e in vers]),
              " embed (all crops)", med([e["latency"]["embedMs"] for e in vers]),
              f"   inference calls/verify (median): {med([e['inferenceCalls'] for e in vers])}")
    locks = [e for e in ev if e["type"] == "reference_locked"]
    if locks:
        print("reference relock ms (median):", med([e["sinceFirstFrameMs"] for e in locks]), f"(n={len(locks)})")
    print()


SOURCE_NOTE = {"REFERENCE": "REFERENCE (v0.2 picture)", "MAP": "MAP (v0.3 home map)"}


def cm(x):
    return f"{x * 100:.0f} cm"


def map_report(events, excluded, rows):
    """v0.3 sections. Printed after the per-query table so older output is unchanged."""
    ev = [e for e in events if e.get("session") not in excluded]

    regs = [e for e in ev if e["type"] == "register"]
    print("\n== Home map (v0.3) ==")
    with_map = sum(e.get("mapPosition") is not None for e in regs)
    print(f"registrations with a map position: {with_map}/{len(regs)}")
    fixes = [e for e in ev if e["type"] == "map_fix"]
    applied = [e for e in fixes if e.get("applied", True)]
    kinds = Counter(e.get("kind", "?") for e in applied)
    quality = Counter(e.get("quality", "?") for e in applied)
    if fixes:
        print("applied fixes by kind:", dict(kinds), "  by quality:", dict(quality),
              "  by type:", dict(Counter(e.get("fixType", "?") for e in applied)))
    else:
        print("map fixes: none")

    print("\n== Marker error by position source ==")
    by_src = defaultdict(list)
    for r in rows:
        if r["marker_error_m"] is not None:
            by_src[r["marker_source"] or "unknown"].append(r)
    if not by_src:
        print("  none")
    for src in sorted(by_src):
        rs = by_src[src]
        errs = [r["marker_error_m"] for r in rs]
        horiz = [r["marker_horizontal_m"] for r in rs if r["marker_horizontal_m"] is not None]
        line = f"  {SOURCE_NOTE.get(src, src):<26} n={len(rs):<3} median {cm(statistics.median(errs))}  max {cm(max(errs))}"
        if horiz:
            line += f"  (horizontal median {cm(statistics.median(horiz))})"
        print(line)

    print("\n== Map fix jumps (refresh and manual fixes while located) ==")
    jumps = [e for e in applied if e.get("jumpM") is not None]
    if not jumps:
        print("  none")
    else:
        for label, js in [("all", jumps)] + [(t, [e for e in jumps if e.get("fixType") == t]) for t in ("IMAGE", "SPOT")]:
            if not js:
                continue
            m = [e["jumpM"] for e in js]
            d = [e["jumpDeg"] for e in js if e.get("jumpDeg") is not None]
            walked = [e["metersSincePrevFix"] for e in js if e.get("metersSincePrevFix") is not None]
            print(f"  {label:<6} n={len(js):<3} jump median {cm(statistics.median(m))} / {statistics.median(d):.1f} deg   "
                  f"max {cm(max(m))} / {max(d):.1f} deg   walked since previous fix (median) {med(walked)} m")
    asked = [e for e in fixes if e.get("disagreement")]
    if asked:
        print(f"  disagreements asked: {len(asked)}   accepted: {sum(bool(e.get('accepted')) for e in asked)}   "
              f"kept current: {sum(e.get('accepted') is False for e in asked)}")

    print("\n== Queries by guidance mode at query start ==")
    modes = defaultdict(list)
    for r in rows:
        if r["guidance_mode"]:
            modes[r["guidance_mode"]].append(r)
    if not modes:
        print("  none (no guidance mode logged; pre-v0.3 data)")
    for mode in ("NO_FIX", "FAR_ROUTE", "FAR_NO_ROUTE", "NEAR"):
        rs = modes.get(mode)
        if not rs:
            continue
        reached = [r for r in rs if r["near_reached"]]
        t_near = [r["time_to_near_ms"] / 1000 for r in reached if r["time_to_near_ms"] is not None]
        route = [r["route_m"] for r in rs if r["route_m"] is not None]
        print(f"  {mode:<13} n={len(rs):<3} NEAR reached {len(reached)}/{len(rs)}   "
              f"time to NEAR (median) {med(t_near)} s   route at start (median) {med(route)} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--dev-session", action="append", default=[], help="sessions used only to pick the threshold")
    ap.add_argument("--exclude-session", action="append", default=[])
    ap.add_argument("--csv")
    a = ap.parse_args()

    events = load_events(a.data)
    excluded = set(a.exclude_session)
    system_summary(events, excluded)
    rows = build_queries(events, excluded)

    tau = a.threshold
    if a.dev_session:
        dev = [r for r in rows if r["session"] in a.dev_session and r["sim"] is not None]
        picked = pick_threshold(dev)
        if picked is None:
            print("Dev sessions need both present and absent queries with captures; keeping", tau)
        else:
            tau = picked
            print(f"Threshold picked on dev sessions {a.dev_session}: {tau:.2f}")
        rows = [r for r in rows if r["session"] not in a.dev_session]
        print("Reporting on held-out sessions only.\n")

    new_truths = "".join(f", {t} {sum(r['truth'] == t for r in rows)}" for t in ("displaced", "replaced")
                         if any(r["truth"] == t for r in rows))
    print(f"== Queries with ground truth: {len(rows)} "
          f"(present {sum(r['truth'] == 'present' for r in rows)}, absent {sum(r['truth'] == 'absent' for r in rows)}"
          f"{new_truths}) ==")
    paired = [r for r in rows if r["sim"] is not None]
    unpaired = len(rows) - len(paired)
    print(f"Paired (have a fresh capture): {len(paired)}; without capture: {unpaired}")
    print(f"Policy said VERIFY on {sum(r['policy_verify'] for r in paired)}/{len(paired)} paired queries\n")

    if paired:
        print(f"== Conditions on the paired set (threshold {tau:.2f}) ==")
        print(fmt_table({c: evaluate(paired, c, tau) for c in ("memory_only", "always_verify", "selective")}))
        print("WRONG = current-location claim when the object was absent (the error the study targets).")
        print("unc|pres = present but not confirmed (lost retrieval); unc|abs = correctly withheld.\n")
        if new_truths:
            print("displaced and replaced count as absent above: the item is not at the marker "
                  "(WRONG, unc|abs and FPR include them).\n")

        print("== Threshold sweep (always_verify): TPR = present confirmed, FPR = absent confirmed ==")
        for t, tpr, fpr in sweep(paired):
            print(f"  tau {t:.2f}   TPR {tpr:5.2f}   FPR {fpr:5.2f}")
        print()

    errs = [r["marker_error_m"] for r in rows if r["marker_error_m"] is not None]
    if errs:
        print(f"== Marker error (n={len(errs)}): median {statistics.median(errs) * 100:.0f} cm, "
              f"max {max(errs) * 100:.0f} cm ==\n")
    moved = [r["displacement_m"] for r in rows if r["displacement_m"] is not None]
    if moved:
        print(f"== Displacement of displaced items (n={len(moved)}): median {statistics.median(moved) * 100:.0f} cm, "
              f"max {max(moved) * 100:.0f} cm ==\n")

    print("== Per query ==")
    for r in sorted(rows, key=lambda r: (r["session"], r["query"])):
        sim = "-" if r["sim"] is None else f"{r['sim']:.2f}"
        print(f"  {r['session']} {r['query']} {r['name'][:14]:<14} truth={r['truth']:<7} "
              f"policy={'V' if r['policy_verify'] else '-'} src={r['position_source']:<9} sim={sim}")

    held_out = excluded | set(a.dev_session)
    map_report(events, held_out, rows)

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["query"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {a.csv}")

    far = sum(r["marker_error_m"] is not None and r["marker_error_m"] > R_PRESENT_M for r in rows)
    if far:
        print(f"\nNOTE: {far} 'present' quer{'y has' if far == 1 else 'ies have'} a measured marker error above "
              f"{R_PRESENT_M * 100:.0f} cm. The study spec counts those as not at the marker\n"
              "(a lead to them is a wrong lead); the tables above count every 'present' as present. "
              "paper_figures.py applies the rule.")


if __name__ == "__main__":
    main()
