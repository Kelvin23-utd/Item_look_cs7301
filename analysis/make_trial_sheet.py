#!/usr/bin/env python3
"""Trial sheet for one round of the controlled paired trials, E3 (stdlib only).

From records.json, a round number and a seed it writes what the experimenter does with each
item, the order of the queries, and before which queries to walk a long loop.

  action        default share   ground truth to enter at Done
  leave         50%             At the marker (then tap the item's true centre)
  remove        20%             Not in this area
  move nearby   15%             Nearby, but not at the marker (then tap the item)
  replace       15% together    Another object is at the marker
  swap                          Another object is at the marker

Independence. The change must not depend on any signal a capture policy may use (memory age,
mobility class, earlier moves, position source, metres walked, tracking losses and time since
the fix, fix type). So the draw sees nothing but the record ids:
  * draw() takes the sorted ids, the round and the seed, and no other field of a record;
  * the actions are a fixed multiset (the shares above, rounded) in a uniformly random order,
    so every item has the same chance of every action;
  * the query order, the long loops and the replace-or-swap choice each come from their own
    random stream, so none of them tells anything about the actions.
Names and rooms are read only to print them.

The long loop exists to vary the drift signals within a round (metres and time since the fix)
without tying them to the changes.

Usage:
  python3 analysis/make_trial_sheet.py findback_data --round 1 --seed 20261119
  python3 analysis/make_trial_sheet.py path/to/records.json --round 2 --seed 7 --tag "home=A;exp=E3;round=2;split=test"
Writes <out-dir>/trial_sheet_round<R>_seed<S>.md and .csv (default out-dir: analysis/trial_sheets).
The same records, round and seed always give the same sheet.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys

ACTIONS = ("leave", "remove", "move", "replace_or_swap")
DEFAULT_MIX = {"leave": 0.50, "remove": 0.20, "move": 0.15, "replace_or_swap": 0.15}
DEFAULT_LOOP_SHARE = 0.20
INSTRUCTION = {
    "leave": "Leave it where it is.",
    "remove": "Take it out of the room, out of sight.",
    "move": "Move it 0.5 to 1.5 m away. It must stay visible from its old place.",
    "replace": "Take it away and put a clearly different object on its place.",
    "swap": "Take it away and put its look-alike on its place. No look-alike: use a different object and write R in Notes.",
}
EXPECTED = {
    "leave": ("present", "At the marker, then tap its true centre"),
    "remove": ("absent", "Not in this area"),
    "move": ("displaced", "Nearby, but not at the marker, then tap the item"),
    "replace": ("replaced", "Another object is at the marker"),
    "swap": ("replaced", "Another object is at the marker"),
}


def stream(seed, round_no, purpose):
    """An independent random stream for one purpose."""
    digest = hashlib.sha256(("findback-trial-sheet|%s|%s|%s" % (seed, round_no, purpose)).encode()).hexdigest()
    return random.Random(int(digest, 16))


def shuffled(items, rng):
    """Fisher-Yates, written out so the result does not depend on the Python version."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = rng.randrange(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def counts_for(n, mix, rng):
    """How many of n items get each action: the shares rounded by largest remainder, with ties
    for the last places broken at random."""
    total = sum(mix.values())
    exact = {a: n * mix[a] / total for a in ACTIONS}
    counts = {a: int(exact[a]) for a in ACTIONS}
    left = n - sum(counts.values())
    by_remainder = sorted(shuffled(ACTIONS, rng), key=lambda a: -(exact[a] - counts[a]))
    for a in by_remainder[:left]:
        counts[a] += 1
    return counts


def draw(ids, round_no, seed, mix=None, loop_share=DEFAULT_LOOP_SHARE):
    """The whole randomisation. `ids` is all it knows about the items.

    Returns (actions, order, loops): actions maps id -> leave / remove / move / replace / swap;
    order is the list of ids in query order; loops is the set of ids before whose query the
    experimenter walks a long loop.
    """
    mix = dict(DEFAULT_MIX if mix is None else mix)
    ids = sorted(ids)
    n = len(ids)
    counts = counts_for(n, mix, stream(seed, round_no, "counts"))
    labels = shuffled([a for a in ACTIONS for _ in range(counts[a])], stream(seed, round_no, "actions"))
    kind = stream(seed, round_no, "replace-or-swap")
    actions = {}
    for rid, a in zip(ids, labels):
        actions[rid] = ("swap" if kind.random() < 0.5 else "replace") if a == "replace_or_swap" else a
    order = shuffled(ids, stream(seed, round_no, "order"))
    n_loops = int(round(loop_share * n))
    loops = set(shuffled(ids, stream(seed, round_no, "loops"))[:n_loops])
    return actions, order, loops


def load_records(path):
    """(records, records.json path, {room id: room name}). `path` is the file or its folder."""
    rec_path = os.path.join(path, "records.json") if os.path.isdir(path) else path
    with open(rec_path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("%s is not a list of records" % rec_path)
    rooms = {}
    map_path = os.path.join(os.path.dirname(os.path.abspath(rec_path)), "map.json")
    if os.path.exists(map_path):
        try:
            with open(map_path, encoding="utf-8") as f:
                rooms = {r.get("id"): r.get("name") for r in json.load(f).get("rooms") or []}
        except (OSError, ValueError):
            rooms = {}
    return [r for r in records if isinstance(r, dict) and r.get("id")], rec_path, rooms


def parse_mix(text):
    mix = dict(DEFAULT_MIX)
    for part in text.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in mix:
            raise ValueError("unknown action %r in --mix (use %s)" % (k, ", ".join(ACTIONS)))
        mix[k] = float(v)
    if min(mix.values()) < 0 or sum(mix.values()) <= 0:
        raise ValueError("--mix shares must be non-negative and not all zero")
    return mix


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Trial sheet for one round of the controlled paired trials (E3).")
    ap.add_argument("records", help="records.json, or the data folder that holds it")
    ap.add_argument("--round", type=int, required=True, dest="round_no")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mix", default="", help="shares, e.g. leave=0.5,remove=0.2,move=0.15,replace_or_swap=0.15 (the default)")
    ap.add_argument("--loop-share", type=float, default=DEFAULT_LOOP_SHARE,
                    help="share of queries preceded by a long loop (default %(default)s)")
    ap.add_argument("--only", action="append", default=[], help="use only this record id (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], help="leave this record id out (repeatable)")
    ap.add_argument("--tag", default="", help="study tag to print on the sheet, e.g. home=A;exp=E3;round=2;split=test")
    ap.add_argument("--out-dir", default=os.path.join(here, "trial_sheets"))
    a = ap.parse_args(argv)

    try:
        records, rec_path, rooms = load_records(a.records)
        mix = parse_mix(a.mix) if a.mix else dict(DEFAULT_MIX)
    except (OSError, ValueError) as e:
        print("ERROR:", e, file=sys.stderr)
        return 1
    if any(r.get("synthetic") for r in records):
        print("NOTE: these records are marked synthetic; the sheet is for testing only.", file=sys.stderr)
    by_id = {r["id"]: r for r in records}
    unknown = [i for i in a.only + a.exclude if i not in by_id]
    if unknown:
        print("ERROR: no such record id: " + ", ".join(unknown), file=sys.stderr)
        return 1
    ids = [i for i in by_id if (not a.only or i in a.only) and i not in a.exclude]
    if not ids:
        print("ERROR: no items to put on the sheet", file=sys.stderr)
        return 1
    if not 0 <= a.loop_share <= 1:
        print("ERROR: --loop-share must be between 0 and 1", file=sys.stderr)
        return 1

    actions, order, loops = draw(ids, a.round_no, a.seed, mix, a.loop_share)

    def name(rid):
        return str(by_id[rid].get("name") or rid)

    def room(rid):
        r = by_id[rid].get("roomId")
        return str(rooms.get(r) or r or "")
    id_hash = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:12]
    done = {k: sum(v == k for v in actions.values()) for k in ("leave", "remove", "move", "replace", "swap")}

    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, "trial_sheet_round%d_seed%d" % (a.round_no, a.seed))
    md = []
    md.append("# FindBack trial sheet: round %d" % a.round_no)
    md.append("")
    md.append("Seed %d. %d items from `%s` (id list %s)." % (a.seed, len(ids), os.path.basename(rec_path), id_hash))
    md.append("Mix: " + ", ".join("%d %s" % (v, k) for k, v in done.items()) + "; %d long loop(s)." % len(loops))
    if a.tag:
        md.append("Study tag for this round: `%s`, capture mode PAIRED." % a.tag)
    md.append("")
    md.append("## 1. Change the items (keep the app closed)")
    md.append("")
    md.append("| Done | Item | Room | Do this |")
    md.append("| --- | --- | --- | --- |")
    for rid in sorted(ids, key=lambda i: (room(i), name(i).lower(), i)):
        md.append("| [ ] | %s (`%s`) | %s | %s |" % (name(rid), rid, room(rid), INSTRUCTION[actions[rid]]))
    md.append("")
    md.append("## 2. Reopen the app, set the tag, locate, then query in this order")
    md.append("")
    md.append("Press Verify once per query. Answer the ground truth from what you see. If it differs from the last column, "
              "enter what you see and say why in Notes.")
    md.append("")
    md.append("| # | Item | Room | Before the query | Scripted ground truth | Notes |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for k, rid in enumerate(order, 1):
        md.append("| %d | %s (`%s`) | %s | %s | %s | |" % (
            k, name(rid), rid, room(rid), "**Walk a long loop first**" if rid in loops else "", EXPECTED[actions[rid]][1]))
    md.append("")
    md.append("A long loop: walk through every room and back, about a minute, without visiting a fix point.")
    md.append("")
    md.append("## 3. Afterwards")
    md.append("")
    md.append("Put every item back exactly where it was saved.")
    md.append("")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "seed", "order", "record", "name", "room", "action", "long_loop_first", "scripted_truth"])
        for k, rid in enumerate(order, 1):
            w.writerow([a.round_no, a.seed, k, rid, name(rid), room(rid), actions[rid], int(rid in loops), EXPECTED[actions[rid]][0]])
    print("Wrote %s.md and .csv: %d items (%s), %d long loop(s)" % (
        base, len(ids), ", ".join("%d %s" % (v, k) for k, v in done.items()), len(loops)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
