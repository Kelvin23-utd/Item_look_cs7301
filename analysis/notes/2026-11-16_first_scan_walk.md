# First scan walk: what the log shows

Session `f52b04d2`, 2026-11-16, app v0.6 (scan-once Block A), one room, no study tag. One Locate,
one scan, five searches. Numbers below come from `events.jsonl`, `inventory.json` and the saved
crops of that session. It is one uncontrolled session: these are leads to test, not results.

The second scan of the planned test (move one item away, hide another, scan again) was not run, so
the "missing versus hidden" rule has no data yet.

## What worked

| Step | Evidence |
| --- | --- |
| Locate at the home spot | 10 samples, position spread 1.1 mm, heading spread 0.4° |
| Scan speed | 110 keyframes in 68.7 s (1.65 per second) while walking 18.6 m (0.27 m/s); 211 ms median per keyframe (detect 104, image conversion 70) |
| Depth geometry | Image centre against an ARCore depth hit: 1.1 cm median (n = 9); off-centre point: 2.9 cm median, 5.3 cm maximum (n = 20) |
| Repeat sightings of one item | Position differs by 4.1 cm median, 12.1 cm at the 90th percentile (n = 37) |
| Guidance | All 4 searches with a map position reached the ring; first Verify 10.5–17 s after starting; routes 8.1–8.3 m for 5.1–6.1 m straight |
| Item from an earlier session | An item saved 9 h earlier in another session was found at its marker (user's answer; no error measured) |

## Challenges the log points to

1. **Verification rejects items that are there.** All 5 captures were "not confirmed"
   (similarity 0.34, 0.39, 0.40, 0.54, 0.67; threshold 0.75), in 3 searches the user ended with
   "Yes, it's here". Over every session so far, 13 captures have the answer "present": 6 confirmed
   (0.78–0.86) and 7 rejected (0.34–0.67). The one capture answered "absent" scored 0.24. The crops
   show three different causes:
   - a correct, well-framed match that still scores 0.67 (threshold too strict for this embedder);
   - a different side of the object, or the neighbouring look-alike 28 cm away (0.40; the log
     cannot tell which);
   - a capture from 38 cm among five bottles, where the comparison window slides onto the
     neighbours (0.54, 0.34).
2. **Appearance is unstable even seconds apart.** During the scan, repeat sightings of the same
   entry score 0.74 median, 0.51 at the 10th percentile; only 46% reach 0.75 and 76% reach 0.60
   (the "seen again" threshold). Several scan crops are visibly motion-blurred.
3. **Names.** Of the 24 entries, by the auditor's reading of the crops (not confirmed by the
   user): 12 are distinct items with a fitting name, 9 have a wrong name (for example a paper
   towel roll as "cup", a yellow bag as "banana", a snack bag as "book"), 3 are duplicates or parts
   of another entry. Only 10 of the 80 class names were ever used.
4. **Duplicates.** One appliance became two entries 0.5 m apart, one bottle two entries 3 cm apart,
   and a laptop's keyboard a separate entry.
5. **Low yield per frame.** 638 detections gave 105 kept (16%): 346 were below the score limit, 93
   on the furniture stop list, 63 "person", 30 too large. 63 of 110 keyframes kept nothing. 20 of
   44 candidate items were seen in one keyframe only and dropped.
6. **"Person" detections.** 37 of 110 keyframes had at least one. Those frames are not saved and
   overlapping boxes are dropped, by design. Whether a person was in view is unknown; plush toys
   are in the room and one was detected as a person in an earlier test.
7. **Depth settles late.** Against ARCore plane hits the depth map read 15–20 cm far in keyframes
   4–9 (the first seconds) and within 4 cm from keyframe 11 on; one 55 cm outlier at a depth
   edge. The depth image is 160×90 pixels for a 1920×1080 camera image.

## Not measured in this session

Marker error after a restart, drift with distance from the spot, a second room, the "missing
versus hidden" rule, and how many items in the room the scan did not list at all.
