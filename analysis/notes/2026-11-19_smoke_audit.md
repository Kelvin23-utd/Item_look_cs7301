# Visual audit of the two-pass smoke test

Date: 2026-11-19. One reading by the project's assistant, at the request of the person who ran the test. Data: the private pull of the smoke test (scans `e4144f49` = pass 1 and `03f9e780` = pass 2; see `2026-11-19_smoke_test.md`).

**Ground truth from the person who ran it** (not derived from the app): between the passes one cup was moved and nothing else changed; pass 2 covered only two of the rooms that pass 1 covered.

**What was looked at.** For each of the 38 saved entries: its stored photo and up to four retained views per pass (contact sheets made in a scratch folder outside the repository, not kept); larger views of three cup entries and of the cup-like detections that pass 2 dropped; and the old place of one cup cut from two pass-2 keyframes. 101 of the 221 observations were looked at and labelled with a physical-item code in a private label file kept with the data; the other 120 are recorded as unknown.

## The moved cup

- In pass 1 it is entry `e4af587b` (3 observations).
- In pass 2 its old place was looked at in ten keyframes and is empty; the app marked the entry **missed**, which is right.
- In pass 2 the cup was detected **once**, 0.67 m from its old place and showing its other side (observation `03f9e780-48-1`, score 0.55). A new entry needs two sightings, so its tentative entry was dropped at the end of the scan. **The app saved nothing for the cup at its new place.**
- By the phone's embedding, that one sighting is closer to the cup's own pass-1 observations than to any other of the 82 pass-1 entries and tentative entries: cosine 0.57 (0.52, 0.57, 0.57 against its three observations), next best 0.43 (another cup 0.23 m from the new place). 0.57 is under the 0.70 that the app's switched-off "moved candidate" step would require.

## The 10 entries that pass 2 created

| Pass-2 entry (label) | What it is | Reading |
| --- | --- | --- |
| `919272fa` (book) | the same object as pass-1 entry `d4e61a45`, 0.06 m away, which was also matched in pass 2 | **duplicate** |
| `02a863d1` (bowl) | an object that pass 1 had merged into entry `a571d2c0` (one of that entry's two pass-1 views shows it) | first entry of its own |
| `6074aeea`, `d0377680` (bowl), `389d29da` (carrot), `cef516de` (cell phone), `640be207` (cup), `c631578a` (keyboard), `1fb71dad` (laptop), `9d1eb7cc` (suitcase) | objects that no pass-1 entry shows | first-time detections of unchanged objects |

So 1 of the 10 is a duplicate, and 9 are objects that pass 1 had not saved. The new "cup" entry `640be207` is not the moved cup.

## The 28 pass-1 entries after pass 2

| App outcome | n | Reading |
| --- | --- | --- |
| seen | 17 | 11 show the same object in the views of both passes; 2 were "seen" by the appearance check with no detection; 4 have no pass-2 view or an unclear one (unknown) |
| missed | 2 | `e4af587b`: right (the moved cup). `91861bef`: by the ground truth it did not move, so this is a **false "missed"**; its place was checked in three keyframes with similarity 0.49 to 0.52, and those keyframes have no image (a person was detected), so it could not be looked at |
| not observed | 9 | consistent with pass 2 covering fewer rooms |

## Mixed entries seen along the way

At least five entries hold views of more than one object within a single pass (for example an entry whose pass-1 views show two different containers, and a keyboard entry with views of two keyboards). Two "bottle" entries 0.1 m apart may show the same bottle; recorded as unknown.

## What this changes

- The earlier reading of the logs alone ("the matcher over-creates") was too strong: 9 of the 10 new entries are real objects first saved in pass 2. The detector simply saves a different subset of objects on each pass.
- For the one moved object, the failure is not a wrong match but a **lost sighting**: a single detection at the new place is discarded before any matching.
- The appearance evidence for the moved cup is weak in absolute terms (0.57) but clear in rank (first of 82, margin 0.14), and the place evidence is strong (old place seen empty). A rule that uses rank, margin and the empty old place would link it; a fixed threshold of 0.70 would not.

n: one moved object, one test pass. This is a description of one case, not a rate.
