# Identity replay: design, parameters and a development run

Date: 2026-11-19. Script: `analysis/identity_replay.py`. Tests: `analysis/test_identity_replay.py` (14 tests, all pass under `python3` and `analysis/.venv-bench/bin/python`). Protocol: `paper/IDENTITY_PILOT_PROTOCOL.md`, Part B, "Planned comparison". Data format: `FindBack/docs/PLAN_v0.8_study_logging.md`.

This note describes what the script does, every parameter and where its value came from, how outcomes are scored, and what one development run on the smoke data showed. The smoke run tests the code and the rules. It is not evidence about any method: one room, two passes, one moved object.

## What it does

Input: one pulled data folder, the history scan ids and the test scan ids, in order. Optional private inputs: an observation label file (`obs,item`), an entry label file (`entry,item`) and the filled ground-truth sheet.

The known items are the app's real entries of the map alive at the end of the last history scan, rebuilt from `inventory_log.jsonl` (with the `evidence` lines' history items applied). Every test scan is decided against that state alone. Test scans do not feed each other, so T2 does not see what T1 decided.

For each test scan and each method the script gives:

- one decision per sighting: `update <item>`, `add`, or `unresolved <candidates>`. The current matcher decides per observation and also has `drop`: the observation went to a tentative entry that the scan dropped, so nothing was saved;
- what the method then believes about each known item: seen in place, seen by the phone's appearance check, moved (with the sightings and the corrected position), missed (its place was looked at and found empty), not observed, or unresolved (a candidate of an unresolved sighting);
- the alignment correction it estimated;
- the cost: unresolved sightings (what would go to a server), the distinct candidate items in them, the pack's size and build time, and the compute time.

Without label files it reports the label-free part: decisions, single-observation sightings, beliefs, agreement per observation between every pair of methods, and the list of sightings on which the methods disagree. With labels it adds the scored outcomes (below).

Output: a report on stdout and `analysis/out/identity_replay/<folder>/result.json` with the full parameter set, the git commit, the script's sha256, every decision and belief, and the scoring. The pack files go to the same folder. Both folders are git-ignored because they hold positions and item codes.

No image is read. Every similarity is the cosine of two logged embeddings (`emb` of the observation lines, `embedding` of the entries).

## Sightings

A sighting does not depend on the current matcher's entries. It is built from the phone's view groups (`viewGroup`), which the retention rule makes without the matcher: group anchors are at least 0.30 m apart, and every observation joins the group of the nearest anchor within 0.30 m. Within a group, observations are taken in file order. Each one joins the most similar sighting of its group, if that sighting has no observation from the same keyframe and the cosine to one of its members is at least `SIGHTING_SIM`. Otherwise it starts a new sighting. Its position is the mean of its map points, and its labels are the score-weighted votes of its observations.

No sighting is dropped, however few observations it has. A single detection is a sighting of its own and goes through every rule. The current matcher, by contrast, drops an entry seen only once: its tentative entry needs a second sighting at least 300 ms later.

An extra view is a sighting of its own, at `target.mapPoint`, the place the camera was aimed at. That place is read in the frame of the session that captured it, the same as the test scan. An extra view has no detector label.

## Methods

**current.** The logged `decision` of each test observation, which `analysis/study_check.py` re-checks through `analysis/scan_replay.py`. `known` to an entry that existed before the scan is `update`. An entry the scan created (listed in `scan_end.created`) is `add`, and so is every observation linked to it. An entry in `scan_end.tentativeDropped` is `drop`. Beliefs come from the scan's `inv_evidence` outcomes: seen, missed or not observed.

**generic.** Rules that need no knowledge of this home. Each item is an entry with its last confirmed place (`confirmed.mapPosition`, else `mapPosition`), its one stored embedding and the fixed thresholds.

- (a) Alignment between launches. Candidate pairs: a sighting and an item with the same detector label, within `ALIGN_SEARCH_M`, at cosine `ALIGN_MIN_SIM` or more. Only mutual-best pairs are kept.
  - The fit starts with a translation, the median of the differences per axis. Pairs within `ALIGN_INLIER_M` of it agree with it.
  - With at least `ALIGN_MIN_PAIRS_YAW` agreeing pairs spread at least `ALIGN_MIN_SPAN_M` horizontally, a yaw about +Y is also fitted (least squares on the agreeing pairs, refitted once). The yaw is kept only if it is within `ALIGN_MAX_YAW_DEG` and its median residual over all pairs is smaller than the translation's.
  - A correction is refused when there are fewer than `ALIGN_MIN_PAIRS` pairs, when no translation has that many agreeing pairs, when the centre of the agreeing pairs moves more than `ALIGN_MAX_SHIFT_M`, or when the median residual does not fall.
  - The correction maps test-frame points into the history frame. Its inverse puts each item's place into the test frame for the look-at test. `--align off` (or `both`) switches it off.
- (b) In place. A sighting within the item's radius (`PLACE_RADIUS_M`) of the item's place, with cosine `SIM_PLACE` or more to its references, can update it; no label is required.
- (c) Exclusivity. The in-place pairs are taken best cosine first, then nearest. A sighting updates at most one item. An item takes several sightings only if no two of them share a keyframe: two detections of one keyframe are two objects. An item matched in place is no candidate anywhere else in the scan.
- Place looked at. For each item not matched in place, the script counts the keyframes in which its place, in the test frame, was observable. It applies the app's own rule (`EvidenceRules.classify`): inside the image less 8 % at each side, 0.3 to 3.0 m along the optical axis, and not hidden by nearer depth. Depth comes from the saved depth image of that keyframe. Without one, the script uses the phone's own logged check of that entry in that keyframe (`scan_keyframe.checks`); otherwise the keyframe says nothing. An item whose expected window the phone checked at `SEEN_SIMILARITY` or more is "seen by appearance" and stays in place. Otherwise an item is missed with at least `MISSED_MIN_KEYFRAMES` observable keyframes, and not observed with fewer. An extra view aimed within the item's radius counts as `EXTRA_VIEW_LOOKS` looks.
- (d) Moved. The remaining sightings are taken strongest moved candidate first. The candidates are the missed items with the sighting's detector label (`ELSEWHERE_SAME_LABEL`) that no other sighting has taken. The best one is updated, as moved, if its cosine reaches its moved threshold (`SIM_MOVED` for generic) and beats every other candidate by `MOVED_MARGIN`. The other candidates are missed or not-observed items with the same label, other than the best and its aliases. A later sighting within the radius of an item's new place, sharing no keyframe with it and at `SIM_PLACE` or more, joins that item.
- (e) Not looked at. An item whose place was not looked at keeps its last confirmed place. It can be a runner-up and a listed candidate, so it can make a sighting unresolved, but it is never updated.
- (f) Otherwise. If any candidate reaches `SIM_UNRESOLVED`, the sighting is `unresolved`, listing up to `UNRESOLVED_MAX_CANDIDATES` items. If none does, it is `add`.
- An extra view is only ever matched in place. It never adds or moves an item.

**household.** The same rules, read from a pack built only from the history scans. The method reads only the pack file and the test scan. The pack holds:

- History frames. Each later history scan is aligned to the first with the same robust fit, on the history positions of entries the app linked in both scans. Places, positions and impostors are expressed in the first scan's frame.
- References: the entry's stored embedding, its `confirmed.refs`, and the views of later history items whose view is retained, all as float16. A view joins only at `REF_CONSISTENCY_SIM` or more to the entry's own embedding, which drops views of a swapped neighbour.
- Look-alike groups. Two items are look-alikes when the median cosine between their references is at least the smaller of their self-similarities: the median between an item's own references, or `SELF_FALLBACK` with one reference. Look-alikes closer than `PLACE_RADIUS_M` are aliases: one object with several entries. If one alias is seen in place, all of them are, and an alias is never a runner-up of another.
- A threshold per item and group. The impostor cosine of an item is the highest cosine between its references and any history observation at least `FAR_M` from its place that the app did not link to a member of its group. The item threshold is the impostor cosine plus `HH_IMPOSTOR_MARGIN`, at least `HH_THR_MIN`, plus `STABLE_MOVED_BONUS` when the item is stable. With no impostor it is `SIM_MOVED`. Over `HH_THR_MAX` the item can only be unresolved. A group uses the highest threshold of its members.
- Stability prior. An item's spread is the largest distance of its aligned history positions from their mean, which needs two history scans. A spread of at most `STABLE_SPREAD_M` makes the item stable. The place radius is the spread plus `STABILITY_PAD_M`, kept between `PLACE_RADIUS_M` and `PLACE_RADIUS_MAX_M`. With one history scan there is no prior: every radius is `PLACE_RADIUS_M` and no bonus applies.

**extra view.** The household method with each extra view logged after the test scan as one more sighting. An extra view belongs to a test scan when its time lies after that scan's end and before the next scan of any session on the map. The report counts how many sighting decisions change against household, in which direction (for example `unresolved -> update`), and the logged interruption times: request to capture (`msSinceQueryStart`) and the whole query (`query_end.t − requestStartMs`), as a median and range with n. The extra view's own `entry` (the queried entry) and the user's answer are not used to decide.

## Parameters

All are in `PARAMETERS` at the top of the script. They are printed with every run and written into the result file. `--freeze FILE` writes them with the commit and the script's sha256. `--frozen FILE` refuses to run when any parameter differs. If only the script changed, it prints a note, so that a change of rules is reported as a change. "Pass 1" is scan `e4144f49` of the smoke pull, the only data used to choose values. Nothing was chosen from its pass 2 or from the audit labels.

| Name | Value | Where it came from |
| --- | --- | --- |
| `SIGHTING_SIM` | 0.4 | an observation joins a sighting of its view group only at this cosine to one of its members: the same value as SIM_PLACE |
| `PLACE_RADIUS_M` | 0.3 | between launches the same entries moved by a median of 0.12 m, at most 0.29 m (notes/2026-11-19_smoke_test.md); in pass 1, pairs of observations with cosine >= 0.75 and the same label lay a median 0.04 m, 95th percentile 0.24 m apart (n = 53) |
| `SIM_PLACE` | 0.4 | pass 1: 95th percentile (0.402) of the cosine between observations of the same label more than FAR_M apart, which are different objects (n = 1209 pairs); 94 % of pairs of the same label within 0.10 m in different keyframes lie above it (n = 83) |
| `SIM_MOVED` | 0.6 | pass 1: the highest cosine (0.598) between observations of the same label more than FAR_M apart (n = 1209); the app's SEEN_SIMILARITY is also 0.60 |
| `MOVED_MARGIN` | 0.1 | chosen, not derived: the best moved candidate must beat every other candidate by this much |
| `SIM_UNRESOLVED` | 0.4 | a candidate at or above this (SIM_PLACE) and not decided makes the sighting unresolved; below it the sighting is added |
| `UNRESOLVED_MAX_CANDIDATES` | 3 | candidates listed (and sent to a server) per unresolved sighting |
| `ELSEWHERE_SAME_LABEL` | true | moved and unresolved candidates need the sighting to carry the item's detector label (the app's moved rule also asks for the same label); in place, appearance alone decides |
| `MISSED_MIN_KEYFRAMES` | 3 | app ScanConfig.MISSED_MIN_KEYFRAMES |
| `VIEW_MARGIN` | 0.08 | app ScanConfig.VIEW_MARGIN |
| `VIEW_MIN_M` | 0.3 | app ScanConfig.VIEW_MIN_M |
| `VIEW_MAX_M` | 3.0 | app ScanConfig.VIEW_MAX_M |
| `DEPTH_PATCH` | 5 | app ScanConfig.DEPTH_PATCH |
| `HIDDEN_MIN_M` | 0.15 | app ScanConfig.HIDDEN_MIN_M |
| `HIDDEN_FRACTION` | 0.15 | app ScanConfig.HIDDEN_FRACTION |
| `USE_PHONE_APPEARANCE_CHECK` | true | an unmatched item whose expected window the phone checked (scan_keyframe.checks[].sim) at or above SEEN_SIMILARITY counts as seen in place, not missed |
| `SEEN_SIMILARITY` | 0.6 | app ScanConfig.SEEN_SIMILARITY |
| `ALIGN` | true | estimate the correction (switch off with --align off; --align both runs both) |
| `ALIGN_SEARCH_M` | 0.5 | pairs for the correction are searched this far: above the 0.29 m largest launch-to-launch difference of the smoke note; the app's largest merge radius |
| `ALIGN_MIN_SIM` | 0.6 | a pair for the correction needs a confident appearance match: SIM_MOVED |
| `ALIGN_MIN_PAIRS` | 3 | fewest mutual-best pairs for a translation |
| `ALIGN_INLIER_M` | 0.2 | a pair agrees with a correction within this residual: about the 90th percentile (0.18 m) of the pass-1 spread of very similar observations |
| `ALIGN_MIN_PAIRS_YAW` | 4 | fewest agreeing pairs to fit a yaw as well |
| `ALIGN_MIN_SPAN_M` | 0.5 | and their horizontal spread around their centre must reach this |
| `ALIGN_MAX_SHIFT_M` | 0.5 | a correction that moves the pairs' centre more than this is refused |
| `ALIGN_MAX_YAW_DEG` | 10.0 | a yaw larger than this is refused (falls back to the translation) |
| `FAR_M` | 1.0 | observations of one scan this far apart are different objects (the impostor set) |
| `REF_CONSISTENCY_SIM` | 0.4 | a history view becomes a reference of an entry only at this cosine to the entry's own stored embedding (SIM_PLACE): drops views of a swapped neighbour |
| `SELF_FALLBACK` | 0.75 | self-similarity of an item with one reference: pass-1 median (0.758) of pairs of the same label within 0.10 m in different keyframes (n = 83) |
| `HH_IMPOSTOR_MARGIN` | 0.05 | chosen: an item's moved threshold is its highest impostor cosine plus this |
| `HH_THR_MIN` | 0.4 | an item's moved threshold is never below SIM_PLACE |
| `HH_THR_MAX` | 0.9 | an item whose threshold would exceed this can only be unresolved, never moved |
| `STABLE_SPREAD_M` | 0.1 | an item whose history positions (aligned) spread at most this is stable |
| `STABLE_MOVED_BONUS` | 0.05 | a stable item needs this much more similarity to be moved |
| `STABILITY_PAD_M` | 0.1 | an item's place radius is its history spread plus this ... |
| `PLACE_RADIUS_MAX_M` | 0.5 | ... between PLACE_RADIUS_M and this |
| `PACK_EMBEDDING` | "float16" | references are stored as little-endian float16 in base64 |
| `EXTRA_VIEW_LOOKS` | 3 | an extra view aimed within an item's place radius counts as this many looks at the place (MISSED_MIN_KEYFRAMES: the user was guided there) |

The pass-1 numbers in the table were computed over the observation lines of scan `e4144f49`: pairs from different keyframes, cosine of the logged embeddings, distance between map points.

## Scoring

Physical item codes come from the observation label file. Observations without a label or labelled `unknown` are left out of scoring and counted. Each history entry is mapped to an item in one of two ways:

- from the entry label file, when given;
- otherwise, the majority item over the labelled observations the app linked to it in the history (its history items and `confirmed.refs`). A tie, or no labelled observation, leaves the entry unmapped, and the script counts each reason.

An item is *known* if a history entry maps to it.

The condition of each item in test scan *k* comes from the sheet. The columns `h1…`, `h2…` belong to the history scans in order, and `t1…`, `t2…` to the test scans.

- `not_visited`: a known item with `t_visited` no.
- `absent`: a known item with `t_present` no.
- `moved`: a known item whose place code differs from the last history scan's, or, without place codes, whose `t_moved` is yes.
- `stays`: any other known item.
- `new_same_model`: an item that is not known, has a same-model group, and is absent from every history scan.
- `new_to_inventory`: any other item that is not known, such as an object present all along that the history never saved.

Items not on the sheet are `known (no sheet)` or `new (no sheet)`.

Each observation inherits its sighting's decision. An *entity* is what a decision links observations to: an existing entry for `update`; a new entry (current) or the sighting itself (generic and household) for `add`. Each item gets one outcome per method and test scan, taken in this order:

1. **wrong merge**: an entity holds observations of two items. This includes an existing entry of item X that takes an observation of item Y, and then both X and Y are wrong merges.
2. **kept, not observed**: a known item with no labelled test observation and no wrong merge.
3. **duplicate**: a known item whose observations got `add`.
4. **unresolved**: an item with an `unresolved` observation.
5. **correct update**: a known item whose observations updated one of its own entries.
6. **correct add**: a new item whose observations were only added.
7. **dropped** (current only): all its observations went to dropped tentative entries.
8. **unscorable**: the rest, for example an update of an entry that is not mapped to any item.

The report gives counts per condition with n, never rates. It also gives the belief about each known item's entries, the most positive over its entries in this order: seen in place, seen, seen by appearance, moved, unresolved, missed, not observed. One more line counts the keyframes in which two labelled observations show the same item, which breaks the assumption behind rule (c).

## Fixed now, frozen later

**Fixed now**, by this note and the commit that adds it:

- every rule above;
- every value in the parameter table;
- the sighting construction, the scoring order and the conditions.

**Frozen after H1 and H2**, before any test-scan folder is read:

- Run `python3 analysis/identity_replay.py --freeze <file outside git>` and keep the file with the commit.
- If the history scans show that a value or rule must change, change it then, freeze again, and record the change with its reason in a changes note.
- Run the pilot with `--frozen <file>`. A change after that is reported as a change, with both results (protocol, last paragraph).

**Built from H1 and H2 by the script, not chosen by hand** (part of the frozen procedure, written to the pack):

- the household references;
- the groups and aliases;
- the impostor cosines and the per-item thresholds;
- the history frames;
- the stability prior.

**Not decided yet:**

- whether T2 should be decided against the history only (as now) or also against T1. The app itself starts T2 from the state T1 left, including user confirmations;
- whether `MOVED_MARGIN` and the household margins should be derived from the history rather than chosen;
- whether the household references should include every linked history observation, not only those with a retained view. The protocol says retained views, and that is what is implemented. But every observation line carries its embedding, with or without a view (a view is skipped when a person is in the frame or when it adds no new direction).
  - On the smoke data the moved cup's entry has 3 linked pass-1 observations and only 1 retained view. Its two other observations are the closest to the moved cup: cosine 0.57 against 0.52 (the audit note, `2026-11-19_smoke_audit.md`, reports the same 0.57).
  - This was seen after the scored run and was not changed.

## Development run on the smoke data

```bash
python3 analysis/identity_replay.py backups/phone_files_2026-11-19_smoke_1752 --history e4144f49 --test 03f9e780 --align both --no-labels
python3 analysis/identity_replay.py backups/phone_files_2026-11-19_smoke_1752 --history e4144f49 --test 03f9e780 --align both \
    --sheet <private sheet> [--entry-labels <private entry labels>]
```

Pass 1 (`e4144f49`) is the only history scan and pass 2 (`03f9e780`) is the test scan. With one history scan there is no stability prior. The parameters were frozen (`--freeze`, in `analysis/out/`) before the labelled run, and the scored runs passed `--frozen`. The label file comes from a separate visual audit: 101 observations carry an item code and 120 are `unknown`.

There is no sheet from the people who ran the test. The sheet used here is a private file made from the operator's statement as relayed: one cup moved, nothing else changed, places and visits not recorded. So every item is `stays` except the moved cup, and items the history never saved are `new_to_inventory`. The audit also identified which pass-1 entry the moved cup is, and which pass-2 observation shows it at its new place: one detection, 0.67 m from its old place.

### Label-free

- **Size of the test scan:** 81 observations in 49 sightings. 31 sightings have one observation; the others have 2 (8), 3 (7), 4 (2) and 5 (1). 28 known items.
- **Geometry check:** recomputing the phone's own logged checks for entries the matcher never linked in pass 2 gives the same pixel and depth along the axis (n = 32, largest difference 0). It gives the same measured depth where a depth image was saved (n = 13, difference 0).
- **Alignment correction:**
  - generic: yaw −1.99° and t = (0.069, −0.068, −0.001) m, from 9 mutual-best pairs, 5 of which agree. The pairs' centre moved 0.084 m, and the median residual fell from 0.154 to 0.091 m (0.115 m with the translation alone).
  - household: yaw −1.12° and t = (0.011, −0.050, 0.009) m, from 11 pairs, 7 of which agree. The centre moved 0.056 m, and the median residual fell from 0.123 to 0.093 m.

Decisions:

| Method | Per sighting (n = 49) | Per observation (n = 81) | One-observation sightings (n = 31) |
| --- | --- | --- | --- |
| current | (per observation only) | update 36, add 25, drop 20 | update 10, add 2, drop 19 |
| generic | update in place 13, moved 1, add 31, unresolved 4 | update 27, add 49, unresolved 5 | update 7, add 21, unresolved 3 |
| household | update in place 13, moved 2, add 29, unresolved 5 | update 29, add 45, unresolved 7 | update 8, add 20, unresolved 3 |
| extra view | as household: no extra view was logged after pass 2 | | |

Switching the alignment off changed no decision of either method (81 of 81 observations agree). One sighting that was in place with the correction was "moved" without it.

Beliefs about the 28 known items:

| Method | Beliefs |
| --- | --- |
| current | seen 17, missed 2, not observed 9 |
| generic | seen in place 12, seen by appearance 2, moved 1, missed 3, not observed 7, unresolved 3 |
| household | seen in place 12, seen by appearance 2, moved 2, missed 3, not observed 7, unresolved 2 |

Agreement per observation (n = 81):

- current agrees with generic on 42 and with household on 35;
- generic agrees with household on 72;
- the methods disagree on 34 of the 49 sightings:
  - 20 are sightings in which the current matcher drops some or all of the observations;
  - 5 are sightings the current matcher updates and generic adds; 2 of them are laptop sightings the app merged by label;
  - 2 are two neighbouring bottle sightings, which generic and household give to each other's entries. The one labelled observation among them supports the app's choice, and the other entry has no labelled history observation;
  - 7 are other combinations.

Cost:

- generic: 4 unresolved sightings (3 distinct candidate items), about 31 ms to decide.
- household: 5 unresolved sightings (2 distinct items), about 62 ms. The pack is 165,850 bytes and builds in 120 to 150 ms on this Mac: 28 items, median 2 references each, no look-alike pair.

### Scored (development, not evidence)

There are 41 labelled pass-2 observations, 40 `unknown` and 0 without a label. With the map from observation labels only, 22 of the 28 history entries map to an item. Of the other 6, 4 have no labelled observation and 2 are ties.

| Condition (n) | current | generic | household |
| --- | --- | --- | --- |
| stays (21) | correct update 6, duplicate 1, wrong merge 1, kept not observed 13 | correct update 5, duplicate 2, unscorable 1, kept not observed 13 | as generic |
| moved (1) | dropped 1 | unresolved 1 | correct update 1 |
| new to inventory (13) | correct add 8, wrong merge 3, unscorable 2 | correct add 10, unresolved 1, unscorable 2 | correct add 9, unresolved 2, unscorable 2 |

The extra-view method equals household, and the variants without the alignment give the same table.

As a sensitivity check, the entries can be named by the audit's code rule instead: each code is formed from the id of the entry that mainly shows the object. That maps 25 of 28 entries and moves 3 items from "new to inventory" to "stays":

| Condition (n) | current | generic | household |
| --- | --- | --- | --- |
| stays (24) | correct update 8, duplicate 1, wrong merge 1, kept not observed 14 | correct update 6, duplicate 2, wrong merge 1, unscorable 1, kept not observed 14 | as generic |
| moved (1) | dropped 1 | unresolved 1 | correct update 1 |
| new to inventory (11) | correct add 8, wrong merge 3 | correct add 9, unresolved 1, wrong merge 1 | correct add 8, unresolved 2, wrong merge 1 |

The wrong merge that generic and household gain under this map is one bowl sighting. It updated a history entry whose two history views show two different objects. Under the label-only map that case is unscorable.

What the cases show:

- **The moved cup.** It is one detection, at cosine 0.52 to its entry's single reference.
  - The current matcher dropped it as a tentative entry and marked its entry missed.
  - Generic left it unresolved: 0.52 is under `SIM_MOVED` (0.60).
  - Household updated the right entry: the cup's impostor cosine in pass 1 was 0.42, so its own threshold is 0.47, and the margin to the next cup is above 0.10.
- **The two duplicates of generic and household.**
  - The laptop: its pass-2 sightings have cosines 0.13 to 0.36 to its references, under `SIM_PLACE`, at 0.14 m from its place. Rule (b) requires appearance to agree, and with this embedder it does not.
  - The book: the labels show it detected twice in the same keyframes. Rule (c) then splits it into two sightings, and one of them is added. The current matcher also made a duplicate of it.
- **Wrong merges of the current matcher.** The 4 wrong-merge items come from two label-only merges within its radius. An observation of another object, detected as a bottle, went to a bottle entry. Two objects detected as keyboards went to one new entry.
- **False "missed".** For items that stayed:
  - current: 1, the pass-1 cup that the operator says did not move;
  - generic and household: 3, that cup, the laptop and a suitcase, whose sightings appearance did not match.
  Both also called one bowl that stayed "moved": its sighting lay 0.35 m from its place after the correction, and its observations still went to it.
- **The one extra-view query** of the smoke run came after pass 1, not after the test scan, so no method used it. Its two captures came 15.7 s and 18.1 s after the request, and the whole query took 21.1 s (n = 1 query, 2 captures).

### Changes during development

All changes come after the parameters were frozen. No parameter value changed. Two rules changed after the first scored run:

1. **The yaw is kept only when its median residual is smaller than the translation's.** Before, it was kept whenever it was within the yaw limit. A synthetic test showed the old rule taking a worse yaw when one item moved 0.16 m, inside the inlier band. On the smoke data the yaw still wins for both methods, and every outcome is the same as in the first scored run.
2. **An extra view that matches nothing in place is evidence only.** Before, it could have been added as a new item. The smoke data have no extra view after the test scan, so nothing changed there.

The per-item beliefs and the count of same-keyframe double detections were added to the report after the first scored run. They are reporting only.

## Limits

- Two identical objects cannot be told apart by appearance. Only place, exclusivity and the "looked at" rule separate them, and when both are candidates the sighting is unresolved. The extra view resolves such a case only by showing one of them in its place.
- With one history scan there is no stability prior. Every place radius is 0.30 m and no item is "stable".
- An item with one stored embedding is matched on that one view. If the embedder does not recognise the object from another side, as with the laptop here, the generic and household rules add it again instead of merging it by label.
- Rule (c) assumes one detection per object and keyframe. The detector sometimes boxes one object twice in a frame, as with the book here, and the rule then forces a duplicate.
- "Looked at" uses saved depth images. Keyframes with a person in them have none, and there the script falls back to the phone's own check of that entry, made at the uncorrected position, or counts nothing. The phone logs a check only for entries it did not match in that keyframe.
- The phone's appearance check is computed against the entry's stored embedding at its uncorrected place. Using it makes the generic method depend on that check, though not on the matcher's entry assignment.
- Sightings are built inside the phone's view groups. An object whose observations fall into two groups gives two sightings. Both may update the same item, if they share no keyframe.
- Test scans are decided against the history only, never against each other.
- The scoring is only as good as the labels. History entries that mix two objects can be unscorable or scored differently depending on how they are mapped (both maps are shown above). Unlabelled observations are excluded, so a wrong merge or a false "moved" claim on them is not counted; the belief table still shows the claim.
- The smoke data have no sheet from the people who ran the test, no place codes, no visit record and no look-alike pair, so the conditions "not visited", "new same-model" and "live extra view" are tested only on synthetic scenes.
