# FindBack study spec (v0.4): what the paper claims and what the app and analysis must record

Status: spec, 2026-11-16. Written for the people implementing the app instrumentation and the analysis pipeline. The experimenter-facing steps are in `paper/EXPERIMENT_PROTOCOL.md`.

## 1. The paper in one paragraph

A phone that remembers where the user put things answers "where is my X?" from a stored observation. That observation is a fact about the past. Showing it as the item's current location is wrong whenever the item has moved, or the phone's position in its own map has drifted. Taking a fresh look on every query avoids that, but costs the user a capture each time, and visual matching makes its own mistakes. FindBack treats memory as a historical claim and decides per query, from signals available before any new capture, whether a fresh look is worth it.

Claims and the experiment that supports each:

| Claim | Experiment |
| --- | --- |
| C1. A map kept by the app and re-attached from one fix point gives room-correct guidance across rooms and restarts, without markers or a cloud service. | E1: marker error and room accuracy |
| C2. On-device matching separates "the item is there" from "nothing", "another object" and "a look-alike", across viewpoint and lighting. | E2: labelled verification benchmark |
| C3. A selective policy needs far fewer captures than always verifying, for a similar rate of wrong leads. Memory alone gives a wrong lead whenever memory is stale. | E3: controlled paired trials. E5: multi-day deployment with natural movement |
| C4. The system is cheap in latency, storage and energy. | E4 |

E3 uses scripted changes, so it tests the mechanism and the cost, not how often things really move. Only E5 can show that the signals predict staleness in real use. The paper must say this plainly.

## 2. Definitions used everywhere

**Ground truth** of a query, chosen by the experimenter at Done:

| Value | Meaning |
| --- | --- |
| `present` | The item is at the marker. The experimenter then taps its true centre, which gives the marker error. |
| `displaced` | The item is visible nearby but not at the marker. The experimenter taps its true centre, which gives the displacement. |
| `absent` | The item is not in this area. |
| `replaced` | Another object is at the marker. |
| `skip` | Not usable. |

**System answer** under a policy: `LAST_SEEN_HERE` (no capture), `HERE_NOW` (capture confirmed), `NOT_CONFIRMED` (capture did not confirm). The system never says "moved".

**Outcomes:**

- *Correct lead:* answer is `LAST_SEEN_HERE` or `HERE_NOW`, and the item is at the marker.
- *Wrong lead:* answer is `LAST_SEEN_HERE` or `HERE_NOW`, and the item is not at the marker. This is the error the paper targets.
- *Correct warning:* `NOT_CONFIRMED`, and the item is not at the marker.
- *Lost retrieval:* `NOT_CONFIRMED`, but the item is at the marker.

"At the marker" means ground truth `present` with a measured marker error of at most `R_PRESENT_M` (0.30 m, an analysis constant), or `present` with no measurement.

**Cost:** captures per query, inference calls per query, seconds per capture (from reaching the item to the result), and energy per capture.

**Pre-capture signals** (the only inputs a policy may use to decide on a capture): memory age, mobility class, number of earlier moves of this item, position source, metres walked since the last map fix, tracking losses since the fix, time since the fix, fix type. Similarity to other stored items is not a trigger: it is used after a capture, as a margin test (section 4).

## 3. App instrumentation (v0.4)

Everything here is additive. No existing event key changes name or meaning. The verification pipeline and the on-device `VerificationPolicy.decide()` stay as they are: policies are compared offline.

### S1. Study settings

Location sheet → "Study settings…". Stored in SharedPreferences.

- `studyTag`: free text, up to 60 characters, default empty. Convention: `key=value` pairs separated by `;`, for example `home=A;exp=E3;round=2;light=evening;split=dev`.
- `captureMode`: `FREE` (default, today's behaviour), `PAIRED`, `BENCHMARK`.

Log `study_settings` when either changes. Add `studyTag` and `captureMode` to `session_start`, `register`, `query_start`, `verify` and `query_end`.

### S2. PAIRED mode

One capture per query, so every query can be replayed under every policy.

- Verify is enabled until one capture has succeeded, then disabled for that query.
- Done without a capture: if the item can be resolved now, ask "Capture once before finishing?" with "Back" and "Finish without capture". Otherwise finish directly.
- `query_end.noCaptureReason`: null, `user_skipped` or `not_located`.

### S3. BENCHMARK mode

Any number of captures per query. After each verification result is dismissed, ask "What is at the marker right now?" with: The item, Nothing, A different object, A look-alike. The same dialog has a sticky "Condition" choice: default, near, far, angled, dim, bright, occluded.

Log `verify_label` with `query`, `record`, `attempt`, `label` (`target`, `empty`, `other`, `lookalike`, or null when dismissed) and `condition`.

### S4. Ground truth at Done

Replace the three-way prompt with the five values in section 2. `displaced` enters the same tap-to-measure step as `present`. `query_end.groundTruth` carries the value; `query_end.markerError` keeps its fields in both cases and gains `displaced` (true or false).

### S5. Embeddings and record history

- For every verification, write `verify/<vid>_emb.json`: model name, dimension, target record id, and for each candidate its kind, rectangle, similarity and embedding (base64, little-endian float32). Add `embFile` to the `verify` event.
- Append every record change to `records_log.jsonl`: `{t, session, op, record}` with `op` in `add`, `rename`, `delete`, `supersede`, and the full record including its embedding. `records.json` only holds active records, and offline analysis needs all of them.

### S6. Mobility class

The register dialog asks "How often does it move?": Rarely, Sometimes, Often. Default Sometimes. New nullable record field `mobility` (`rare`, `some`, `often`). Log it in `register` and `query_start`.

### S7. Update location

A user who finds an item somewhere else tells the app. Items → the item → "Update location".

- The app waits for a tap on the item where it is now, runs the normal capture, and shows the register dialog titled "Update location of X" with the name, room and mobility filled in.
- On save, the new record gets `lineageId` (the old record's `lineageId`, or its id), `supersedes` (the old id) and `moveCount` (old + 1). The old record leaves the active list. Its files and log entries stay.
- New ordinary records get `lineageId` equal to their own id and `moveCount` 0. Old files without these fields read as that.
- Log `supersede`: old id, new id, `lineageId`, `moveCount`, the old record's age in minutes, and the distance between old and new map positions when both are in the current map.
- `query_start` gains `lineageId`, `moveCount`, `lineageAgeMin` (since the lineage's first registration) and `mobility`.
- Cancelling returns to normal with nothing changed.

### S8. Timing

- `verify.msSinceQueryStart`, and `verify.msSinceNear` (null if NEAR was never reached).
- `query_end.msToFirstVerify`.

### Constraints

- Records stay immutable except through S7. A confirmed verification does not refresh `lastSeenMs`.
- Old `records.json` and `map.json` files must load.
- New dialogs must never block the GL thread or leave the app waiting if dismissed.
- Unit tests for the record fields, lineage logic, and the embedding file encoding.

## 4. Analysis

`analysis/evaluate.py` stays the stdlib-only summary and keeps printing its current sections first. Figures and paper numbers come from a new `analysis/paper_figures.py` (numpy, matplotlib, scikit-learn allowed).

### Inputs

- One or more data folders pulled from phones.
- Sessions are selected by `studyTag`. A session with no tag is excluded from paper output. `exp=` picks the experiment, `split=dev|test` the split, `home=` the site.

### Outputs

- `paper/figures/*.pdf`
- `paper/generated/results.tex`: one LaTeX macro per number quoted in the paper. Every macro is always defined. A number with no data yet expands to a red "TBD".
- `paper/generated/*.tex` tables.

**No synthetic number may reach the paper.** A synthetic generator is needed to test the code, but `paper_figures.py` must refuse to write under `paper/` when any input session is synthetic, unless given `--allow-synthetic`, in which case every figure is watermarked SYNTHETIC and every macro expands to a red "SYNTHETIC".

### Figures and tables

| Id | Content |
| --- | --- |
| T1 | Dataset: sites, rooms, items, registrations, queries by ground truth, captures, labelled benchmark captures |
| F1 (E1) | CDF of marker error by position source (`SESSION`, `MAP`) and by whether the item is in the fix point's room. Error against metres walked since the fix. Table of median and 90th percentile. Jumps on re-fix (`map_fix` with `kind` refresh or manual): median and maximum, by fix type. Share of queries where the item's room was named correctly. |
| F2 (E2) | ROC of benchmark captures, positives `target`, negatives split by `empty`, `other`, `lookalike`. AUC with bootstrap intervals. True-positive rate per condition at the threshold chosen on the dev split. Two decision rules: similarity threshold; threshold plus margin over the best-matching other record (from the embedding files and `records_log.jsonl`). |
| F3 (E3, E5) | Policy replay on paired queries: captures per query against wrong-lead rate, and against lost-retrieval rate. Policies: memory only; always verify; the logged on-device heuristic; random capture with probability p (the diagonal baseline); oracle (captures exactly the stale queries); a risk score from logistic regression on the pre-capture signals, trained on dev sessions or leave-one-session-out, swept over its threshold. Signal ablation for the learned policy. Bootstrap intervals clustered by session. |
| T2 (E4) | Latency of registration and verification (median, 95th percentile, by stage), storage per item, map file size, seconds per capture, power per mode and energy per verification |

### Other tools

- `analysis/make_trial_sheet.py`: from `records.json`, a round number and a seed, write a sheet that tells the experimenter what to do with each item (leave, remove, move nearby, replace with another object, swap with a look-alike), the query order, and whether to walk a long loop first. The change must be drawn independently of every pre-capture signal. Default mix: 50% leave, 20% remove, 15% move nearby, 15% replace or swap.
- `analysis/power_measure.py`: record the Pixel's power rails for a fixed time under a label and save the totals. `adb shell dumpsys powerstats` lists the rails on the study phone (camera, GPU, CPU clusters, TPU, display, memory); work out how to read their energy counters on a user build (Perfetto's `android.power` data source is one route). Rail energy is valid while charging over USB.

## 5. Experiments and target sizes

| Id | What | Target |
| --- | --- | --- |
| E0 | Walk test of v0.3 (two rooms, one restart) | Pass before anything else |
| E1 | Localization: all items measured after each of 3 restarts; 20 re-locks at a spot; picture and spot fixes | 2 sites × 12 items × 3 = 72 or more measurements |
| E2 | Verification benchmark in BENCHMARK mode | 20 items × about 12 labelled captures = 240 |
| E3 | Controlled paired trials in PAIRED mode, from trial sheets | 2 sites × 20 items × 4 rounds = 160 queries |
| E4 | Cost: latency from logs; 3 power runs of 5 minutes per mode | tracking only; tracking plus a verification every 20 s |
| E5 | Deployment: live normally, query every item twice a day, update location when found elsewhere | 2 sites × 20 items × 2 rounds × 5 days = 400 queries |
