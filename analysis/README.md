# FindBack analysis

Two tools read the phone's logs. `evaluate.py` is the quick stdlib-only summary of everything in one data folder. `paper_figures.py` produces what goes into the paper, and only from sessions logged under a study tag.

```bash
# 1. Pull data from the phone (USB or wireless adb)
adb pull /sdcard/Android/data/edu.cs6301.findback/files/. findback_data/

# 2. Quick summary: three-condition comparison, threshold sweep, latency and marker error
python3 analysis/evaluate.py findback_data --exclude-session cab39b87 --exclude-session 1debdfe5

# 3. Pick the threshold on a development session and report on the rest
python3 analysis/evaluate.py findback_data --dev-session <dev_session_id> --csv results.csv

# 4. Visual check of crops and verification frames (red = projected window, green = best box)
python3 analysis/contact_sheet.py findback_data analysis/sheet.png

# 5. Figures, tables and numbers for the paper (one or more data folders)
python3 analysis/paper_figures.py findback_data backups/phone_files_2026-11-16 --power-dir analysis/power_runs

# 6. Trial sheet for one E3 round
python3 analysis/make_trial_sheet.py findback_data --round 1 --seed 20261119 --tag "home=A;exp=E3;round=1;split=dev"

# 7. One power run (E4); start the app and locate first
python3 analysis/power_measure.py --mode tracking_only --label run1 --seconds 300
python3 analysis/power_measure.py --mode tracking_verify --label run1 --seconds 300

# 8. Tests (about 30 s)
python3 analysis/test_analysis.py

# 9. Synthetic logs, for testing the code only (written to the system temp directory)
python3 analysis/synth_generate.py

# 10. v0.7 detailed trace: one report per traced session (Markdown and plots in analysis/out/)
python3 analysis/trace_summary.py findback_data
python3 analysis/trace_summary.py findback_data --session ab12cd34 --no-plots
python3 analysis/test_trace_summary.py      # its tests (a few seconds)
```

`trace_summary.py` needs numpy, pandas and (for the plots) matplotlib. With pytest installed, `python3 -m pytest analysis -q` runs both test files.

`paper_figures.py`, `synth_generate.py` and the tests need numpy, matplotlib, scipy and scikit-learn. `evaluate.py`, `make_trial_sheet.py` and `power_measure.py` use the standard library only.

## evaluate.py

From v0.3 the report ends with three more sections: marker error by position source (`SESSION`, `MAP`, and v0.2's `REFERENCE`), map-fix jumps while located (median and max, by fix type; the drift measurement), and queries by guidance mode at query start (NEAR reached, time to NEAR). Everything above them is unchanged. The `--csv` file gains matching columns at the end.

v0.4 ground truth. `displaced` and `replaced` are counted like `absent`: the item is not at the marker, so a claim is WRONG and a withheld claim is `unc|abs`. When either occurs, the "Queries with ground truth" line lists its count and one line under the table says so. The tap after `displaced` measures the item's displacement, so it is left out of the marker-error lines and reported on its own line (`displacement_m` is the last CSV column). On data without the new values the output is unchanged.

`evaluate.py` still treats every `present` as present. The study spec also requires a measured marker error of at most 0.30 m. When any `present` query exceeds that, a NOTE at the very end says how many; `paper_figures.py` applies the rule. `analysis/testdata/evaluate_baseline/` holds the output on the pilot data from before this change, and the tests check that today's output starts with it.

## paper_figures.py

Implements section 4 of `FindBack/docs/STUDY_SPEC_v0.4.md`. Output goes to `paper/` unless `--out` says otherwise.

| Output | Content |
| --- | --- |
| `figures/F1_marker_error_cdf.pdf`, `F1_error_vs_walked.pdf` | E1: CDF of marker error by position source and room; error against metres walked since the fix |
| `figures/F2_roc.pdf`, `F2_conditions.pdf` | E2: ROC by negative type with both rules' operating points; true-positive rate by condition |
| `figures/F3_policy_E3.pdf`, `F3_policy_E5.pdf` | Measured policy replay: wrong leads (top) and lost retrievals (bottom) against captures per query. Points: memory only, always verify, the logged on-device rule, oracle, and the frame-signals policy at the point chosen on dev. Curves: all signals, frame signals only (with its 95% band), age-only threshold, random capture |
| `figures/F4_what_if.pdf` | A model, not a measurement: captures per query needed against the AUC of a staleness predictor, for 10%, 30% and 50% of queries stale. Its measured inputs are the two confirmation shares of E3. Measured AUCs are marked on the AUC axis |
| `generated/results.tex` | One macro per number. Every macro is always defined |
| `generated/table_dataset.tex`, `table_cost.tex` | T1 and T2 as complete `table` floats (labels `tab:dataset`, `tab:cost`); `paper/findback.tex` inputs the first |
| `generated/table_*.tex` (others) | Tabular only: dataset by experiment and site, marker error, re-fix jumps, the two rules, policies and ablation (age only, world signals, frame signals, all) for E3 and E5, confirmation shares by ground-truth class for E3 (`table_confirm_E3.tex`), power by rail group |
| `generated/provenance.txt` | The input folders, which sessions counted, and which protocol each number used |

Every figure file is always written. Without data it shows its axes and a red "TBD"; with too little data it says "n too small". Figures are vector PDF, 3.33 in wide, with embedded TrueType fonts. Each policy has one colour and one marker in every figure (`POLICY_STYLE`); marker shape and line style carry the identity in greyscale. A swept curve that cannot be drawn keeps its legend entry, which says "n too small" or "not available"; `provenance.txt` gives the reason.

**Macros.** `results.tex` needs `xcolor`. A macro without data is a red TBD. A quantity that needs more data than there is says "n too small". Macros ending in `Rate` expand to `percent (k/n)`, so a rate never appears without its counts; other macros are bare numbers with the unit in the name. The last block defines the names `paper/findback.tex` uses (`\NSites`, `\WrongLeadMemoryPct`, ...). Those are bare numbers for use in a sentence, so a percentage from fewer than 20 cases expands to "n too small (k/n)" instead. Each macro's definition is in the comment on its line.

Bare-number macros for the selective-policy evaluation (spec section 4, "How the selective policy is evaluated"):

| Macro | What it is |
| --- | --- |
| `\OnDeviceWrongLeadPct`, `\OnDeviceCaptures`, `\OnDeviceCaptureSavingPct` | Measured, E3: the logged on-device rule |
| `\FramePolicyWrongLeadPct`, `\FramePolicyCaptures`, `\FramePolicyCaptureSavingPct`, `\FramePolicyLostPct` | Measured, E3 test split: the frame-signals policy at the operating point chosen on dev |
| `\FrameStaleAUC`, `\FrameStalePct` | Measured, E1 and E3 `present` queries: AUC of the frame signals for a marker error over 0.30 m, and how often that happens |
| `\ConfirmAtMarkerPct`, `\ConfirmStalePct` | Measured, E3: share of captures that confirm with the item at the marker (c_p) and not at it (c_s) |
| `\WhatIfCapturesAucFifty`, `...Seventy`, `...Eighty`, `...Ninety`, `...Hundred` | MODEL, not measured: captures per query needed at predictor AUC 0.5 to 1.0, for `\WhatIfBaseRatePct` percent of queries stale and a tolerance of `\WhatIfTolerancePoints` points |
| `\CrossRoomNearPct`, `\CrossRoomSecondsToNear` | Measured, E1 and E3: queries that start in another room than the item's |

**Which events count.** Only events logged while a study tag was in effect. The tag in effect is the event's own `studyTag`, or the last one seen earlier in the same session, so a tag changed half-way through a session (as the E2 protocol does) is followed event by event. `exp=` selects the experiment, `split=dev|test` the split, `home=` the site. The same session in two folders is counted once.

**Synthetic data.** A session is synthetic if any event has `"synthetic": true`, its id starts with `SYNTH`, its tag has a `synthetic` key, its folder holds `SYNTHETIC_DATA.txt`, or its folder's records are marked synthetic. A power run is synthetic if it says so. With any synthetic input, every macro and table cell expands to a red SYNTHETIC and every figure is watermarked, wherever the output goes. If the output directory is `paper/` (or any directory named `paper`), the run is refused and writes nothing unless `--allow-synthetic` is given. `--debug-json FILE` dumps the raw values for testing and refuses a path under `paper/`.

### Definitions and choices

| Topic | What the code does |
| --- | --- |
| At the marker | Ground truth `present` with `markerError.errorM` (the 3D error) at most 0.30 m, or `present` without a measurement. `displaced`, `absent` and `replaced` are never at the marker, even if a displacement is under 0.30 m. `skip` and unfinished queries are left out. "Stale" means not at the marker |
| Outcomes | Correct lead, wrong lead, correct warning, lost retrieval, as in the spec. Rates are per replayed query, so the four sum to 1. `...LostOfAtMarkerRate` is the lost retrievals among queries whose item is at the marker |
| Paired queries | Queries of `exp=E3` (or E5) with a usable ground truth, capture mode `PAIRED` (or not logged) and at least one capture. The first capture is used. Queries without a capture, and queries logged in FREE or BENCHMARK mode, are counted and left out |
| Confirmed | Best similarity of the capture at least τ. τ is `--threshold` if given; else the threshold chosen on the E2 dev split; else the on-device threshold logged with each capture. The same τ is used in F2 and F3 |
| τ on dev | Youden's J on a 0.01 grid over dev captures (target against all other labels). Ties go to the higher threshold. Needs 10 target and 10 other captures in dev |
| F2 evaluation set | The test split when both splits exist; otherwise all labelled captures, with the on-device threshold (nothing is tuned). `\VerifyAUC` alone uses all labelled captures of both splits |
| Margin rule | Confirm when rule 1 holds and (similarity of the best candidate to the target) minus (its highest similarity to any other record active at the capture time) is at least m. Other records exclude the target's lineage. m is chosen on dev at the fixed τ (Youden's J, grid 0 to 0.5, ties to the smaller m). With no other record the margin test passes. The two rules are compared on the captures that have an embedding file |
| Record history | `records_log.jsonl` replayed up to the capture time (`add`, `rename`, `delete`, `supersede` in either shape). Records in `records.json` that the log never mentions count as active from their `lastSeenMs`. Without a log: `records.json` with `lastSeenMs` up to the capture time, so deleted records are missing |
| Policies | Memory only; always verify; the logged on-device rule (`query_start.policyVerify`); oracle (captures exactly the stale queries); random capture with probability p, computed exactly as the straight line between memory only and always verify; three swept policies: the learned risk score on all signals, the learned risk score on the frame signals only, and a threshold on the memory age |
| On-device rule | The app's own decision, replayed as logged. In E3 a script decides what moves and every record is older than the rule's 30-minute limit, so the rule is expected to verify every query. It is reported as a labelled row, under `\OnDeviceWrongLeadPct`, `\OnDeviceCaptures` and `\OnDeviceCaptureSavingPct`. The older names `\WrongLeadSelectivePct`, `\CapturesSelective` and `\CaptureSavingPct` expand to the same three numbers |
| World and frame signals | World signals say the item may have moved: memory age, mobility class, earlier moves. Frame signals say the marker may be drawn in the wrong place: position source, metres since the fix, tracking losses, time since the fix, fix type. `WORLD_SIGNALS` and `FRAME_SIGNALS` in `fb_metrics.py` |
| Age-only policy | Captures the queries with the oldest memory first (`query_start.ageMin`). Nothing is fitted, so it needs no training data and uses no split. A query without an age is captured last |
| Frame-signals policy | The learned risk score restricted to the frame-signal columns, trained to predict "not at the marker" like the all-signals score, under the same train and test rule. It is the curve in F3, the ablation row, and the policy behind `\FramePolicy...` |
| Operating point chosen on dev | For `\FramePolicy...` and the hollow point in F3. The frame-signals model is fitted on the dev queries; on those same dev queries (in-sample, used for nothing else) the risk threshold with the fewest captures whose wrong-lead rate is within 5 percentage points of always verifying is taken (`WL_TOLERANCE`); that threshold is then applied unchanged to the test queries. Needs a dev and a test split; without one the macros stay TBD and the output says why. If on dev no capture at all is already within 5 points, the chosen policy never captures |
| Frame staleness (measured) | Among `present` queries of E1 and E3, the marker counts as wrong when `markerError.errorM` is over 0.30 m; a `present` query without a measurement counts as not wrong and is counted in the output. `\FrameStaleAUC` is the AUC of the frame-signals score for that label, out of sample by the train and test rule above, with a bootstrap interval over sessions. No capture is needed, so E1 queries count. Under the protocol's tags E1 is all `split=test`, so the model is trained on the `present` queries of the E3 dev rounds only |
| Verification behaviour (measured) | On all paired E3 queries (both splits; nothing is fitted on them): the share of first captures that confirm when the item is at the marker and when it is not, and for `absent`, `displaced`, `replaced`, and `present` with a marker error over 0.30 m. Counts and bootstrap intervals over sessions. `table_confirm_E3.tex` and `\EThreeConfirm...` |
| What-if model | Spec section 4, item 3, exactly: with base rate b, predictor operating point (FPR, TPR) and the two measured shares c_p and c_s, wrong-lead rate = b (1 − TPR (1 − c_s)), lost-retrieval rate = (1 − b) FPR (1 − c_p), captures per query = b TPR + (1 − b) FPR. The predictor's ROC is the equal-variance binormal curve with the given AUC: FPR = Φ(Φ⁻¹(TPR) − √2 Φ⁻¹(AUC)). Always verifying leaves b c_s wrong leads, so "within 5 points" needs TPR ≥ 1 − 0.05 / (b (1 − c_s)); the figure plots the captures per query at that TPR, and 0 where b (1 − c_s) ≤ 0.05 (memory alone is already within 5 points). AUC 0.5 gives the random-capture line and AUC 1 the oracle. Only c_s moves the curve; c_p enters the lost-retrieval rate, which the figure does not show. The curve is drawn only when both shares rest on at least 20 captures; otherwise the figure and `\WhatIfCapturesAuc...` say "n too small". Every number from it is a model output, and its macros say MODEL in their comments |
| Marks on the what-if axis | Every AUC that was measured: age only and all signals in E3 (and E5 if there is data, plus mobility class alone as a score), and the frame-signals AUC for frame staleness. The last one predicts a different label (marker error among unmoved items) than the axis describes |
| Learned policy | Logistic regression, L2 with C = 1 (not tuned), on nine columns from the eight pre-capture signals: log(1+age in minutes), mobility as 0/1/2, earlier moves, position source is MAP, log(1+metres since fix), tracking losses, log(1+minutes since fix), fix type as two indicators. Standardised with training statistics. Missing mobility counts as `some`, missing move count as 0, and drift proxies as 0 when not located |
| Train and test | With both `split=dev` and `split=test` in the experiment: fitted on dev, and every policy is replayed on test only. Otherwise leave-one-session-out: each session is scored by a model fitted on the others, and every policy is replayed on all queries. The output and `\EThreeProtocol` say which |
| Ties | Queries with equal risk are captured together; there is no random tie-break. A point between two thresholds is a linear interpolation (the expected rate of choosing between them at random) |
| Ablation | Four rows on the same queries and protocol: age only (threshold, not fitted), world signals only, frame signals only, all signals. Each with the AUC of its score for "not at the marker", its interval, and the area under its wrong-lead curve |
| Bootstrap | 2000 resamples of whole sessions, percentile 95% interval, fixed seed. The learned model is not refitted in a resample, and the operating point chosen on dev is not re-chosen: an interval describes the fixed policy on other sessions like these. `--e2-cluster item` resamples items instead of sessions for E2 |
| E1 groups | `present` queries with a measurement, by `markerError.positionSource`. For `MAP`, "the fix point's room" compares `query_start.itemRoom` with the room of the nearest labelled walk-graph node within 1.5 m of the fix point in `map.json` |
| Re-fix jumps | `map_fix` with `kind` `refresh` or `manual` that was applied, by fix type. Accepted disagreements are included and counted separately |
| Guidance across rooms | E1 and E3 queries with a usable ground truth whose `query_start.userRoom` and `itemRoom` are both known and differ. `\NavCrossRoomNearRate`: the share with `query_end.nearReached` true, over those that logged it. `\NavCrossRoomSecondsMedian`: median `timeToNearMs` (wall time from the start of the query to the first NEAR guidance) over those that reached it. Room-naming accuracy is not measured; `\EOneRoomNamedRate` only says how often a room was named |
| T2 | Latency over all tagged `register` and `verify` events. Seconds per capture is `verify.msSinceNear` plus `latency.totalMs` of a query's first capture: the app takes `msSinceNear` when Verify is pressed, and the result appears after the processing time. Power is the mean over runs of the summed monitored rails. Energy per verification is (energy of a verifying run minus tracking power times its duration) divided by the `verify` events tagged `exp=E4` inside the run. It is reported only with three runs per mode and every verifying run above every tracking-only run (an exact rank test, p = 1/20); otherwise the macro says "n too small" or "not resolved" |

Minimum sizes (constants at the top of `fb_metrics.py`): median 5, 90th percentile 10, 95th percentile 20, a CDF curve 10, a bar 5, a ROC curve or AUC 10 per class, the policy figure 20 queries, a learned score 20 training queries with 5 of each class in every fold, leave-one-session-out 3 sessions, a bootstrap interval 4 sessions, each of the two shares behind the what-if model 20 captures. A bare percentage in a sentence needs 20 cases (`MIN_N_BARE_RATE` in `paper_figures.py`).

### Event fields the pipeline reads

Required means the row is unusable without it. Everything else may be missing or null.

| Event or file | Required | Optional |
| --- | --- | --- |
| every event | `session`, `type` | `t`, `studyTag`, `captureMode`, `synthetic` |
| `study_settings` | | `studyTag`, `captureMode` |
| `register` | | `record`, `lineageId`, `room`, `recordBytes`, `latency.convertMs`, `latency.detectMs`, `latency.embedMs` |
| `query_start` | `query` | `record`, `lineageId`, `ageMin`, `mobility`, `moveCount`, `positionSource`, `metersSinceFix`, `trackingLossesSinceFix`, `msSinceFix`, `fixType`, `fixPoint`, `policyVerify`, `itemRoom`, `itemRoomName`, `userRoom`, `mapId` |
| `verify` | `query`; `bestSimilarity` or `scores[].sim` | `record`, `attempt`, `threshold`, `inferenceCalls`, `latency.convertMs`, `latency.detectMs`, `latency.embedMs`, `latency.totalMs`, `embFile`, `msSinceNear` |
| `verify_label` | `query`, `label` | `attempt` (else the latest capture of the query), `condition` |
| `query_end` | `query`, `groundTruth` | `markerError.errorM`, `markerError.displaced`, `markerError.positionSource`, `markerError.metersSinceFix`, `markerError.fixPoint`, `noCaptureReason`, `nearReached`, `timeToNearMs` |
| `map_fix` | | `kind`, `fixType`, `applied`, `jumpM`, `jumpDeg`, `disagreement`, `accepted` |
| `records.json` | `id` | `embedding`, `lastSeenMs`, `lineageId`, `synthetic` |
| `records_log.jsonl` | `op`, `record.id` | `t`, `record.embedding`, `record.lineageId`, `record.supersedes` |
| `verify/<vid>_emb.json` | `candidates[].embedding` | `dim`, `record`, `candidates[].sim`. Aliases: `dimension`, `embeddingDim`; `target`, `targetRecord`, `recordId`; `scores`; `similarity`; `emb`, `vector`. An embedding is base64 little-endian float32 or a JSON list |
| `map.json` | | `mapId`, `fixPoints[].id/x/z`, `nodes[].x/z/roomId` |

Values are case-insensitive for `groundTruth`, `label`, `condition`, `mobility`, `captureMode` and the `exp` and `split` tag values. `verify_label.label` is one of `target`, `empty`, `other`, `lookalike`; anything else counts as unlabelled.

## make_trial_sheet.py

Writes `analysis/trial_sheets/trial_sheet_round<R>_seed<S>.md` and `.csv`: what to do with each item, the query order, and which queries follow a long loop. Default mix: 50% leave, 20% remove, 15% move nearby, 15% replace or swap (half each), rounded by largest remainder. One query in five follows a long loop (`--loop-share`).

The draw uses the record ids, the round and the seed, and nothing else from a record, so it cannot depend on any pre-capture signal. Actions, query order, long loops and the replace-or-swap choice come from separate random streams. The same records, round and seed give the same sheet. The CSV holds the scripted ground truth per item, for checking the logged answers afterwards.

## power_measure.py

On the Pixel 8 user build, `dumpsys powerstats` lists the rails but prints no energy, and the power-stats HAL refuses a dump from the shell. Perfetto's `android.power` data source reads the cumulative energy counter of each rail without root. The script records one trace for the requested time, pulls it, deletes it on the phone, and saves per-rail energy and mean power, sums by group (camera, GPU, CPU, TPU, display, memory), the phone's battery and screen state and foreground app before and after, and the raw trace. Rail energy is valid while charging over USB.

- The sum of the monitored rails is not the phone's total power: unmonitored supplies are missing, and a monitored LDO fed from a monitored buck is counted twice. Compare modes on the same rails.
- Runs go to `analysis/power_runs/`. `paper_figures.py --power-dir analysis/power_runs` uses a run for T2 only if its mode is `tracking_only` or `tracking_verify` and the FindBack app was the foreground app before and after the recording. A phone with the screen off reports no foreground app, so such a run is left out.
- Tag the E4 session `exp=E4`. The verifications in a run are counted from the `verify` events with that tag inside the run's time window; `--verifications N` is used only when the log has none.
- Keep the screen brightness the same in every run.

## Tests

`python3 analysis/test_analysis.py` (add `-k word` to select tests). The centre is a hand-built log of eleven paired queries in which the count of every outcome class under every policy is worked out by hand in the comments, then checked on the arrays and through the whole pipeline. Other tests cover the 0.30 m rule, the oracle and random baselines (the latter against a simulation), the threshold sweep, AUC and threshold choice, the clustered bootstrap (whole clusters, width, coverage), that the learned policy never scores a query it was trained on, the operating point chosen on dev (by hand and by brute force), the what-if formulas against hand-computed values and at both ends (AUC 0.5 is the random-capture line of the replay, AUC 1 its oracle), the frame-staleness AUC (pair by pair), that a frame-only score cannot see the world signals, the age-only policy (capture by capture), the confirmation shares, guidance across rooms, record history, the margin rule, E1 grouping, cost and energy, the power-trace parser, every way of marking synthetic data, the refusal to write under `paper/`, the trial sheet's mix and independence, and `evaluate.py` against its saved output on the pilot data.

## trace_summary.py

Reads the v0.7 detailed trace (`FindBack/docs/PLAN_v0.7_trace.md`): the new events in `events.jsonl` and the CSV files in `trace/<session>/`. For each session with a trace folder it prints a Markdown report and writes it to `--out` (default `analysis/out/`, which git ignores: the trajectory plot is a floor plan of the home) as `trace_summary_<session>.md`, with `trace_<session>_trajectory.png` (top view coloured by tracking state and by usable depth share) and `trace_<session>_timeline.png` (speed and usable depth over time). A session without a trace (setting off, or an older app) is left out unless named with `--session`; it then gets the sections that come from events only. Nothing in the data folder is changed.

| Section | From | What |
| --- | --- | --- |
| Session | `session_start`, `ar_config`, `ui_screen`, `ui_action`, `ui_search`, `app_pause`/`app_resume` | Duration and paused time, app version and commit, ARCore version, camera configuration, time per screen (paused time left out), button and control counts, dialog answers, Items searches (length and empty results only) |
| Messages | `ui_message` | Every banner text with its count, most frequent first, and the screens it appeared on |
| Rejected captures and locks | `capture_rejected`, `register`, `hit_rejected`, `lock_attempt` | Refusals by stage and reason (median of the value, such as the depth), refused registration taps per saved item, Lock success rate by kind, presses per successful lock, failure reasons, first-press success |
| Tracking | `pose.csv`, `tracking_lost` | Losses (tracking row followed by a non-tracking row), lost time and share of the traced time, lost time by failure reason |
| Motion | `pose.csv` | Metres walked (horizontal, tracking only, steps over 0.5 m left out), speed percentiles, time still (< 0.05 m/s), slow, walking (>= 0.3 m/s), turn rate |
| Depth | `depth.csv` | Share of samples without a depth image, mean usable share (0.2–4 m) and valid share by motion class and by turn rate (< 10, 10–30, 30–60, >= 60 deg/s) |
| Scene | `depth.csv` | Feature points per frame, tracked planes and plane area percentiles |
| Light | `sensors.csv` (`light`), `depth.csv`, `verify` | Lux percentiles, ARCore pixel intensity, each verification's similarity against the last lux sample (at most 60 s old), rank correlation with 5 or more |
| Performance | `perf`, `device_state`, `trace_error`, ML events | fps median and 5th percentile over the 10 s windows, frame time mean and p95, trace time on the GL thread, ML latencies (`register`, `verify`, `scan_keyframe`, `model_loaded`), battery drain per hour over stretches not charging, current, battery temperature, maximum thermal status, trace lines dropped, and each trace file's rows, size and rows per traced second against its nominal rate |
| Scan | `scan_*` events | Keyframes, detections kept and dropped by reason, depth self-check error |
| Findings | all of the above | Flags from the constants at the top of the script |

Findings thresholds (constants at the top of `trace_summary.py`): tracking lost more than 5% of the traced time; usable depth while walking below 50%; no depth image in more than 20% of the depth samples; fps 5th percentile below 20; median window's frame time p95 above 50 ms; thermal status moderate (2) or above; any trace line dropped; any trace error; more than 3 refused registration taps per saved item (or more than 3 with none saved); fewer than half of the Lock presses succeeding; scan depth check median above 10 cm. Each is a strict bound: a value exactly at it is not flagged.

Time is taken from the pose rows (10 Hz while the app is resumed): the interval to the next row, with gaps over 0.5 s counted as paused. A file that ends without a line break ends with a line the app did not finish (it was killed): that line is dropped.

`fb_load.py`, `evaluate.py` and `paper_figures.py` read only the event types they know, so the new events change none of their results; `test_trace_summary.py` checks this on the hand-built E3 log with every new event added.

`test_trace_summary.py`: each section on small traces built in a temporary folder (time per screen with a pause, message order, refusal and lock counts, tracking losses and lost time by reason, metres walked with a tracking jump left out, depth by motion and turn rate, light against verifications, battery drain, thermal status, dropped lines, the scan section), the thresholds at and just past their bounds, a session without a trace, a cut-off last line, the whole command with its files and plots, and that `fb_load` and `evaluate.py` give the same result with the new events in the log.

## synth_generate.py

Writes two fake data folders and fake power runs to the system temp directory (or `--out`). It refuses to write under `paper/`, `findback_data/`, `backups/` or `FindBack/`, or over a folder that holds an `events.jsonl` it did not write. Its output exercises every code path: all five ground truths, the three capture modes, benchmark labels and conditions, embedding files, record history, map fixes, both splits and an experiment without splits, two sites. `--variant minimal` writes v0.3-shaped events with only a study tag. Nothing it writes is a measurement.

## Offline baselines (bench_*.py)

Tracks A and B of `BASELINES_PLAN.md`: every method sees the same frames and crops the phone saved, on this Mac, offline. The design (pair sets, methods, operating points, metrics) is fixed in `notes/2026-11-17_baselines_design.md`, with a list of every change made after results were seen; methods, citations, weight versions and licences are in `notes/2026-11-17_methods.md`. Nothing is uploaded: the scripts put the Hugging Face libraries in offline mode and keep Ultralytics' usage analytics off through a private settings folder.

One-time setup (a dedicated environment and the weights, about 2 GB in `analysis/.bench-cache/`; both folders are git-ignored):

```bash
uv venv --python 3.12.12 --python-preference only-managed analysis/.venv-bench
uv pip install --python analysis/.venv-bench/bin/python -r analysis/requirements-bench.txt
analysis/.venv-bench/bin/python analysis/bench_fetch_weights.py     # downloads once, checks sha256
```

On a pull (one or more `--data` folders; developer test sessions of the session log are left out unless `--no-default-exclude`):

```bash
PY=analysis/.venv-bench/bin/python
D=backups/phone_files_2026-11-17_0008          # the new pull
# Track A: appearance matching, about 4 minutes. Table to stdout, analysis/out/bench_match/, and --md
$PY analysis/bench_match.py --data $D --md analysis/notes/<date>_track_a_results.md
# Track B: detectors on the scan keyframes, about 3 minutes. Label-free numbers and the judging sheet
$PY analysis/bench_detect.py --data $D --md analysis/notes/<date>_track_b_label_free.md
# Track B: after a person has filled analysis/out/judge/judge_sheet.html and pressed "Download judgments"
python3 analysis/bench_judge_read.py ~/Downloads/judgments_<sheet>.json --md analysis/notes/<date>_track_b_judged.md
# Tests (no weights or data needed, under a second)
$PY analysis/test_bench.py
```

| Script | What it does |
| --- | --- |
| `bench_common.py` | Loads a pull (events, records, inventory, scan keyframes), cuts crops exactly as the app does (sensor rectangle, clamp, truncate, rotate upright), builds the four pair sets (verify, verify with impostors, scan, cross; with and without possible duplicates), weighted ROC AUC, EER and TAR at 5% FAR, and the item bootstrap |
| `bench_match.py` | Track A. Checks first that the offline app embedder reproduces the logged similarities, then scores every pair with every method (best over the phone's candidate rectangles for a fresh capture). `--methods` picks a subset |
| `bench_detect.py` | Track B. Runs the detectors on the saved half-size keyframes, applies the app's keyframe filter to each, reports label-free numbers and EfficientDet-Lite0's agreement with the phone's own detections, writes `analysis/out/judge/judge_sheet.html` (detectors hidden, names shuffled) and `judge_key.json`. `--detectors` picks a subset; a detector that fails to load is dropped and named in the report |
| `bench_judge_read.py` | Reads the downloaded judgments with the key: item precision, naming precision (strict and lenient) and pooled relative recall per detector, over complete frames only. Standard library only |
| `bench_vocab_household.txt` | The fixed vocabulary of the open-vocabulary detectors (items and a stop section) |
| `bench_fetch_weights.py` | Downloads and checks the weights |
| `test_bench.py` | Crop geometry against the app's rules, pair construction on a hand-built pull (labels, impostors, duplicates, excluded sessions, tentative and unsaved detections), every metric against brute force or a hand-worked case, the item bootstrap, the keyframe filter rule by rule, the depth median, box mapping, vocabulary parsing, that the judging sheet hides detectors, and the judgment reader by hand |

The judging sheet embeds keyframes, so it and every other output stay in `analysis/out/` (git-ignored); `pair_scores.csv` names crops by key only. Only number tables go to `analysis/notes/`.

## VLM naming (vlm_*.py)

`VLM_NAMING_PLAN.md`: open vision-language models on the lab workstation name the stored items, judged against the names the user typed and by two judging models, with full profiling. The design (models, prompt, inputs, metrics, judging, contingencies) is fixed in `notes/2026-11-17_vlm_naming_design.md`; results in `notes/2026-11-17_vlm_naming_results.md`, every later change in `notes/2026-11-17_vlm_naming_changes.md`.

**Model names stay private.** The repository names the models only by aliases: contestants `model_a` to `model_d`, judges `judge_a` (primary) and `judge_b`. Their repositories, pinned revisions, download sizes and serving limits are in `analysis/vlm_models.private.json` (git-ignored; template `vlm_models.example.json`), which every script reads at run time and the sync commands copy into the private folder on the GPU machine. Tests use a fake one.

The workstation is reached by SSH. Its alias and the private folder there come from two environment variables and are never written to a file; the folder is one directory inside the account's own folder on the data disk, mode 700, laid out as `data/` (copied images), `code/`, `models/` (all weights), `env/` (Python environment with vLLM), `runs/<date>_<name>/` and `scratch/` (home, caches, temporary files). Only the images naming needs are copied (`analysis/out/vlm/package/`, about 4.5 MB: stored crops, 320-px copies, one marked keyframe per scan entry, the crops of other sightings), plus the pooled candidate names for the judges. Telemetry of vLLM and Hugging Face is off; weights come from the publishers' repositories at pinned revisions.

```bash
export FB_WORKSTATION=<ssh alias>                 # never committed
export FB_WS_DIR='/data/$USER/fb_vlm'             # single quotes: $USER expands on the workstation
PY=analysis/.venv-bench/bin/python
D=backups/phone_files_2026-11-17_0008

$PY analysis/vlm_prepare.py --data $D             # images and item lists in analysis/out/vlm (seconds)
$PY analysis/vlm_baselines.py                     # phone, MobileCLIP2 and Track B names (needs bench_detect output)
python3 analysis/vlm_ws.py setup                  # once: folder, environment with vLLM 0.33.0 (about 10 GB, 2 min)
python3 analysis/vlm_ws.py status                 # who is on, load, GPU owners, the queue's STATUS
for m in model_a model_b model_d model_c judge_a; do
  python3 analysis/vlm_ws.py fetch $m             # pinned weights (refuses to pass the 80 GB disk budget)
done
python3 analysis/vlm_ws.py queue-start            # detached queue on the workstation: waits for a GPU nobody else
                                                  #   uses (10 free minutes), runs every naming session, builds the
                                                  #   judging pool, runs both judges, yields to other users, ends
                                                  #   by itself when done or after 24 h
python3 analysis/vlm_ws.py queue-start --models model_b model_c model_d --until 2026-11-23T23:59
                                                  # the same queue for these contestants only (additional runs:
                                                  #   no judge, no judging pool), giving up at that workstation
                                                  #   time instead of after 24 h (changes note, item 20)
python3 analysis/vlm_ws.py queue-status           # the six-line runs/STATUS and the log tail
FB_WORKSTATION=<ssh alias> analysis/vlm_monitor.sh   # read-only view: GPU owners, our processes, STATUS
python3 analysis/vlm_ws.py queue-stop             # stop early (the running session finishes its item first)
python3 analysis/vlm_ws.py roundtrip model_a      # with the Mac connected: round trip through an SSH tunnel
python3 analysis/vlm_ws.py pull                   # mirror the workstation's runs/ to analysis/out/vlm/ws_runs
$PY analysis/vlm_report.py report --data $D --md analysis/notes/2026-11-17_vlm_naming_results.md \
    --changes-file analysis/notes/2026-11-17_vlm_naming_changes.md
python3 analysis/vlm_ws.py procs                  # must list nothing of ours (except a queue still waiting)
python3 analysis/vlm_ws.py cleanup --yes          # removes the whole folder on the workstation
python3 analysis/test_vlm.py                      # tests (standard library parts; the bench environment also
$PY analysis/test_vlm.py                          #   runs the numpy/Pillow parts), a few seconds
```

**On a university Slurm cluster** (`vlm_cluster.py`; the same session and judging code, run as batch jobs that do not depend on the Mac). Host, folder and Slurm options come from the environment only; pulled text has the folder, user, account and node names replaced by placeholders. Job names, batch scripts, output files and run folders use the aliases, the server is started from a neutral weights path (`models/<alias>`) with its arguments in a file, so nothing other users of the cluster can list (queue, job details, process command lines) names a model.

```bash
export FB_CLUSTER=<ssh host of the login nodes>
export FB_CLUSTER_DIR='<work storage>/$USER/fb_vlm'      # private folder (mode 700); $USER expands there
export FB_SLURM_OPTS='-A <account> -q <qos> -p <gpu partition> --gres=gpu:<gpu type>:1'
export FB_SLURM_CPU_OPTS='-A <account> -q <qos> -p <cpu partition>'   # install and downloads (need internet)
export FB_CLUSTER_PYTHON='<path of a full Python 3.12 there (with ctypes)>'
export FB_CLUSTER_PRE='<commands run first in every job: a CUDA toolkit on PATH, a host compiler that matches
                       that Python's libstdc++>'           # vLLM JIT-builds a FlashInfer module at start-up

python3 analysis/vlm_cluster.py check             # start estimate of a GPU job, free storage
python3 analysis/vlm_cluster.py setup             # folder, env.sh, vLLM 0.33.0 environment (CPU job, uv)
python3 analysis/vlm_cluster.py pack-env          # archive of the environment; every job unpacks it on node-local
                                                  #   storage (importing from the network file system took minutes)
python3 analysis/vlm_cluster.py fetch             # pinned weights, one CPU job per model (130 GB budget)
python3 analysis/vlm_cluster.py submit --one-gpu  # one GPU job per contestant session; both judges wait for the
                                                  #   contestants (Slurm dependencies), build the blind pool from
                                                  #   the primary runs and judge it; --one-gpu runs Judge B on
                                                  #   one large GPU (recorded)
python3 analysis/vlm_cluster.py smoke model_a     # optional software test on a synthetic image (FB_SLURM_SMOKE_OPTS)
python3 analysis/vlm_cluster.py jobs              # our jobs and the state of each run folder
python3 analysis/vlm_cluster.py wait              # poll the queue every 3 minutes until our jobs are gone
python3 analysis/vlm_cluster.py pull              # mirror runs/ to analysis/out/vlm/cluster_runs (sanitised)
python3 analysis/vlm_cluster.py cancel            # cancel our queued or running jobs
python3 analysis/vlm_cluster.py cleanup --yes     # delete the whole folder on the cluster
```

Which run counts: a model's *primary run* is its first complete run in time at any site; naming quality, the label-free signals and the judging pool use primary runs only. A later run of a model (for example Model A on the cluster) gives cost on its own hardware and a name-agreement check against the primary run. Every cost table has a site and GPU column and never combines sites; the round trip from the Mac exists for the workstation only.

| Script | What it does |
| --- | --- |
| `vlm_common.py` | The aliases and the reader of the private model file, prompts and answer schemas (quoted in the design note), request bodies, a streaming HTTP client that times a request, the vLLM metrics reader, name normalisation, the human-reference match rule, same-name rule, the blind judge pool, Wilson intervals, exact McNemar, Cohen's kappa, energy integration. Standard library only; runs on every machine |
| `vlm_prepare.py` | Item sets (tapped records, scan entries, other sightings) and the image package; the human reference and phone labels stay in `analysis/out/vlm/local/` on the Mac |
| `vlm_baselines.py` | Phone detector label (offline EfficientDet-Lite0 for tapped crops), MobileCLIP2-S2 zero-shot over the household vocabulary, Track B detector names matched to the scan entries |
| `vlm_ws.py` | The workstation over SSH through one shared connection: setup, sync, fetch, queue start/status/stop, round trip through `ssh -L`, single sessions by hand, pull (a mirror of `runs/`, with the workstation's paths and account replaced by placeholders), disk use, process check, cleanup |
| `vlm_cluster.py` | The cluster over SSH: setup and downloads as CPU jobs, one GPU batch job per session with the judges chained by Slurm dependencies, job status, pull (sanitised), cancel, cleanup |
| `vlm_remote.py` | Runs on the GPU machine: the workstation's queue (lock, 24 h limit or a give-up time given with `--until`, `--models` for additional contestant runs without judges, `runs/STATUS`, waits by NVML every 5 minutes, disk budget, contingencies, judging pool); the Slurm job driver (`job`: one session on the allocated GPU with the listed contingencies; judges build the pool from the primary runs first); and each session: idle-GPU check, a watchdog that yields to another account's GPU process after the item in flight, GPU power/energy/memory sampler (NVML, 100 ms), vLLM server bound to 127.0.0.1 and always stopped in a `finally` block, sequential, cross-view, concurrency and ping phases, judging |
| `vlm_report.py` | `pool`: blind judging pool from the primary runs. `report`: every table of the results note (primary and additional runs, one cost row per site) and `analysis/out/vlm/quality_cost.png` |
| `vlm_monitor.sh` | Read-only view of the workstation for a person: GPU owners, our processes, the queue's STATUS, newest log |
| `test_vlm.py` | The aliases and former keys, primary and additional runs across sites, the judge run that judged the primary pool, the cluster's pool equal to the Mac's, sanitising of cluster text, the queue's STATUS format, its model filter (only the listed contestants, in the queue's order, no judge and no pool, Model A skipped by its former-key run) and give-up time, the yield check and the choice of the latest finished run, singulars and normalisation, head words, the match and same-name rules, answer and verdict parsing, the pool's independence of method order and its blindness, request bodies, the metrics reader, every statistic against hand-worked values, energy integration, the report's counts on a hand-built run, Track B box matching, frame marking geometry, path sanitising |

## Session log

| Session | Date | What | Use in study |
| --- | --- | --- | --- |
| f1c1bc02, 774f5516, a0de418f | Nov 13 (phone clock still said Jun 15) | App launch only, before Wi-Fi | Exclude |
| ff5fbb39 | Nov 13 | First pilot by Xiao: 8 registrations, 3 queries, all present | Smoke test only |
| cab39b87, 1debdfe5 | Nov 13 | Claude's remote UI tests (phone lying still; one bogus 15.9 m depth) | Exclude |
| 848a45bc, b524e9b7, 182fad2b, 46e6c45e | Nov 13–16 | App launch only | Exclude |
| 0919ba37 | Nov 13 | Second pilot by Xiao: 8 registrations, 5 queries, 4 verifications. No room reference, so positions were session-only | Smoke test only |
| 2b883ce9, 5a4b83e9, 51e9efc9 | Nov 16 | Claude's v0.3 device checks with the phone still (home spot, room, restart, ghost, re-fix). The test map was deleted from the phone afterwards | Exclude |
| 0115c406 | Nov 16 | Xiao's first walk on v0.3 (one room, home spot, 3 items, 2 queries). Route guidance went from the route arrow to the ring; marker error 3.9 cm from a same-session anchor | Smoke test only |
| 007cc164 | Nov 16 | Claude's v0.4 checks with the phone still (settings, item dialog, ground-truth list; one query ended with Skip) | Exclude |
| 7a7718f6, c5b9a6ae, 735b0f23 | Nov 16 | Claude's v0.5 scan checks with the phone still, on a temporary map that was deleted afterwards (scan start and stop, detector load, summary) | Exclude |
| 1925c109, acbda081 | Nov 16 | Claude's v0.6 screen checks with the phone still, on the user's map: one untagged search for "my phone" opened and left without an answer (no `query_end`), two refused taps (depth 43 m) | Exclude |
| f52b04d2 | Nov 16 | The user's first scan walk on v0.6 (one Locate, one 69 s scan with 24 entries, five untagged searches). Summary in `analysis/notes/2026-11-16_first_scan_walk.md` | Untagged: not in the study numbers; use for pilot observations |
| c8990ea7 | Nov 17 | Claude's v0.7 trace check with the phone locked: 5 s, no user actions | Exclude |

None of these sessions has a study tag, so `paper_figures.py` uses none of them.

## First findings (ff5fbb39 + checks)

- Different objects score a median cosine similarity of 0.13 against each other (max 0.43, 28 pairs). The same object seen again scores 0.50, 0.81 and 0.81 (the last from an identical static view). The 0.75 threshold is too strict. Choose it on a dev session that includes **absent** trials, where you query the spot after moving the object away.
- The ML Kit "smallest box containing the tap" was often a scene-sized box. v0.2 rejects boxes over 25% of the frame or wider than 0.6 m and falls back to a 0.22 m window around the tap.
- A still phone can return depth hits 15–20 m away for an object about 1.5 m away. v0.2 rejects hits beyond 5 m.
- Latency on the Pixel 8 (median): registration ≈ 60 ms convert + 70 ms detect + 21 ms embed. Verification is ≈ 230 ms total with 3–5 embedder calls.
