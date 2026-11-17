# FindBack v0.7: detailed trace

Status: plan, 2026-11-16, git branch `scan-once`, on top of v0.6 (commit `a5cbdcb`).
The user's request: "log as much data as possible when using the app, to dig out more challenges
and information."

## 1. Goal

Until now the app logs events only (a registration, a query, a verification, a scan keyframe).
Nothing continuous is recorded, so after a session we cannot say how the phone moved, how good the
depth was, how bright the room was, how warm the phone got, or which error messages the user hit.
v0.7 records these, so that challenges can be found from normal use and backed with numbers.

Nothing in the app's behaviour changes. Every existing event keeps its name and fields. Study mode
is untouched.

## 2. Two layers

- **Events**: new low-rate event types in the existing `events.jsonl` (section 3).
- **Trace**: high-rate CSV files in a new per-session folder `trace/<sessionId>/` (section 4).

A setting, Menu (⋮) → Settings → "Detailed trace" (default **on**), turns both layers on or off.
With it off, the app registers no sensor listener, writes no trace file and logs none of the new
events except `session_start`'s new fields. The value is logged as `traceOn` in `session_start`.

No new permission is needed and no image or audio is recorded by this version.

## 3. New events

| Event | When | Fields |
| --- | --- | --- |
| `session_start` (existing, fields added) | as now | `appVersion`, `appCommit` (git short hash from `BuildConfig`, "unknown" if not available), `arcoreVersion` (installed ARCore package version), `cameraConfig` (CPU image size, texture size, fps range), `depthMode`, `displayPx`, `traceOn` |
| `app_pause`, `app_resume` | activity lifecycle | `sessionMs`; on pause also `traceBytes`, `traceDropped` |
| `ui_screen` | `UiModel.screen` changes | `from`, `to`, `msInPrevious` |
| `ui_action` | every button press or choice: bottom-bar actions, pill, menu entries, Items open/close, room chip, item sheet buttons, each dialog answer (including "Did you find it?" and Keep looking) | `action`, `screen`, optional `detail` (for example the room filter, the dialog's answer) |
| `ui_search` | Items search text settles for 1 s or the panel closes | `chars`, `results`, `total`, `roomFilter` (the typed text itself is not logged) |
| `ui_message` | every banner message | `text`, `screen` |
| `capture_rejected` | a registration tap, a Verify press or a measuring tap is refused before any capture | `stage` (`register`, `verify`, `measure`), `reason` (`no_hit`, `depth_far`, `image_not_ready`, `busy`, `model_loading`, `behind`, `not_in_view`, `not_located`, `tracking`), plus the value that caused it when there is one. The existing `hit_rejected` stays as it is. |
| `lock_attempt` | every Lock press ends | `spot`, `kind` (`add`, `use`), `ok`, `reason` on failure, `samples`, `posSpreadM`, `yawSpreadDeg`, `durationMs`, `attemptInBar` (1 for the first press since the bar opened) |
| `device_state` | at session start, every 30 s, and on pause | `batteryPct`, `batteryTempC`, `charging`, `thermalStatus` (`PowerManager.currentThermalStatus`), `availMemMb`, `lowMemory`, `freeStorageMb` |
| `perf` | every 10 s while resumed | `frames`, `fps`, `frameMsMean`, `frameMsP95`, `frameMsMax` (time inside the GL frame callback), `mlBusyShare`, `traceQueue`, `traceDropped` |
| `trace_error` | a trace file or sensor cannot be opened | `site`, `error` |

`ui_action` and `ui_message` are logged from `MainView` and the dialog classes at the single points
where actions are dispatched and messages shown, so no call site is missed.

## 4. Trace files

Folder `trace/<sessionId>/`. Plain CSV with a header row, `.` decimal point, `Locale.US`, one row per
sample. Every file's first column is `t` (epoch ms, `System.currentTimeMillis()`); the second is
`ns` (`SystemClock.elapsedRealtimeNanos()`, or the sensor event's own timestamp for sensors), so
files can be aligned with each other and with `events.jsonl`.

### 4.1 `pose.csv`, 10 Hz (GL thread samples, written elsewhere)

`t, ns, tracking, failure, wx, wy, wz, qx, qy, qz, qw, located, mx, my, mz, myaw, metersSinceFix,
speed, turnRate, screen, room, query`

- `tracking`: ARCore tracking state; `failure`: tracking failure reason or empty.
- `wx..qw`: `camera.pose` in the ARCore world frame.
- `mx..myaw`: the camera in the map frame when located, else empty.
- `speed` (m/s) and `turnRate` (deg/s): from the previous sample.
- `screen`: the current `Screen`; `room`: current room id; `query`: active query id or empty.

### 4.2 `sensors.csv`

`t, ns, type, x, y, z, accuracy`

| `type` | Android sensor | Rate |
| --- | --- | --- |
| `acc` | `TYPE_ACCELEROMETER` | 50 Hz |
| `gyro` | `TYPE_GYROSCOPE` | 50 Hz |
| `mag` | `TYPE_MAGNETIC_FIELD` | 10 Hz |
| `baro` | `TYPE_PRESSURE` (x = hPa) | 1 Hz |
| `light` | `TYPE_LIGHT` (x = lux) | on change, at most 5 Hz |

Listeners are registered on a background `HandlerThread` in `onResume` and removed in `onPause`. A
missing sensor is logged once as `trace_error` and skipped. The app's logic does not use these
values; they are recorded only.

### 4.3 `depth.csv`, 2 Hz (GL thread)

`t, ns, ok, w, h, validShare, usableShare, median, p10, p90, centre, points, planes, planeArea,
lightIntensity, speed, turnRate`

- `ok`: 1 when the 16-bit depth image was acquired; the remaining depth columns are empty when 0.
- `validShare`: share of pixels with depth > 0; `usableShare`: share within 0.2–4.0 m (the scan
  range); `median`, `p10`, `p90` over valid pixels in metres; `centre`: median of the central 5×5.
  Compute on a subsample (every 4th pixel in both directions) to keep the cost low.
- `points`: number of ARCore feature points in the frame's point cloud; `planes`: tracked planes;
  `planeArea`: sum of `extentX × extentZ` in m².
- `lightIntensity`: ARCore light estimate pixel intensity when the estimate is valid. If light
  estimation is disabled in the session config, enable `AMBIENT_INTENSITY` only if that does not
  change anything the app renders; otherwise leave the column empty and say so in the report.
- Close every acquired image at once. During a scan the keyframe path keeps priority: skip a depth
  sample rather than delay a keyframe.

### 4.4 `guidance.csv`, 5 Hz while a query is active (GL thread)

`t, ns, query, mode, resolvable, distance, bearingDeg, elevationDeg, routeM, straightM, inView,
ringX, ringY`

Distance and angles from the camera to the item's remembered position; `inView` and the ring's
screen position as the overlay uses them.

### 4.5 Writer

One `TraceWriter` owns all files: a single background thread, a bounded queue (8192 lines) fed by
`offer` (never blocks; when full the line is dropped and counted), `BufferedWriter`s flushed every
2 s and on pause. The GL thread and sensor thread only format a line and offer it. Formatting must
not allocate heavily at 100 Hz: reuse a `StringBuilder` per thread.

Size: about 30–40 MB per hour, mostly `sensors.csv`. Nothing is deleted automatically. Settings
shows "Trace data on this phone: N MB".

## 5. Analysis

New `analysis/trace_summary.py <data_dir> [--session ID] [--out DIR]` (numpy and pandas are
available; matplotlib for plots). For each session with a trace it prints, and writes as Markdown:

- **Session**: duration, app version, time per screen, counts of each `ui_action`.
- **Messages**: every `ui_message` text with its count, most frequent first. These are the
  failures the user met.
- **Rejected captures and locks**: `capture_rejected` by stage and reason; lock success rate and
  presses per lock.
- **Tracking**: number of losses, total lost time and share of the session, by reason.
- **Motion**: metres walked, speed percentiles, share of time still (< 0.05 m/s), slow, walking.
- **Depth**: usable share overall and split by motion (still, slow, walking) and by turn rate; the
  share of time with no depth image.
- **Scene**: feature points and plane area percentiles.
- **Light**: lux percentiles; where the session has verifications, similarity against lux.
- **Performance**: fps median and 5th percentile, frame time p95, ML latencies from existing events,
  battery drain per hour (when not charging), maximum thermal status, trace lines dropped.
- **Scan** (when present): keyframes, detections kept and dropped by reason, depth-check error.
- **Findings**: a short list of flags from fixed thresholds written as named constants at the top
  of the script, for example tracking lost more than 5% of the time, usable depth below 50% while
  walking, fps 5th percentile below 20, thermal status at or above moderate, more than 3 refused
  captures per saved item.

Plots (PNG next to the Markdown, default `analysis/out/`, which must be git-ignored because a
trajectory is a floor plan of the user's home): top-view trajectory coloured by tracking state and
by usable depth share; speed and usable depth over time.

`fb_load.py` and the existing analysis must ignore the new events without warnings.

## 6. Tests and acceptance

1. Kotlin unit tests for the pure parts: CSV line formatting (locale, empty fields), the rate
   limiters, depth statistics on a synthetic array, the bounded queue's drop counting, speed and
   turn rate from two poses.
2. Python tests on small traces built inside the tests (no real data, nothing synthetic written
   outside the test's temp folder): each summary section, the thresholds, a session without a
   trace, and that `fb_load` reads a log containing the new events.
3. `./gradlew testDebugUnitTest assembleDebug` passes (154 tests before this work);
   `python3 -m pytest analysis` passes.
4. With "Detailed trace" off: no `trace/` folder for the session, no sensor listener, none of the
   new events except the added `session_start` fields.
5. On the phone (checked by the auditor): the files grow at the stated rates, `perf` shows the GL
   frame time is not worse than before by more than 1 ms on average, and `traceDropped` stays 0.
6. `FindBack/README.md` documents the events, files, columns and the summary script.
