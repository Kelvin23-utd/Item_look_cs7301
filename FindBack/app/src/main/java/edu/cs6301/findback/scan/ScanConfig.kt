package edu.cs6301.findback.scan

/**
 * Every tunable number of scan mode (docs/PLAN_v0.5_scan_once.md), in one place. They are the
 * plan's starting values: tune them here after field data. Section numbers refer to the plan.
 */
object ScanConfig {
  // ------------------------------------------------------------------ detections (section 3)

  /** A detection is kept only with at least this score. */
  const val MIN_SCORE = 0.35f
  /** At most this many kept detections per keyframe (the best by score). */
  const val MAX_PER_KEYFRAME = 10
  /**
   * The detector itself runs at this lower threshold. Detections between it and [MIN_SCORE] are
   * only logged (reason `low_score`), except `person`: a person at this score already keeps the
   * frame off the disk and drops the detections on that person.
   */
  const val RAW_SCORE_MIN = 0.20f
  /** Most raw detections taken from the detector per keyframe. */
  const val RAW_MAX_RESULTS = 25
  /** `effdet0` sees the upright image at this fraction of the sensor size (it resizes to 320 px). */
  const val DETECTOR_INPUT_SCALE = 0.5
  /** Never stored, cropped to disk or logged with an image. */
  const val PERSON_LABEL = "person"
  /** Furniture-scale classes that are not items to find. */
  val STOP_LABELS: Set<String> =
    setOf("couch", "bed", "dining table", "toilet", "tv", "refrigerator", "oven", "sink", "chair", "bench", "potted plant")
  /** Detections physically larger than this (longer box side, metres) are dropped. */
  const val MAX_OBJECT_M = 0.6
  /** A detection with more than this share of its box inside a person box is dropped (`on_person`). */
  const val ON_PERSON_OVERLAP = 0.5f
  /** Within one keyframe, same label and box overlap (IoU) over this is one detection. */
  const val DUPLICATE_IOU = 0.6f
  /** ML Kit detector: the image labeler's confidence threshold, and the score of an unnamed box. */
  const val MLKIT_LABEL_MIN = 0.35f
  /** ML Kit detector: the name of a box the labeler could not name. */
  const val MLKIT_UNNAMED = "object"

  // ------------------------------------------------------------------ keyframes (section 4)

  /** No two keyframes closer in time than this. */
  const val KEYFRAME_MIN_INTERVAL_MS = 400L
  /** Scan mode: next keyframe once the camera has moved this far... */
  const val KEYFRAME_MOVE_M = 0.10
  /** ...or turned this much since the last one. */
  const val KEYFRAME_TURN_DEG = 8.0
  /** Passive mode: one keyframe every this long. */
  const val PASSIVE_INTERVAL_MS = 2000L
  /** No keyframe while the camera turns faster than this (motion blur, rolling shutter). */
  const val MAX_TURN_RATE_DEG_S = 60.0
  /** After a failed capture (no depth, no image, nothing in view) try again after this. */
  const val CAPTURE_RETRY_MS = 250L
  /** Scan hint changes to "waiting for depth" when no keyframe was taken for this long. */
  const val NO_KEYFRAME_HINT_MS = 3000L
  /** `scan_end.metersWalked`: a camera step longer than this between two frames is a tracking jump. */
  const val WALK_JUMP_M = 0.5

  // ------------------------------------------------------------------ depth and position (section 5)

  /** Depth of a detection: median over this central fraction of the box (each side). */
  const val BOX_CENTRAL_FRACTION = 0.40f
  /** Detections with a depth outside this range are dropped. */
  const val DEPTH_MIN_M = 0.2
  const val DEPTH_MAX_M = 4.0
  /** Crops are the box grown by this fraction on every side, as tap registration does. */
  const val CROP_PAD = 0.08f
  /** The depth self-check runs on this many keyframes at the start of every scan. */
  const val SELF_CHECK_KEYFRAMES = 20
  /** Self-check: second test point, as a fraction of the image width and height. */
  const val SELF_CHECK_OFF_CENTRE_U = 0.30f
  const val SELF_CHECK_OFF_CENTRE_V = 0.30f

  // ------------------------------------------------------------------ merging (section 6)

  /** Merge radius = clamp(FACTOR × max(sizes) + OFFSET, MIN, MAX) metres. */
  const val MERGE_RADIUS_FACTOR = 0.5
  const val MERGE_RADIUS_OFFSET_M = 0.15
  const val MERGE_RADIUS_MIN_M = 0.20
  const val MERGE_RADIUS_MAX_M = 0.50
  /** A detection with another label still merges when the embeddings are at least this similar. */
  const val MERGE_SIMILARITY = 0.55f
  /** An entry's position is the mean of this many latest sightings; that many are stored. */
  const val POSITION_SIGHTINGS = 10
  /** A tentative entry becomes real at a second sighting from a keyframe at least this much later. */
  const val TENTATIVE_CONFIRM_MS = 300L

  // ------------------------------------------------------------------ evidence (section 7)

  /** An entry is in view inside the image less this margin on every side... */
  const val VIEW_MARGIN = 0.08
  /** ...and this far in front of the camera (along the optical axis, metres). */
  const val VIEW_MIN_M = 0.3
  const val VIEW_MAX_M = 3.0
  /** Depth at the entry's pixel: median of a patch of this many depth pixels a side. */
  const val DEPTH_PATCH = 5
  /** Hidden when the measured depth is less than z − max(HIDDEN_MIN_M, HIDDEN_FRACTION × z). */
  const val HIDDEN_MIN_M = 0.15
  const val HIDDEN_FRACTION = 0.15
  /** Appearance match: expected-window similarity at or above this counts as seen. To calibrate. */
  const val SEEN_SIMILARITY = 0.60f
  /** Appearance checks per keyframe, least recently checked entries first. */
  const val APPEARANCE_CHECKS_PER_KEYFRAME = 4
  /** `missed` needs this many keyframes in which the place was observable and looked at. */
  const val MISSED_MIN_KEYFRAMES = 3
  /** Passive evidence is decided per window of this length. */
  const val PASSIVE_WINDOW_MS = 60_000L
  /** A new entry may be a moved one: same label and at least this similar to a `missed` entry. */
  const val MOVED_SIMILARITY = 0.70f

  // ------------------------------------------------------------------ saved frames (section 9)

  /** Saved keyframes are JPEGs of this width (sensor orientation). */
  const val SAVED_FRAME_WIDTH = 960
  const val SAVED_FRAME_JPEG_QUALITY = 85
  /** Inventory crops (the entry's photo) are saved at this JPEG quality. */
  const val CROP_JPEG_QUALITY = 90
  /** The overlay drops a keyframe's boxes after this long without a newer keyframe. */
  const val OVERLAY_HOLD_MS = 1500L
  /** inventory.json is rewritten at most this often while a scan changes it. */
  const val STORE_FLUSH_MS = 5000L

  // ------------------------------------------------------------------ vocabulary (logs and files)

  const val DETECTOR_EFFDET0 = "effdet0"
  const val DETECTOR_EFFDET2 = "effdet2"
  const val DETECTOR_MLKIT = "mlkit"
  const val DEFAULT_DETECTOR = DETECTOR_EFFDET0
  val DETECTORS: List<String> = listOf(DETECTOR_EFFDET0, DETECTOR_EFFDET2, DETECTOR_MLKIT)

  fun detectorOrDefault(id: String?): String = if (id != null && id in DETECTORS) id else DEFAULT_DETECTOR

  const val OUTCOME_SEEN = "seen"
  const val OUTCOME_MISSED = "missed"
  const val OUTCOME_NOT_OBSERVED = "not_observed"

  const val SOURCE_SCAN = "scan"
  const val SOURCE_PASSIVE = "passive"
  const val SOURCE_VERIFY = "verify"

  /** `scan_start.target` of a scan that is not of a container. */
  const val TARGET_OPEN = "open"
}
