package edu.cs6301.findback.study

/**
 * Study instrumentation vocabulary (docs/STUDY_SPEC_v0.4.md, section 3). Pure Kotlin, no Android
 * imports: the strings below are what reaches events.jsonl and records.json, so they are fixed
 * here and unit-tested.
 */

/** S1: how many captures a query may take. */
enum class CaptureMode {
  /** Any number of captures; no extra prompts. The behaviour before v0.4. */
  FREE,
  /** Exactly one capture per query, so every query can be replayed under every policy (S2). */
  PAIRED,
  /** Any number of captures, each labelled by the experimenter (S3). */
  BENCHMARK;

  companion object {
    /** Stored preference to mode; anything unknown is [FREE]. */
    fun parse(s: String?): CaptureMode = values().firstOrNull { it.name == s } ?: FREE
  }
}

object StudySettings {
  /** S1: longest study tag kept. */
  const val TAG_MAX = 60

  /**
   * The tag as stored and logged: one line, trimmed, at most [TAG_MAX] characters. The
   * `key=value;key=value` form is a convention for the analysis and is not enforced here.
   */
  fun cleanTag(raw: String?): String {
    if (raw == null) return ""
    val oneLine = raw.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').trim()
    return if (oneLine.length <= TAG_MAX) oneLine else oneLine.substring(0, TAG_MAX).trim()
  }

  /** Second line of the status panel, or null when nothing is set (FREE and no tag). */
  fun statusLine(tag: String, mode: CaptureMode): String? =
    when {
      mode == CaptureMode.FREE && tag.isEmpty() -> null
      tag.isEmpty() -> "Study: " + mode.name
      else -> "Study: " + mode.name + " · " + tag
    }
}

/** S4: ground truth of a query, chosen by the experimenter at Done (spec section 2). */
object GroundTruth {
  /** The item is at the marker. */
  const val PRESENT = "present"
  /** The item is visible nearby but not at the marker. */
  const val DISPLACED = "displaced"
  /** The item is not in this area. */
  const val ABSENT = "absent"
  /** Another object is at the marker. */
  const val REPLACED = "replaced"
  /** Not usable. */
  const val SKIP = "skip"

  /** Value and the dialog's wording, in the order shown. */
  val CHOICES: List<Pair<String, String>> =
    listOf(
      PRESENT to "At the marker",
      DISPLACED to "Nearby, but not at the marker",
      ABSENT to "Not in this area",
      REPLACED to "Another object is at the marker",
      SKIP to "Skip (not usable)",
    )

  /** These two continue with the tap-to-measure step (marker error or displacement). */
  fun measured(truth: String): Boolean = truth == PRESENT || truth == DISPLACED
}

/** S6: how often the item moves, asked at registration. */
object Mobility {
  const val RARE = "rare"
  const val SOME = "some"
  const val OFTEN = "often"
  const val DEFAULT = SOME

  /** Value and the dialog's wording, in the order shown. */
  val CHOICES: List<Pair<String, String>> = listOf(RARE to "Rarely", SOME to "Sometimes", OFTEN to "Often")

  /** A stored value if it is one of the three, else the default (records saved before v0.4). */
  fun orDefault(stored: String?): String = if (CHOICES.any { it.first == stored }) stored!! else DEFAULT
}

/** S3: the experimenter's label of one BENCHMARK capture. */
object VerifyLabel {
  const val TARGET = "target"
  const val EMPTY = "empty"
  const val OTHER = "other"
  const val LOOKALIKE = "lookalike"

  /** Value and the dialog's wording, in the order shown. */
  val CHOICES: List<Pair<String, String>> =
    listOf(TARGET to "The item", EMPTY to "Nothing", OTHER to "A different object", LOOKALIKE to "A look-alike")

  const val DEFAULT_CONDITION = "default"

  /** Capture condition, sticky across captures. */
  val CONDITIONS: List<String> = listOf(DEFAULT_CONDITION, "near", "far", "angled", "dim", "bright", "occluded")
}

/** S2: capture gating and the reason a query ended without a capture. */
object CaptureRules {
  const val USER_SKIPPED = "user_skipped"
  const val NOT_LOCATED = "not_located"

  /** PAIRED allows Verify until one capture has succeeded; the other modes never limit it. */
  fun verifyAllowed(mode: CaptureMode, captures: Int): Boolean = mode != CaptureMode.PAIRED || captures == 0

  /** PAIRED, Done without a capture, and a capture is possible: ask before finishing. */
  fun askCaptureFirst(mode: CaptureMode, captures: Int, resolvable: Boolean): Boolean =
    mode == CaptureMode.PAIRED && captures == 0 && resolvable

  /**
   * `query_end.noCaptureReason`: null when a capture succeeded, [USER_SKIPPED] when the item
   * could be resolved at Done (a capture was possible), else [NOT_LOCATED].
   */
  fun noCaptureReason(captures: Int, resolvableAtDone: Boolean): String? =
    when {
      captures > 0 -> null
      resolvableAtDone -> USER_SKIPPED
      else -> NOT_LOCATED
    }
}
