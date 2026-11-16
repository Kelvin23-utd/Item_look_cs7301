package edu.cs6301.findback.study

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The strings and rules of the v0.4 study instrumentation that reach the logs. */
class StudyTest {
  @Test
  fun captureModeParsing() {
    assertEquals(CaptureMode.FREE, CaptureMode.parse(null))
    assertEquals(CaptureMode.FREE, CaptureMode.parse(""))
    assertEquals(CaptureMode.FREE, CaptureMode.parse("paired")) // stored names are upper case
    assertEquals(CaptureMode.PAIRED, CaptureMode.parse("PAIRED"))
    assertEquals(CaptureMode.BENCHMARK, CaptureMode.parse("BENCHMARK"))
    assertEquals(listOf("FREE", "PAIRED", "BENCHMARK"), CaptureMode.values().map { it.name })
  }

  @Test
  fun tagIsOneTrimmedLineOfAtMost60Characters() {
    assertEquals("", StudySettings.cleanTag(null))
    assertEquals("", StudySettings.cleanTag("   "))
    val example = "home=A;exp=E3;round=2;light=evening;split=dev"
    assertEquals(example, StudySettings.cleanTag("  $example \n"))
    assertEquals("a b", StudySettings.cleanTag("a\nb"))
    val long = "x".repeat(59) + " yz"
    assertEquals("x".repeat(59), StudySettings.cleanTag(long)) // cut at 60, then trimmed
    assertEquals(60, StudySettings.cleanTag("y".repeat(200)).length)
    // Format characters are kept as typed: the tag is data, never a format string.
    assertEquals("exp=%s;%d%%", StudySettings.cleanTag("exp=%s;%d%%"))
  }

  @Test
  fun statusLineIsAbsentForFreeWithoutATag() {
    assertNull(StudySettings.statusLine("", CaptureMode.FREE))
    assertEquals("Study: PAIRED", StudySettings.statusLine("", CaptureMode.PAIRED))
    assertEquals("Study: FREE · exp=E1", StudySettings.statusLine("exp=E1", CaptureMode.FREE))
    assertEquals("Study: BENCHMARK · exp=%s", StudySettings.statusLine("exp=%s", CaptureMode.BENCHMARK))
  }

  @Test
  fun groundTruthValues() {
    assertEquals(listOf("present", "displaced", "absent", "replaced", "skip"), GroundTruth.CHOICES.map { it.first })
    assertTrue(GroundTruth.measured("present"))
    assertTrue(GroundTruth.measured("displaced"))
    for (v in listOf("absent", "replaced", "skip")) assertFalse(v, GroundTruth.measured(v))
  }

  @Test
  fun mobilityValues() {
    assertEquals(listOf("rare", "some", "often"), Mobility.CHOICES.map { it.first })
    assertEquals(listOf("Rarely", "Sometimes", "Often"), Mobility.CHOICES.map { it.second })
    assertEquals("some", Mobility.DEFAULT)
    assertEquals("some", Mobility.orDefault(null)) // a record saved before v0.4
    assertEquals("some", Mobility.orDefault("weekly"))
    assertEquals("often", Mobility.orDefault("often"))
  }

  @Test
  fun verifyLabelValues() {
    assertEquals(listOf("target", "empty", "other", "lookalike"), VerifyLabel.CHOICES.map { it.first })
    assertEquals(listOf("default", "near", "far", "angled", "dim", "bright", "occluded"), VerifyLabel.CONDITIONS)
    assertEquals("default", VerifyLabel.DEFAULT_CONDITION)
  }

  @Test
  fun pairedAllowsExactlyOneCapture() {
    assertTrue(CaptureRules.verifyAllowed(CaptureMode.PAIRED, 0))
    assertFalse(CaptureRules.verifyAllowed(CaptureMode.PAIRED, 1))
    for (n in 0..5) {
      assertTrue(CaptureRules.verifyAllowed(CaptureMode.FREE, n))
      assertTrue(CaptureRules.verifyAllowed(CaptureMode.BENCHMARK, n))
    }
  }

  @Test
  fun capturePromptOnlyInPairedWithoutACaptureWhenResolvable() {
    assertTrue(CaptureRules.askCaptureFirst(CaptureMode.PAIRED, 0, resolvable = true))
    assertFalse(CaptureRules.askCaptureFirst(CaptureMode.PAIRED, 0, resolvable = false)) // finish directly
    assertFalse(CaptureRules.askCaptureFirst(CaptureMode.PAIRED, 1, resolvable = true))
    assertFalse(CaptureRules.askCaptureFirst(CaptureMode.FREE, 0, resolvable = true))
    assertFalse(CaptureRules.askCaptureFirst(CaptureMode.BENCHMARK, 0, resolvable = true))
  }

  @Test
  fun noCaptureReason() {
    assertNull(CaptureRules.noCaptureReason(1, resolvableAtDone = true))
    assertNull(CaptureRules.noCaptureReason(3, resolvableAtDone = false))
    assertEquals("user_skipped", CaptureRules.noCaptureReason(0, resolvableAtDone = true))
    assertEquals("not_located", CaptureRules.noCaptureReason(0, resolvableAtDone = false))
  }
}
