package edu.cs6301.findback.policy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VerificationPolicyTest {
  private fun inputs(
    source: PositionSource,
    meters: Double? = null,
    losses: Int? = null,
    age: Double = 1.0,
    sim: Float = 0.2f,
  ) = PolicyInputs(age, sim, source, false, meters, losses, 1000L, if (meters != null) "IMAGE" else null)

  @Test
  fun mapDriftTrigger() {
    assertFalse(VerificationPolicy.decide(inputs(PositionSource.MAP, 5.0, 0)).verify)
    assertFalse(VerificationPolicy.decide(inputs(PositionSource.MAP, VerificationPolicy.DRIFT_METERS, 0)).verify)
    val far = VerificationPolicy.decide(inputs(PositionSource.MAP, 25.0, 0))
    assertTrue(far.verify)
    assertEquals(1, far.reasons.size)
    assertTrue(VerificationPolicy.decide(inputs(PositionSource.MAP, 1.0, 1)).verify)
  }

  @Test
  fun driftDoesNotApplyToSessionAnchors() {
    assertFalse(VerificationPolicy.decide(inputs(PositionSource.SESSION, 50.0, 3)).verify)
  }

  @Test
  fun unchangedRules() {
    val none = VerificationPolicy.decide(inputs(PositionSource.NONE, 50.0, 3))
    assertFalse(none.verify)
    assertEquals(listOf("no spatial cue; historical photo only"), none.reasons)
    assertTrue(VerificationPolicy.decide(inputs(PositionSource.SESSION, age = 31.0)).verify)
    assertTrue(VerificationPolicy.decide(inputs(PositionSource.SESSION, sim = 0.81f)).verify)
    assertEquals(0.75f, VerificationPolicy.MATCH_THRESHOLD)
  }
}
