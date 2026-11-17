package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Plan section 4: when a keyframe is taken. */
class KeyframePacerTest {
  private fun at(x: Double, yawDeg: Double = 0.0) = ScanFixtures.pose(Vec3(x, 1.4, 0.0), yawDeg = yawDeg)

  @Test
  fun scanKeyframesNeedTimeAndMotion() {
    val p = KeyframePacer()
    p.scanStarted()
    // The first keyframe of a scan is taken at once.
    assertEquals(KeyframePacer.Mode.SCAN, p.onFrame(1_000, at(0.0), scanning = true, passiveAllowed = false, busy = false))
    p.taken(1_000, at(0.0))
    // Moved far enough, but not 400 ms later.
    assertNull(p.onFrame(1_300, at(0.2), true, false, false))
    // 400 ms later, but neither moved 10 cm nor turned 8 degrees.
    assertNull(p.onFrame(1_433, at(0.05, yawDeg = 3.0), true, false, false))
    // Moved 10 cm.
    assertEquals(KeyframePacer.Mode.SCAN, p.onFrame(1_466, at(0.10, yawDeg = 3.0), true, false, false))
    p.taken(1_466, at(0.10, yawDeg = 3.0))
    // Turning 8 degrees within 400 ms of the last keyframe is not enough yet: at 1.5 degrees per
    // 33 ms frame (45 degrees per second) the turn passes 8 degrees at 1_664 ms, but the keyframe
    // waits for the first frame at least 400 ms after the last one.
    var t = 1_466L
    var yaw = 3.0
    var got: KeyframePacer.Mode? = null
    while (got == null && t < 5_000) {
      t += 33
      yaw += 1.5
      got = p.onFrame(t, at(0.10, yawDeg = yaw), true, false, false)
    }
    assertEquals(KeyframePacer.Mode.SCAN, got)
    assertEquals(1_466L + 13 * 33, t) // 1_895: the first frame 400 ms on
    p.taken(t, at(0.10, yawDeg = yaw))
    // Holding still for longer than 400 ms gives nothing: in scan mode time alone is not enough.
    val still = yaw
    repeat(20) {
      t += 33
      assertNull(p.onFrame(t, at(0.10, yawDeg = still), true, false, false))
    }
    // Then a slow turn on the spot: the keyframe comes at the first frame 8 degrees or more from
    // the last one (9 degrees after six frames; 7.5 after five is not enough).
    var frames = 0
    got = null
    while (got == null && frames < 20) {
      frames++
      t += 33
      got = p.onFrame(t, at(0.10, yawDeg = still + 1.5 * frames), true, false, false)
    }
    assertEquals(KeyframePacer.Mode.SCAN, got)
    assertEquals(6, frames)
  }

  @Test
  fun noKeyframeWhileBusyOrTurningFast() {
    val p = KeyframePacer()
    p.scanStarted()
    assertNull(p.onFrame(1_000, at(0.0), true, false, busy = true))
    assertEquals(KeyframePacer.Mode.SCAN, p.onFrame(1_033, at(0.0), true, false, false))
    p.taken(1_033, at(0.0))
    // 3 degrees per 33 ms frame is 91 degrees per second: too fast, however far it has turned.
    var t = 1_033L
    var yaw = 0.0
    repeat(30) {
      t += 33
      yaw += 3.0
      assertNull(p.onFrame(t, at(0.0, yawDeg = yaw), true, false, false))
    }
    assertEquals(90.9, p.turnRateDegS, 0.1)
    // The turn stops: the next frame is slow, and the keyframe is taken.
    assertEquals(KeyframePacer.Mode.SCAN, p.onFrame(t + 33, at(0.0, yawDeg = yaw), true, false, false))
    // A gap between frames (tracking lost) is not a turn rate.
    p.onFrame(t + 5_000, at(0.0, yawDeg = yaw + 120), true, false, true)
    assertEquals(0.0, p.turnRateDegS, 0.0)
  }

  @Test
  fun passiveKeyframesAreTimedAndCanBeOff() {
    val p = KeyframePacer()
    assertNull(p.onFrame(1_000, at(0.0), scanning = false, passiveAllowed = false, busy = false))
    assertEquals(KeyframePacer.Mode.PASSIVE, p.onFrame(1_033, at(0.0), false, true, false))
    p.taken(1_033, at(0.0))
    assertNull(p.onFrame(2_900, at(1.0), false, true, false)) // moving does not matter, 2 s do
    assertEquals(KeyframePacer.Mode.PASSIVE, p.onFrame(3_033, at(1.0), false, true, false))
    // The capture failed (nothing in view): asked again after the retry delay, not every frame.
    p.failed(3_033)
    assertNull(p.onFrame(3_066, at(1.0), false, true, false))
    assertEquals(KeyframePacer.Mode.PASSIVE, p.onFrame(3_033 + ScanConfig.CAPTURE_RETRY_MS, at(1.0), false, true, false))
    // A scan that starts takes its first keyframe without waiting for motion.
    p.taken(3_300, at(1.0))
    p.scanStarted()
    assertNull(p.onFrame(3_500, at(1.0), true, true, false)) // still 400 ms after the last one
    assertEquals(KeyframePacer.Mode.SCAN, p.onFrame(3_700, at(1.0), true, true, false))
  }
}
