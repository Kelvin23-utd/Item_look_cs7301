package edu.cs6301.findback.scan

import edu.cs6301.findback.map.Vec3
import edu.cs6301.findback.map.distance
import edu.cs6301.findback.scan.ScanFixtures.embedding
import edu.cs6301.findback.scan.ScanFixtures.keyframe
import edu.cs6301.findback.scan.ScanFixtures.observe
import edu.cs6301.findback.scan.ScanFixtures.poseLookingAt
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/** Plan section 6: merging repeated sightings into inventory entries. */
class ScanSessionTest {
  private fun store() = InventoryStore(Files.createTempDirectory("inventory").toFile(), "sess0001")

  private var nextId = 0

  private fun session(
    store: InventoryStore,
    mode: ScanSession.Mode = ScanSession.Mode.SCAN,
    container: String? = null,
    id: String = "scan0001",
    mapId: String = "map00001",
  ) = ScanSession(id, mode, mapId, container, "effdet0", "sess0001", "mobilenet_v3_small", 1_000L, store) { "e%07d".format(nextId++) }

  private val cup = Vec3(0.4, 0.9, -1.1)

  @Test
  fun radiusFollowsTheSizeWithinItsLimits() {
    assertEquals(0.20, Merger.radius(0.05, 0.08), 1e-12) // 0.19 is raised to the minimum
    assertEquals(0.30, Merger.radius(0.30, 0.10), 1e-12) // 0.5 × the larger size + 0.15
    assertEquals(0.30, Merger.radius(0.10, 0.30), 1e-12)
    assertEquals(0.50, Merger.radius(0.90, 0.10), 1e-12) // capped
  }

  @Test
  fun oneObjectOverTenKeyframesGivesOneEntry() {
    val store = store()
    val s = session(store)
    val kinds = ArrayList<String>()
    for (i in 0 until 10) {
      // The user walks past: 60 cm sideways in all, and the depth is a few centimetres off each time.
      val pose = poseLookingAt(cup, back = 1.0 + 0.02 * (i % 3), side = -0.3 + 0.066 * i)
      val seen = Vec3(cup.x + 0.01 * (i % 4 - 1.5), cup.y, cup.z + 0.015 * (i % 3 - 1))
      val obs = observe(seen, pose, "cup", embedding(0))!!
      kinds += s.merge(keyframe(i, 1_000L + 500L * i, pose), listOf(obs)).results.single().kind
    }
    assertEquals(listOf(ScanSession.KIND_NEW, ScanSession.KIND_PROMOTED) + List(8) { ScanSession.KIND_KNOWN }, kinds)
    val e = store.snapshot().single()
    assertEquals(10, e.sightingCount)
    assertEquals(10, e.sightings.size)
    assertEquals("cup", e.name)
    assertEquals("cup", e.label)
    assertEquals(8.0, e.labelVotes.getValue("cup"), 1e-5)
    assertTrue("the entry is at the item, within the jitter", distance(e.position, cup) < 0.02)
    assertEquals("map00001", e.mapId)
    assertEquals("room1", e.roomId)
    assertNull(e.container)
    assertEquals(1_000L, e.firstSeenMs)
    assertEquals(5_500L, e.lastSeenMs)
    assertEquals("effdet0", e.detector)
    assertEquals("scan0001", e.createdScan)
    assertEquals(InventoryEntry.cropPath(e.id), e.cropFile)
    assertEquals(0, s.tentativeCount)
    assertEquals(1, s.itemCount)

    val sum = s.finish(6_000L)
    assertEquals(1, sum.newCount)
    assertEquals(0, sum.seenAgain)
    assertEquals(0, sum.tentativeDropped)
    assertEquals(ScanConfig.OUTCOME_SEEN, store.snapshot().single().evidence.lastOutcome)
    assertEquals(1, store.snapshot().single().evidence.seenCount)
  }

  @Test
  fun thePositionIsTheMeanOfTheLastTenSightings() {
    val store = store()
    val s = session(store)
    // Twelve sightings; the first two are 10 cm off and must have left the mean by the end.
    for (i in 0 until 12) {
      val p = if (i < 2) Vec3(cup.x + 0.10, cup.y, cup.z) else cup
      val pose = poseLookingAt(cup, side = 0.02 * i)
      s.merge(keyframe(i, 1_000L + 500L * i, pose), listOf(observe(p, pose, "cup", embedding(0))!!))
    }
    val e = store.snapshot().single()
    assertEquals(12, e.sightingCount)
    assertEquals(ScanConfig.POSITION_SIGHTINGS, e.sightings.size)
    assertTrue(distance(e.position, cup) < 0.002) // only the millimetre rounding of the depth is left
  }

  @Test
  fun twoSameLabelObjectsSixtyCentimetresApartGiveTwoEntries() {
    val store = store()
    val s = session(store)
    val left = Vec3(cup.x - 0.3, cup.y, cup.z)
    val right = Vec3(cup.x + 0.3, cup.y, cup.z)
    for (i in 0 until 4) {
      val pose = poseLookingAt(cup, back = 1.2, side = -0.1 + 0.05 * i)
      // The two look alike (similarity 0.95): only the distance tells them apart.
      val obs = listOf(observe(left, pose, "cup", embedding(0))!!, observe(right, pose, "cup", embedding(0, 0.95))!!)
      s.merge(keyframe(i, 1_000L + 500L * i, pose), obs)
    }
    val entries = store.snapshot()
    assertEquals(2, entries.size)
    assertEquals(listOf(4, 4), entries.map { it.sightingCount })
    assertTrue(entries.any { distance(it.position, left) < 0.01 })
    assertTrue(entries.any { distance(it.position, right) < 0.01 })
  }

  @Test
  fun aLabelFlipWithAMatchingEmbeddingStaysOneEntry() {
    val store = store()
    val s = session(store)
    val labels = listOf("cup", "cup", "bowl", "cup", "bowl", "bowl")
    for ((i, label) in labels.withIndex()) {
      val pose = poseLookingAt(cup, side = 0.05 * i)
      // The detector changes its mind, the item looks the same (0.8 to the first crop).
      val obs = observe(cup, pose, label, embedding(0, if (i == 0) 1.0 else 0.8), score = if (label == "cup") 0.8f else 0.5f)!!
      val r = s.merge(keyframe(i, 1_000L + 500L * i, pose), listOf(obs)).results.single()
      if (i >= 2) assertEquals(ScanSession.KIND_KNOWN, r.kind)
    }
    val e = store.snapshot().single()
    assertEquals(6, e.sightingCount)
    assertEquals("cup", e.label) // 2.4 against 1.5
    assertEquals("cup", e.name)
    assertEquals(2.4, e.labelVotes.getValue("cup"), 1e-5)
    assertEquals(1.5, e.labelVotes.getValue("bowl"), 1e-5)

    // Another label and another look at the same place is another thing: it does not merge.
    val pose = poseLookingAt(cup, side = 0.4)
    val other = observe(cup, pose, "remote", embedding(1))!! // similarity 0 to the cup
    assertEquals(ScanSession.KIND_NEW, s.merge(keyframe(6, 4_000L, pose), listOf(other)).results.single().kind)
    assertEquals(1, store.snapshot().size)
    assertEquals(1, s.tentativeCount)
  }

  @Test
  fun theLabelFollowsTheVotesAndTheNameFollowsTheLabelUntilRenamed() {
    val store = store()
    val s = session(store)
    val labels = listOf("bowl", "bowl", "cup", "cup", "cup")
    for ((i, label) in labels.withIndex()) {
      val pose = poseLookingAt(cup, side = 0.05 * i)
      s.merge(keyframe(i, 1_000L + 500L * i, pose), listOf(observe(cup, pose, label, embedding(0), score = 0.6f)!!))
      if (i == 3) {
        // 1.2 against 1.2: a tie keeps the current label.
        assertEquals("bowl", store.snapshot().single().label)
      }
    }
    assertEquals("cup", store.snapshot().single().label)
    assertEquals("cup", store.snapshot().single().name)
    // After a rename the label still follows the votes, the name does not.
    val id = store.snapshot().single().id
    store.rename(id, "grandma's bowl")
    for (i in 5 until 9) {
      val pose = poseLookingAt(cup, side = 0.05 * i)
      s.merge(keyframe(i, 1_000L + 500L * i, pose), listOf(observe(cup, pose, "vase", embedding(0), score = 0.9f)!!))
    }
    val e = store.snapshot().single()
    assertEquals("vase", e.label)
    assertEquals("grandma's bowl", e.name)
    assertTrue(e.nameEdited)
  }

  @Test
  fun tentativeEntriesNeedASecondLaterSightingAndAreDroppedOtherwise() {
    val store = store()
    val s = session(store)
    val pose = poseLookingAt(cup)
    assertEquals(ScanSession.KIND_NEW, s.merge(keyframe(0, 1_000L, pose), listOf(observe(cup, pose, "cup", embedding(0))!!)).results.single().kind)
    assertEquals(0, store.snapshot().size)
    assertEquals(1, s.tentativeCount)
    // A second sighting 0.2 s later is too soon (it could be the same exposure): still tentative.
    val soon = s.merge(keyframe(1, 1_200L, pose), listOf(observe(cup, pose, "cup", embedding(0))!!))
    assertEquals(ScanSession.KIND_TENTATIVE, soon.results.single().kind)
    assertTrue(soon.added.isEmpty())
    assertEquals(0, store.snapshot().size)
    // 0.3 s after the first sighting it becomes real, with all three sightings.
    val later = s.merge(keyframe(2, 1_300L, pose), listOf(observe(cup, pose, "cup", embedding(0))!!))
    assertEquals(ScanSession.KIND_PROMOTED, later.results.single().kind)
    assertEquals(3, later.added.single().sightingCount)
    assertEquals(setOf(later.added.single().id), later.matchedIds)
    assertEquals(1, store.snapshot().size)

    // Something seen in one keyframe only never becomes an entry.
    val ghost = Vec3(cup.x + 0.6, cup.y, cup.z)
    val wide = poseLookingAt(cup, back = 1.3)
    s.merge(keyframe(3, 2_000L, wide), listOf(observe(ghost, wide, "mouse", embedding(2))!!))
    assertEquals(1, s.tentativeCount)
    val sum = s.finish(3_000L)
    assertEquals(1, sum.tentativeDropped)
    assertEquals(1, sum.newCount)
    assertEquals(1, store.snapshot().size)
    assertEquals(0, s.tentativeCount)
  }

  @Test
  fun withinOneKeyframeAnEntryTakesOnlyOneDetection() {
    val store = store()
    val s = session(store)
    // Two cups 12 cm apart: nearer than the merge radius, but two boxes of one frame are two things.
    val a = Vec3(cup.x - 0.06, cup.y, cup.z)
    val b = Vec3(cup.x + 0.06, cup.y, cup.z)
    for (i in 0 until 3) {
      val pose = poseLookingAt(cup, side = 0.03 * i)
      val obs = listOf(observe(a, pose, "cup", embedding(0), sizeM = 0.08)!!, observe(b, pose, "cup", embedding(0), sizeM = 0.08, score = 0.7f)!!)
      val r = s.merge(keyframe(i, 1_000L + 500L * i, pose), obs)
      assertEquals(2, r.results.map { it.entryId }.toSet().size)
    }
    assertEquals(2, store.snapshot().size)
    assertTrue(store.snapshot().any { distance(it.position, a) < 0.01 })
    assertTrue(store.snapshot().any { distance(it.position, b) < 0.01 })
  }

  @Test
  fun theBestCropAndItsEmbeddingAreReplacedByABetterOne() {
    val store = store()
    val s = session(store)
    val far = poseLookingAt(cup, back = 2.0)
    val near = poseLookingAt(cup, back = 0.8)
    fun obs(pose: CameraPose, emb: FloatArray, crop: String, score: Float = 0.8f): Observation {
      val o = observe(cup, pose, "cup", emb, score = score)!!
      return Observation(o.label, o.score, o.box, o.mapPoint, o.worldPoint, o.depthM, o.distanceM, o.sizeM, o.embedding, o.cropWidthPx, o.cropHeightPx, crop)
    }
    s.merge(keyframe(0, 1_000L, far), listOf(obs(far, embedding(0), "crop-far-1")))
    val promoted = s.merge(keyframe(1, 1_500L, far), listOf(obs(far, embedding(0, 0.99), "crop-far-2", score = 0.6f)))
    // The first crop was the better of the two (same size, higher score): it is the one to save.
    assertEquals(listOf(promoted.added.single().id to "crop-far-1"), promoted.cropUpdates)
    val before = store.snapshot().single()
    assertEquals(2.0f, before.captureDistance, 1e-3f)

    // Closer: a larger box, so a better crop. Photo, embedding and capture geometry follow it.
    val closeEmb = embedding(0, 0.9)
    val better = s.merge(keyframe(2, 2_000L, near), listOf(obs(near, closeEmb, "crop-near")))
    assertEquals(listOf(before.id to "crop-near"), better.cropUpdates)
    val after = store.snapshot().single()
    assertSame(closeEmb, after.embedding)
    assertEquals(0.8f, after.captureDistance, 1e-3f)
    assertTrue(after.bestQuality > before.bestQuality)
    assertTrue(after.cropWidthPx > before.cropWidthPx)
    assertEquals(ScanFixtures.T_WM.inverse().transformPoint(near.position), after.viewPosition)

    // Farther again: a sighting, but the photo stays.
    val worse = s.merge(keyframe(3, 2_500L, far), listOf(obs(far, embedding(0), "crop-far-3")))
    assertTrue(worse.cropUpdates.isEmpty())
    assertSame(closeEmb, store.snapshot().single().embedding)
    assertEquals(4, store.snapshot().single().sightingCount)
  }

  @Test
  fun entriesOfAnotherMapOrAnotherTargetAreNotMergedInto() {
    val store = store()
    val pose = poseLookingAt(cup)
    // An entry at the same coordinates of another map, and one in a drawer at the same place.
    val base = Merger.newEntry("other001", observe(cup, pose, "cup", embedding(0))!!, keyframe(0, 500L, pose), "mapOTHER", null, "scanOLD", "effdet0", "sessOLD", "mobilenet_v3_small")
    store.add(base)
    store.add(base.copy(id = "drawer01", mapId = "map00001", container = "desk drawer"))
    val s = session(store)
    for (i in 0 until 2) s.merge(keyframe(i, 1_000L + 500L * i, pose), listOf(observe(cup, pose, "cup", embedding(0))!!))
    assertEquals(3, store.snapshot().size)
    assertEquals(1, store.get("other001")!!.sightingCount)
    assertEquals(1, store.get("drawer01")!!.sightingCount)

    // A scan of that drawer merges into the drawer's entry, and what it creates carries its name.
    val d = session(store, container = "Desk drawer ", id = "scan0002")
    val r = d.merge(keyframe(0, 5_000L, pose), listOf(observe(cup, pose, "cup", embedding(0))!!)).results.single()
    assertEquals(ScanSession.KIND_KNOWN, r.kind)
    assertEquals("drawer01", r.entryId)
    val pen = Vec3(cup.x + 0.6, cup.y, cup.z)
    val wide = poseLookingAt(cup, back = 1.3)
    for (i in 1 until 3) d.merge(keyframe(i, 5_000L + 500L * i, wide), listOf(observe(pen, wide, "scissors", embedding(3))!!))
    assertEquals("Desk drawer ", store.snapshot().single { it.label == "scissors" }.container)
  }

  @Test
  fun aPassiveSessionCreatesNothingAndChangesNoEntry() {
    val store = store()
    val s = session(store)
    val pose = poseLookingAt(cup)
    for (i in 0 until 2) s.merge(keyframe(i, 1_000L + 500L * i, pose), listOf(observe(cup, pose, "cup", embedding(0))!!))
    s.finish(2_000L)
    val before = store.snapshot().single()

    val p = session(store, mode = ScanSession.Mode.PASSIVE, id = "win00001")
    val other = Vec3(cup.x + 0.6, cup.y, cup.z)
    val wide = poseLookingAt(cup, back = 1.3)
    for (i in 0 until 3) {
      val r = p.merge(keyframe(i, 60_000L + 2_000L * i, wide), listOf(observe(cup, wide, "cup", embedding(0, 0.9), score = 0.99f)!!, observe(other, wide, "mouse", embedding(2))!!))
      assertEquals(listOf(ScanSession.KIND_KNOWN, ScanSession.KIND_UNMATCHED), r.results.map { it.kind })
      assertEquals(setOf(before.id), r.matchedIds)
      assertTrue(r.added.isEmpty())
      assertTrue(r.cropUpdates.isEmpty())
    }
    assertEquals(0, p.tentativeCount)
    // Nothing about the entry changed while the window ran...
    assertSame(before, store.snapshot().single())
    // ...and at its end only the evidence does: seen, at the time of the last matching keyframe.
    val sum = p.finish(70_000L)
    val after = store.snapshot().single()
    assertEquals(ScanConfig.OUTCOME_SEEN, sum.results.single().outcome)
    assertEquals(before.copy(evidence = after.evidence, lastSeenMs = after.lastSeenMs), after)
    assertEquals(64_000L, after.lastSeenMs)
    assertEquals(2, after.evidence.seenCount)
    assertEquals(ScanConfig.SOURCE_PASSIVE, after.evidence.lastSource)
    assertEquals("win00001", after.evidence.lastSourceId)
    assertEquals(2, after.sightingCount)
    assertFalse(sum.results.any { it.isNew })
    assertNotNull(store.lastScanId) // the scan before, not the window
    assertEquals("scan0001", store.lastScanId)
  }
}
