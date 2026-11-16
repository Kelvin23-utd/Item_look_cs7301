package edu.cs6301.findback.map

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.roundToInt

class WalkGraphTest {
  private var t = 0L

  /** Walk in a straight line from [a] to [b], feeding a position every [step] metres. */
  private fun WalkGraph.walk(a: Vec3, b: Vec3, step: Double = 0.1): List<WalkGraph.Visit> {
    val d = horizontalDistance(a, b)
    val n = maxOf(1, (d / step).roundToInt())
    return (0..n).map { i -> visit(a + (b - a) * (i.toDouble() / n), "s1", t++) }
  }

  private fun p(x: Double, z: Double) = Vec3(x, 1.4, z)

  @Test
  fun lShapedWalkThroughDoorwayRoutesAlongTheWalk() {
    val g = WalkGraph()
    // Room A along +x, doorway at (4, 0), room B along -z. A wall separates start and end.
    g.walk(p(0.0, 0.0), p(4.0, 0.0))
    g.walk(p(4.0, 0.0), p(4.0, -4.0))
    val walked = 8.0
    val straight = horizontalDistance(p(0.0, 0.0), p(4.0, -4.0))

    val r = g.route(p(0.0, 0.0), p(4.0, -4.0))
    assertNotNull(r)
    r!!
    assertEquals(walked, r.lengthM, 0.5)
    assertTrue("route ${r.lengthM} vs straight $straight", r.lengthM > 1.35 * straight)
    // About one node per 0.5 m.
    assertTrue(g.nodeCount in 14..20)
    // Every edge is short enough to have been walked.
    for ((a, b) in g.edges()) {
      assertTrue(horizontalDistance(g.node(a)!!.position, g.node(b)!!.position) <= WalkGraph.MAX_EDGE_M)
    }
  }

  @Test
  fun revisitsMergeIntoExistingNodes() {
    val g = WalkGraph()
    g.walk(p(0.0, 0.0), p(3.0, 0.0))
    val n = g.nodeCount
    val e = g.edgeTotal
    // Walk back along the same line (a few cm off): no new nodes, no new edges.
    g.walk(p(3.0, 0.05), p(0.0, 0.05))
    assertEquals(n, g.nodeCount)
    assertEquals(e, g.edgeTotal)
    // A second session's walk in the same place also merges.
    g.breakChain()
    g.walk(p(0.0, -0.1), p(3.0, -0.1))
    assertEquals(n, g.nodeCount)
  }

  @Test
  fun noEdgeAcrossAJump() {
    val g = WalkGraph()
    g.visit(p(0.0, 0.0), "s1", 0)
    g.visit(p(3.0, 0.0), "s1", 1) // 3 m in one step: tracking jump
    assertEquals(2, g.nodeCount)
    assertEquals(0, g.edgeTotal)
    assertNull(g.route(p(0.0, 0.0), p(3.0, 0.0)))
  }

  @Test
  fun noEdgeAcrossABrokenChain() {
    val g = WalkGraph()
    g.visit(p(0.0, 0.0), "s1", 0)
    g.breakChain()
    g.visit(p(0.6, 0.0), "s1", 1) // close enough for an edge, but the chain was broken
    assertEquals(2, g.nodeCount)
    assertEquals(0, g.edgeTotal)
    g.visit(p(1.2, 0.0), "s1", 2)
    assertEquals(1, g.edgeTotal)
  }

  @Test
  fun newNodeOnlyAfterHalfAMetre() {
    val g = WalkGraph()
    g.visit(p(0.0, 0.0), "s1", 0)
    g.visit(p(0.4, 0.0), "s1", 1) // outside the snap radius but under 0.5 m from the current node
    assertEquals(1, g.nodeCount)
    g.visit(p(0.55, 0.0), "s1", 2)
    assertEquals(2, g.nodeCount)
    assertTrue(g.hasEdge(0, 1))
  }

  @Test
  fun routeEndsMustBeNearTheGraph() {
    val g = WalkGraph()
    g.walk(p(0.0, 0.0), p(3.0, 0.0))
    assertNotNull(g.route(p(0.0, 1.4), p(3.0, 0.0)))
    assertNull(g.route(p(0.0, 1.6), p(3.0, 0.0)))
  }

  @Test
  fun routeRemainingAndWaypoint() {
    val g = WalkGraph()
    g.walk(p(0.0, 0.0), p(4.0, 0.0))
    g.walk(p(4.0, 0.0), p(4.0, -4.0))
    val r = g.route(p(0.0, 0.0), p(4.0, -4.0))!!
    // Halfway along the first leg, 2 m done.
    assertEquals(r.lengthM - 2.0, r.remainingFrom(p(2.0, 0.0)), 0.05)
    // 0.5 m off the route counts the way back to it.
    assertEquals(r.lengthM - 2.0 + 0.5, r.remainingFrom(p(2.0, 0.5)), 0.05)
    // The waypoint 1.5 m ahead turns the corner.
    val w = r.pointAhead(p(3.5, 0.0), 1.5)
    assertEquals(4.0, w.x, 0.05)
    assertEquals(-1.0, w.z, 0.05)
    // Past the end it clamps to the end.
    val end = r.pointAhead(p(4.0, -3.9), 5.0)
    assertEquals(-4.0, end.z, 1e-6)
  }

  @Test
  fun unlabelledNodesTakeTheCurrentRoom() {
    val g = WalkGraph()
    g.visit(p(0.0, 0.0), "s1", 0)
    assertEquals(1, g.pickRoom("kitchen", p(0.0, 0.0)))
    g.walk(p(0.0, 0.0), p(2.0, 0.0))
    assertTrue(g.nodes().all { it.roomId == "kitchen" })
    assertEquals("kitchen", g.roomAt(p(1.0, 0.5)))
    assertNull(g.roomAt(p(1.0, 2.0)))
  }

  @Test
  fun walkingNeverOverwritesAndFollowsAfterTwoNodes() {
    val g = WalkGraph()
    // Bedroom already labelled from an earlier walk.
    g.pickRoom("bedroom", null)
    g.walk(p(5.0, 0.0), p(8.0, 0.0))
    g.breakChain()
    // Now in the hallway, walking into the bedroom.
    g.pickRoom("hall", null)
    g.walk(p(2.0, 0.0), p(4.5, 0.0))
    assertEquals("hall", g.currentRoom)
    val first = g.visit(p(5.0, 0.0), "s1", t++)
    assertNull(first.followedRoom)
    assertEquals("hall", g.currentRoom) // one bedroom node is not enough
    val second = g.visit(p(5.5, 0.0), "s1", t++)
    assertEquals("bedroom", second.followedRoom)
    assertEquals("bedroom", g.currentRoom)
    // Once followed, further bedroom nodes are simply the current room.
    assertNull(g.visit(p(6.0, 0.0), "s1", t++).followedRoom)
    // The bedroom labels were never overwritten by walking with "hall" as the current room.
    assertTrue(g.nodes().filter { it.x >= 5.0 - 1e-9 }.all { it.roomId == "bedroom" })
    // Hall nodes stay hall.
    assertTrue(g.nodes().filter { it.x <= 4.5 }.all { it.roomId == "hall" })
  }

  @Test
  fun explicitPickRelabelsWithinOneMetre() {
    val g = WalkGraph()
    g.pickRoom("a", null)
    g.walk(p(0.0, 0.0), p(5.0, 0.0))
    val changed = g.pickRoom("b", p(2.0, 0.0))
    val relabelled = g.nodes().filter { it.roomId == "b" }
    assertEquals(changed, relabelled.size)
    assertTrue(changed in 3..6)
    assertTrue(relabelled.all { horizontalDistance(it.position, p(2.0, 0.0)) <= WalkGraph.RELABEL_M + 1e-9 })
    assertTrue(g.nodes().filter { horizontalDistance(it.position, p(2.0, 0.0)) > 1.0 }.all { it.roomId == "a" })
    assertEquals("b", g.currentRoom)
    // "No room" clears labels nearby.
    val cleared = g.pickRoom(null, p(5.0, 0.0))
    assertTrue(cleared > 0)
    assertNull(g.currentRoom)
  }

  @Test
  fun roundTripThroughNodesAndEdges() {
    val g = WalkGraph()
    g.pickRoom("a", null)
    g.walk(p(0.0, 0.0), p(3.0, 0.0))
    val h = WalkGraph(g.nodes(), g.edges())
    assertEquals(g.nodes(), h.nodes())
    assertEquals(g.edges(), h.edges())
    assertNull(h.currentNodeId) // session start breaks the chain
    assertFalse(h.currentRoom != null)
    // New nodes continue the id sequence.
    h.visit(p(10.0, 0.0), "s2", 99)
    assertEquals(g.nodeCount, h.nodes().last().id)
  }
}
