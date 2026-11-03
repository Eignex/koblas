package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.dense.*
import kotlin.test.Test
import kotlin.test.assertTrue

class TriangularKernelsTest {
    // The scheduling does clamp what it is given, but a recommendation wider than the call has right-hand
    // sides answers about work that is not there, and the next caller need not clamp.
    @Test
    fun `a substitution backend recommends a group inside the sides it was asked about`() =
        withBackends { _, _, triangles ->
            for (sides in intArrayOf(0, 1, 2, 3, 7, 64, 1024)) {
                for (order in intArrayOf(1, TRIANGULAR_DIAGONAL_BLOCK)) {
                    val group = triangles.rightHandSideGroup(order, sides)

                    assertTrue(group >= 1, "${triangles.name} recommended $group for $sides sides")
                    if (sides > 0) {
                        assertTrue(
                            group <= sides,
                            "${triangles.name} recommended $group for $sides sides at order $order",
                        )
                    }
                }
            }
        }
}
