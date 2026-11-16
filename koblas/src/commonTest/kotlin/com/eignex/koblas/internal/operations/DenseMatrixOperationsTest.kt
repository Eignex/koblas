package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

class DenseMatrixOperationsTest {
    @Test
    fun `symmetric rank updates preserve a shared Cholesky buffer`() = withRegionEngines { engine ->
        for (width in intArrayOf(1, 3)) {
            val parent = paddedRegion(9, 9)
            val first = 1 + width
            val height = parent.rows - first
            val panel = parent.view(first, 1, height, width)
            val trailing = parent.view(first, first, height, height)
            val expected = contiguousCopy(trailing)
            val before = parent.values.copyOf()
            ReferenceBlas.syrk(-1.0, contiguousCopy(panel), false, 1.0, expected)

            engine.syrk(-1.0, panel, false, 1.0, trailing, scratch = Scratch())

            assertRegionAgreesWithReference(expected, trailing, before, "Cholesky trailing update width=$width")
        }
    }

    @Test
    fun `custom symmetric rank overrides receive original views and defaults`() {
        val parent = paddedRegion(7, 7)
        val panel = parent.view(3, 1, 4, 2)
        val trailing = parent.view(3, 3, 4, 4)
        for (workspace in listOf(null, Scratch())) {
            var calls = 0
            val engine: KoblasEngine = object : KoblasEngine {
                override val name: String = "custom"
                override fun syrk(
                    alpha: Double,
                    a: DenseMatrix,
                    transpose: Boolean,
                    beta: Double,
                    c: DenseMatrix,
                    lower: Boolean,
                    scratch: Scratch?,
                ) {
                    assertSame(panel, a)
                    assertSame(trailing, c)
                    assertSame(workspace, scratch)
                    assertEquals(-1.0, alpha)
                    assertEquals(1.0, beta)
                    assertFalse(transpose)
                    assertTrue(lower)
                    calls++
                }
            }

            engine.syrk(-1.0, panel, false, 1.0, trailing, scratch = workspace)

            assertEquals(1, calls)
        }
    }
}
