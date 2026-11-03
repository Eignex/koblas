package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.dense.*
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ProductWindowTest {
    /**
     * A product too small or too thin to hide a copy is not packed, and a large square one is.
     *
     * The rule is the backend's, so what is asserted is its shape rather than its constants: an extent below
     * one tile cannot fill the tiles it would pack into, and a matrix-vector shaped call has no arithmetic to
     * amortise a copy over however large its other extent is.
     */
    @Test
    fun `packing is declined where there is nothing to amortise it over`() = withProductKernels { kernels ->
        assertFalse(kernels.packsProduct(1, 1, 1), "${kernels.name} packed a single product")
        assertFalse(kernels.packsProduct(4096, 1, 4096), "${kernels.name} packed a matrix-vector product")
        assertFalse(kernels.packsProduct(1, 4096, 4096), "${kernels.name} packed a vector-matrix product")
        assertTrue(kernels.packsProduct(256, 256, 256), "${kernels.name} declined a large square product")
    }

    // A fixture chosen on one machine can quietly become a test of the unpacked route on another, so the rule
    // is asked directly at tile widths this host may lack: both extents must reach a whole tile, and the work
    // must reach the threshold.
    @Test
    fun `the packing rule needs a whole tile on each axis and enough work at every tile geometry`() {
        for (tileRows in intArrayOf(2, 4, 8, 16, 32)) {
            val tileColumns = 4
            val depth = (PACKED_PRODUCT_MINIMUM_WORK / (tileRows * tileColumns)).toInt() + 1
            val context = "a tile of $tileRows by $tileColumns"

            assertTrue(packsProductByWork(tileRows, tileColumns, depth, tileRows, tileColumns), context)
            assertFalse(packsProductByWork(tileRows - 1, 4096, depth, tileRows, tileColumns), "$context rows")
            assertFalse(packsProductByWork(4096, tileColumns - 1, depth, tileRows, tileColumns), "$context columns")
            assertFalse(packsProductByWork(tileRows, tileColumns, depth - 2, tileRows, tileColumns), "$context work")
        }
    }
}
