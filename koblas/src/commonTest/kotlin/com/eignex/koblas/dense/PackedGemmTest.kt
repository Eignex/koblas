package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The packed product against a written-out one, driven directly so the shapes here are exercised whatever
 * crossover the installed kernels declare.
 *
 * The sizes straddle the tile: one and two are below any tile, seven and nine sit either side of an eight
 * row tile, and 300 crosses the depth block so more than one packed panel is accumulated into the same
 * tile of C.
 */
internal fun assertPackedGemmAgreesWithWrittenOutProduct(kernels: F64Kernels) {
    val rng = Random(20260907)
    val alpha = -0.75
    val sizes = intArrayOf(1, 2, 7, 8, 9, 16, 33)
    for (m in sizes) {
        for (n in sizes) {
            for (k in intArrayOf(1, 5, 17, 300)) {
                for (transposeA in booleanArrayOf(false, true)) {
                    for (transposeB in booleanArrayOf(false, true)) {
                        val lda = if (transposeA) k else m
                        val ldb = if (transposeB) n else k
                        val a = DoubleArray(m * k) { rng.nextDouble(-1.0, 1.0) }
                        val b = DoubleArray(k * n) { rng.nextDouble(-1.0, 1.0) }
                        val expected = DoubleArray(m * n)
                        for (j in 0 until n) {
                            for (p in 0 until k) {
                                val right = if (transposeB) b[j + p * ldb] else b[p + j * ldb]
                                val scaled = alpha * right
                                for (i in 0 until m) {
                                    val left = if (transposeA) a[p + i * lda] else a[i + p * lda]
                                    expected[i + j * m] += left * scaled
                                }
                            }
                        }
                        val actual = DoubleArray(m * n)
                        packedGemm(
                            kernels, alpha, a, lda, transposeA, b, ldb, transposeB, actual, m, n, k, null,
                        )
                        assertClose(
                            expected,
                            actual,
                            context = "m=$m n=$n k=$k transposeA=$transposeA transposeB=$transposeB",
                        )
                    }
                }
            }
        }
    }
}

class PackedGemmTest {
    @Test
    fun `the packed product on the portable tile agrees with a written out product`() {
        assertPackedGemmAgreesWithWrittenOutProduct(F64ScalarKernels)
    }

    @Test
    fun `the packed product on the compiled in kernels agrees with a written out product`() {
        assertPackedGemmAgreesWithWrittenOutProduct(F64PlatformKernels)
    }
}

/**
 * The wrapper the registry composes must hand out the tile the compiled-in kernels implement. Falling back
 * to the interface default here is correct and slow, so nothing else would catch it.
 */
class RoutedTileTest {
    @Test
    fun `the routed kernels expose the compiled in tile`() {
        val routed: F64Kernels = F64RoutedKernels(null)
        assertEquals(F64PlatformKernels.gemmTileRows, routed.gemmTileRows)
        assertEquals(F64PlatformKernels.gemmTileCols, routed.gemmTileCols)

        val depth = 3
        val rows = routed.gemmTileRows
        val cols = routed.gemmTileCols
        val packedA = DoubleArray(depth * rows) { (it + 1).toDouble() / 7.0 }
        val packedB = DoubleArray(depth * cols) { (it + 2).toDouble() / 11.0 }
        val expected = DoubleArray(rows * cols)
        val actual = DoubleArray(rows * cols)
        F64PlatformKernels.gemmTile(depth, packedA, 0, packedB, 0, expected, 0, rows)
        routed.gemmTile(depth, packedA, 0, packedB, 0, actual, 0, rows)
        assertClose(expected, actual, context = "routed tile")
    }
}
