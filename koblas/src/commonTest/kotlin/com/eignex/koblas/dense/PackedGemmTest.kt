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

    @Test
    fun `the triangular tile walk skips the opposite half`() {
        for (lower in booleanArrayOf(true, false)) {
            var calls = 0
            val recording = object : F64Kernels by F64ScalarKernels {
                override fun gemmTile(
                    depth: Int,
                    packedA: DoubleArray,
                    aOff: Int,
                    packedB: DoubleArray,
                    bOff: Int,
                    c: DoubleArray,
                    cOff: Int,
                    ldc: Int,
                ) {
                    calls++
                    F64ScalarKernels.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
                }
            }
            val order = 16
            val depth = 3
            val a = DoubleArray(order * depth) { (it + 1).toDouble() }
            val c = DoubleArray(order * order)

            packedTriangularGemm(
                recording, 1.0, a, order, false, a, order, true, c, order, depth, lower, null,
            )

            // Four tiles per side: only the diagonal and the six tiles in the selected half are evaluated.
            assertEquals(10, calls, "lower=$lower")
        }
    }
}
