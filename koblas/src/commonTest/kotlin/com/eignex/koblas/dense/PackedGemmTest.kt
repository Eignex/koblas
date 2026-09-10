package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * The packed product against a written-out one, driven directly so the shapes here are exercised whatever
 * crossover the installed kernels declare.
 *
 * The sizes straddle the tile: one and two are below any tile, seven and nine sit either side of an eight
 * row tile, and 300 crosses the depth block so more than one packed panel is accumulated into the same
 * tile of C.
 */
internal fun assertPackedGemmAgreesWithWrittenOutProduct(kernels: PackedKernels) {
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
    fun `ordinary product uses the retained panel ordering`() {
        val rows = PortablePackedKernels.gemmTileRows - 1
        val columns = PortablePackedKernels.gemmTileCols - 1
        val depth = 3
        val alpha = -0.75
        val a = DenseMatrix(depth, rows, DoubleArray(rows * depth) { it + 1.0 })
        val b = DenseMatrix(columns, depth, DoubleArray(columns * depth) { 100.0 + it })
        val expectedA = DoubleArray(packedLeftSize(rows, depth, PortablePackedKernels.gemmTileRows))
        val expectedB = DoubleArray(packedRightSize(depth, columns, PortablePackedKernels.gemmTileCols))
        packLeftPanel(
            a, expectedA, rows, depth, 0, 0, transpose = true, alpha = alpha,
            destinationOffset = 0, workspace = null, structure = PackedPanelStructure.General,
            tileRows = PortablePackedKernels.gemmTileRows,
        )
        packRightPanel(
            b, expectedB, depth, columns, 0, 0, transpose = true,
            destinationOffset = 0, workspace = null, structure = PackedPanelStructure.General,
            tileColumns = PortablePackedKernels.gemmTileCols,
        )
        val recording = object : PackedKernels by PortablePackedKernels {
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
                assertContentEquals(expectedA, packedA.copyOfRange(aOff, aOff + expectedA.size))
                assertContentEquals(expectedB, packedB.copyOfRange(bOff, bOff + expectedB.size))
                PortablePackedKernels.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
            }
        }

        packedGemm(
            recording, alpha, a.data, a.rows, true, b.data, b.rows, true,
            DoubleArray(rows * columns), rows, columns, depth, null,
        )
    }

    @Test
    fun `the packed product on the portable tile agrees with a written out product`() {
        assertPackedGemmAgreesWithWrittenOutProduct(PortablePackedKernels)
    }

    @Test
    fun `the packed product on the compiled in kernels agrees with a written out product`() {
        assertPackedGemmAgreesWithWrittenOutProduct(platformDenseKernelFamilies.packed)
    }

    @Test
    fun `the triangular tile walk skips the opposite half`() {
        for (lower in booleanArrayOf(true, false)) {
            var calls = 0
            val recording = object : PackedKernels by PortablePackedKernels {
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
                    PortablePackedKernels.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
                }
            }
            val order = 16
            val depth = 3
            val a = DoubleArray(order * depth) { (it + 1).toDouble() }
            val c = DoubleArray(order * order)

            packedTriangularGemm(
                recording,
                1.0, a, order, false, a, order, true, c, order, depth, lower, null,
            )

            // Four tiles per side: only the diagonal and the six tiles in the selected half are evaluated.
            assertEquals(10, calls, "lower=$lower")
        }
    }
}
