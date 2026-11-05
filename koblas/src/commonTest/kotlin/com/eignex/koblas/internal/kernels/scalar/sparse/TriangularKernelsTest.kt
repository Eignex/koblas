package com.eignex.koblas.internal.kernels.scalar.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import kotlin.random.Random
import kotlin.test.*

class TriangularKernelsTest {

    private fun triangle(n: Int, lower: Boolean, rng: Random, density: Double = 0.4): Pair<SparseMatrix, DenseMatrix> {
        val dense = DenseMatrix(n, n)
        val columns = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                val inTriangle = if (lower) i >= j else i <= j
                if (!inTriangle) continue
                val v = when {
                    i == j -> 2.0 + rng.nextDouble()
                    rng.nextDouble() < density -> rng.nextDouble(-1.0, 1.0)
                    else -> 0.0
                }
                if (v != 0.0) {
                    column.add(i to v)
                    dense[i, j] = v
                }
            }
            columns.add(column)
        }
        return SparseMatrix.ofColumns(n, n, columns) to dense
    }

    @Test
    fun `every direction of the solve agrees with the reference`() {
        val rng = Random(20260811)
        for (n in intArrayOf(1, 2, 6, 11)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    val (sparse, dense) = triangle(n, lower, rng)
                    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

                    val expected = b.copyOf()
                    ReferenceBlas.trsv(dense, expected, lower, transpose)
                    val actual = b.copyOf()
                    koblas.trsv(sparse, actual, lower, transpose)

                    assertClose(expected, actual, "n=$n lower=$lower transpose=$transpose", tolerance = 1e-9)
                }
            }
        }
    }

    @Test
    fun `every direction of the multiply agrees with the reference`() {
        val rng = Random(20260901)
        for (n in intArrayOf(0, 1, 2, 6, 11)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (sparse, dense) = triangle(n, lower, rng)
                        val input = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

                        val expected = input.copyOf()
                        ReferenceBlas.trmv(dense, expected, lower, transpose, unitDiag)
                        val actual = input.copyOf()
                        koblas.trmv(sparse, actual, lower, transpose, unitDiag)

                        assertClose(expected, actual, "n=$n lower=$lower transpose=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `triangular multiplication treats a missing diagonal as zero without division`() {
        val triangle = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 3.0), listOf()))
        val vector = doubleArrayOf(2.0, 4.0)

        koblas.trmv(triangle, vector, lower = true)

        assertContentEquals(doubleArrayOf(4.0, 6.0), vector)
    }

    @Test
    fun `the opposite triangle is never read`() {
        val rng = Random(20260813)
        val n = 7
        val (clean, dense) = triangle(n, lower = true, rng)
        // Same lower triangle, plus NaN through the strict upper one, which must not reach the answer.
        val poisoned = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                if (i < j) {
                    column.add(i to Double.NaN)
                } else if (dense[i, j] != 0.0) {
                    column.add(i to dense[i, j])
                }
            }
            poisoned.add(column)
        }
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val fromClean = b.copyOf().also { koblas.trsv(clean, it, lower = true) }
        val fromPoisoned = b.copyOf().also { koblas.trsv(SparseMatrix.ofColumns(n, n, poisoned), it, lower = true) }

        assertClose(fromClean, fromPoisoned, "a NaN in the unread triangle changed the answer", tolerance = 0.0)
    }

    @Test
    fun `unitDiag takes the diagonal as one without reading it`() {
        // A diagonal of 5.0 that must be ignored, and a NaN one that must not even be looked at.
        val lower = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 5.0, 1 to 3.0), listOf(1 to Double.NaN)))
        val x = doubleArrayOf(1.0, 2.0)
        koblas.trsv(lower, x, lower = true, unitDiag = true)
        assertEquals(1.0, x[0])
        assertEquals(-1.0, x[1], "row 1 is 3*x0 + 1*x1 = 2")

        val t = doubleArrayOf(1.0, 2.0)
        koblas.trsv(lower, t, lower = true, transpose = true, unitDiag = true)
        assertEquals(2.0, t[1])
        assertEquals(-5.0, t[0], "column 0 transposed is 1*x0 + 3*x1 = 1")
    }
}
