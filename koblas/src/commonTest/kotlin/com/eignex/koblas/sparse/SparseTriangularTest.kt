package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.trsm
import com.eignex.koblas.dense.trsv
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.*

class SparseTriangularTest {

    private fun triangle(
        n: Int,
        lower: Boolean,
        rng: Random,
        density: Double = 0.4,
    ): Pair<F64SparseMatrix, F64DenseMatrix> {
        val dense = F64DenseMatrix(n, n)
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
        return F64SparseMatrix.ofColumns(n, n, columns) to dense
    }

    @Test
    fun `every direction agrees with the dense solve`() {
        val rng = Random(20260811)
        for (n in intArrayOf(1, 2, 6, 11)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    val (sparse, dense) = triangle(n, lower, rng)
                    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

                    val fromDense = b.copyOf()
                    dense.trsv(fromDense, lower, transpose)
                    val fromSparse = b.copyOf()
                    sparse.trsv(fromSparse, lower, transpose)

                    assertClose(
                        fromDense,
                        fromSparse,
                        "n=$n lower=$lower transpose=$transpose",
                        tolerance = 1e-9,
                    )
                }
            }
        }
    }

    @Test
    fun `every direction agrees with the dense solve over several right-hand sides`() {
        val rng = Random(20260821)
        val n = 6
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    val (sparse, dense) = triangle(n, lower, rng)
                    val b = if (right) randomMatrix(3, n, rng) else randomMatrix(n, 3, rng)

                    val fromDense = F64DenseMatrix.wrap(b.rows, b.cols, b.data.copyOf())
                    dense.trsm(fromDense, lower, transpose, right = right)
                    val fromSparse = F64DenseMatrix.wrap(b.rows, b.cols, b.data.copyOf())
                    sparse.trsm(fromSparse, lower, transpose, right = right)

                    assertClose(
                        fromDense,
                        fromSparse,
                        "n=$n lower=$lower transpose=$transpose right=$right",
                        tolerance = 1e-9,
                    )
                }
            }
        }
    }

    @Test
    fun `each column of a solve from the left is the one trsv gives it`() {
        val rng = Random(20260822)
        val n = 7
        val (sparse, _) = triangle(n, lower = true, rng)
        val b = randomMatrix(n, 3, rng)

        val fromTrsm = F64DenseMatrix.wrap(n, 3, b.data.copyOf())
        sparse.trsm(fromTrsm, lower = true)

        for (j in 0 until 3) {
            val column = b.data.copyOfRange(j * n, (j + 1) * n)
            sparse.trsv(column, lower = true)
            assertClose(column, fromTrsm.data.copyOfRange(j * n, (j + 1) * n), "column $j", tolerance = 1e-9)
        }
    }

    @Test
    fun `alpha scales the right-hand side before the solve`() {
        val rng = Random(20260823)
        val n = 5
        val (sparse, _) = triangle(n, lower = true, rng)
        val b = randomMatrix(n, 2, rng)

        val scaled = F64DenseMatrix.wrap(n, 2, DoubleArray(b.data.size) { 2.5 * b.data[it] })
        sparse.trsm(scaled, lower = true)
        val withAlpha = F64DenseMatrix.wrap(n, 2, b.data.copyOf())
        sparse.trsm(withAlpha, lower = true, alpha = 2.5)

        assertClose(scaled, withAlpha, "alpha must scale B", tolerance = 1e-9)
    }

    @Test
    fun `an alpha of zero empties the right-hand side without solving`() {
        val n = 3
        // A NaN diagonal, which a solve would spread through the answer.
        val poisoned = F64SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to Double.NaN) })
        val b = F64DenseMatrix.wrap(n, 2, DoubleArray(n * 2) { 1.0 })

        poisoned.trsm(b, lower = true, alpha = 0.0)

        assertTrue(b.data.all { it == 0.0 }, "alpha = 0 must empty B without a solve")
    }

    @Test
    fun `a right-hand side that does not meet the triangle is rejected on both sides`() {
        val rng = Random(20260824)
        val n = 4
        val (sparse, _) = triangle(n, lower = true, rng)

        assertFailsWith<DimensionMismatch> { sparse.trsm(randomMatrix(n + 1, 2, rng), lower = true) }
        assertFailsWith<DimensionMismatch> { sparse.trsm(randomMatrix(2, n + 1, rng), lower = true, right = true) }
    }

    @Test
    fun `the solution reproduces the right-hand side`() {
        val rng = Random(20260812)
        val n = 9
        val (sparse, _) = triangle(n, lower = true, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val x = b.copyOf()
        sparse.trsv(x, lower = true)
        assertClose(b, koblas.gemv(sparse, x), "lower residual", tolerance = 1e-9)

        val xt = b.copyOf()
        sparse.trsv(xt, lower = true, transpose = true)
        assertClose(b, koblas.gemv(sparse, xt, transpose = true), "transposed residual", tolerance = 1e-9)
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
        val fromClean = b.copyOf().also { clean.trsv(it, lower = true) }
        val fromPoisoned = b.copyOf().also { F64SparseMatrix.ofColumns(n, n, poisoned).trsv(it, lower = true) }
        assertClose(fromClean, fromPoisoned, "a NaN in the unread triangle changed the answer", tolerance = 0.0)
    }

    @Test
    fun `a missing or zero diagonal follows IEEE division`() {
        val missing = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf()))
        val absent = doubleArrayOf(0.0, 1.0)
        missing.trsv(absent, lower = true)
        assertEquals(Double.POSITIVE_INFINITY, absent[1])

        val zeroed = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 0.0)))
        val explicit = doubleArrayOf(0.0, -1.0)
        zeroed.trsv(explicit, lower = true)
        assertEquals(Double.NEGATIVE_INFINITY, explicit[1])
    }

    @Test
    fun `sparse trsm retains stored products after a quotient underflows`() {
        val triangle = F64SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to Double.POSITIVE_INFINITY, 1 to Double.POSITIVE_INFINITY), listOf(1 to 1.0)),
        )
        val b = F64DenseMatrix(2, 2, doubleArrayOf(Double.MIN_VALUE, 0.0, 0.0, 0.0))

        triangle.trsm(b, lower = true)

        assertTrue(b[1, 0].isNaN(), "the active underflowed pivot did not form its stored product")
        assertEquals(0.0, b[1, 1], "the zero right hand side was not skipped")
    }

    @Test
    fun `right sparse trsm skips explicit zero triangle coefficients`() {
        val triangle = F64SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to 0.0), listOf(1 to 1.0)),
        )
        val b = F64DenseMatrix(1, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0))

        triangle.trsm(b, lower = true, right = true)

        assertEquals(Double.POSITIVE_INFINITY, b[0, 0])
        assertEquals(1.0, b[0, 1])
    }

    @Test
    fun `unitDiag takes the diagonal as one without reading it`() {
        // A diagonal of 5.0 that must be ignored, and a NaN one that must not even be looked at.
        val lower = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 5.0, 1 to 3.0), listOf(1 to Double.NaN)))
        val x = doubleArrayOf(1.0, 2.0)
        lower.trsv(x, lower = true, unitDiag = true)
        assertEquals(1.0, x[0])
        assertEquals(-1.0, x[1], "row 1 is 3*x0 + 1*x1 = 2")

        val t = doubleArrayOf(1.0, 2.0)
        lower.trsv(t, lower = true, transpose = true, unitDiag = true)
        assertEquals(2.0, t[1])
        assertEquals(-5.0, t[0], "column 0 transposed is 1*x0 + 3*x1 = 1")
    }

    @Test
    fun `it reaches the registered backend`() = withCleanBackends {
        var calls = 0
        val counting = object : F64SparseBlas {
            override val name: String get() = "counting"
            override fun trsv(
                a: F64SparseMatrix,
                x: DoubleArray,
                lower: Boolean,
                transpose: Boolean,
                unitDiag: Boolean,
            ) {
                calls++
                F64ReferenceSparseLinearAlgebra.trsv(a, x, lower, transpose, unitDiag)
            }

            @Suppress("LongParameterList")
            override fun gemv(
                alpha: Double,
                a: F64SparseMatrix,
                x: DoubleArray,
                beta: Double,
                y: DoubleArray,
                transpose: Boolean,
            ) = F64ReferenceSparseLinearAlgebra.gemv(alpha, a, x, beta, y, transpose)

            override fun transpose(a: F64SparseMatrix) = F64ReferenceSparseLinearAlgebra.transpose(a)

            @Suppress("LongParameterList")
            override fun gemm(
                alpha: Double,
                a: F64SparseMatrix,
                transposeA: Boolean,
                b: F64DenseMatrix,
                transposeB: Boolean,
                beta: Double,
                c: F64DenseMatrix,
                right: Boolean,
            ) = F64ReferenceSparseLinearAlgebra.gemm(alpha, a, transposeA, b, transposeB, beta, c, right)

            override fun gemm(a: F64SparseMatrix, b: F64SparseMatrix) = F64ReferenceSparseLinearAlgebra.gemm(a, b)

            @Suppress("LongParameterList")
            override fun trsm(
                a: F64SparseMatrix,
                b: F64DenseMatrix,
                lower: Boolean,
                transpose: Boolean,
                unitDiag: Boolean,
                right: Boolean,
                alpha: Double,
            ) = F64ReferenceSparseLinearAlgebra.trsm(a, b, lower, transpose, unitDiag, right, alpha)
        }
        registerBackend(counting)
        val t = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
        t.trsv(doubleArrayOf(2.0, 4.0), lower = true)
        assertEquals(1, calls, "the extension must forward to the seam")
    }
}
