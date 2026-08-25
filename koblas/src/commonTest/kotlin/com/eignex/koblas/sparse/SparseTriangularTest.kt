package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.trsv
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
                    trsv(dense, fromDense, lower, transpose)
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
    fun `the solution reproduces the right-hand side`() {
        val rng = Random(20260812)
        val n = 9
        val (sparse, _) = triangle(n, lower = true, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val x = b.copyOf()
        sparse.trsv(x, lower = true)
        assertClose(b, sparse.gemv(x), "lower residual", tolerance = 1e-9)

        val xt = b.copyOf()
        sparse.trsv(xt, lower = true, transpose = true)
        assertClose(b, sparse.gemv(xt, transpose = true), "transposed residual", tolerance = 1e-9)
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
    fun `a missing or zero diagonal is reported with its position`() {
        val missing = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf()))
        val absent = assertFailsWith<IllegalArgumentException> { missing.trsv(DoubleArray(2), lower = true) }
        assertTrue("no diagonal entry at 1" in absent.message!!, absent.message!!)

        val zeroed = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 0.0)))
        val explicit = assertFailsWith<IllegalArgumentException> { zeroed.trsv(DoubleArray(2), lower = true) }
        assertTrue("explicit zero" in explicit.message!!, explicit.message!!)
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
        }
        registerBackend(counting)
        val t = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
        t.trsv(doubleArrayOf(2.0, 4.0), lower = true)
        assertEquals(1, calls, "the extension must forward to the seam")
    }
}
