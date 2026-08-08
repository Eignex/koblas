package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.dense.trsv
import com.eignex.koblas.registerBackend
import com.eignex.koblas.withCleanBackends
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** The sparse triangular solve, checked against the dense one. */
class SparseTriangularTest {

    /** A random triangle in both storages, with a comfortably non-zero diagonal. */
    private fun triangle(n: Int, lower: Boolean, rng: Random, density: Double = 0.4): Pair<SparseMatrix, DenseMatrix> {
        val dense = DenseMatrix(n, n)
        val columns = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                val inTriangle = if (lower) i >= j else i <= j
                if (!inTriangle) continue
                val v = when {
                    // A diagonal of at least 2 keeps the triangle well away from singular.
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

    /** The solution must actually satisfy the system, not merely match another implementation. */
    @Test
    fun `the solution reproduces the right-hand side`() {
        val rng = Random(20260812)
        val n = 9
        val (sparse, _) = triangle(n, lower = true, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val x = b.copyOf()
        sparse.trsv(x, lower = true)
        // T · x must be b again, computed independently through gemv.
        assertClose(b, sparse.gemv(x), "lower residual", tolerance = 1e-9)

        val xt = b.copyOf()
        sparse.trsv(xt, lower = true, transpose = true)
        assertClose(b, sparse.gemv(xt, transpose = true), "transposed residual", tolerance = 1e-9)
    }

    /** Only the selected triangle may be read, so junk in the other one must not matter. */
    @Test
    fun `the opposite triangle is never read`() {
        val rng = Random(20260813)
        val n = 7
        val (clean, dense) = triangle(n, lower = true, rng)
        // Same lower triangle, plus NaN scattered through the strict upper one.
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
        val fromPoisoned = b.copyOf().also { SparseMatrix.ofColumns(n, n, poisoned).trsv(it, lower = true) }
        assertClose(fromClean, fromPoisoned, "a NaN in the unread triangle changed the answer", tolerance = 0.0)
    }

    /**
     * A structurally missing diagonal is the sparse-specific failure: a dense triangle always has the entry even when
     * it is zero, so the dense cores can leave it to produce infinities. Here it is free to detect.
     */
    @Test
    fun `a missing or zero diagonal is reported with its position`() {
        // Column 1 has no diagonal entry at all.
        val missing = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf()))
        val absent = assertFailsWith<IllegalArgumentException> { missing.trsv(DoubleArray(2), lower = true) }
        assertTrue("no diagonal entry at 1" in absent.message!!, absent.message!!)

        // Column 1 stores a diagonal, but it is an explicit zero — a different mistake, so a different message.
        val zeroed = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 0.0)))
        val explicit = assertFailsWith<IllegalArgumentException> { zeroed.trsv(DoubleArray(2), lower = true) }
        assertTrue("explicit zero" in explicit.message!!, explicit.message!!)
    }

    @Test
    fun `it reaches the registered backend`() = withCleanBackends {
        var calls = 0
        val counting = object : SparseBlas {
            override val name: String get() = "counting"
            override fun trsv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean) {
                calls++
                super.trsv(a, x, lower, transpose)
            }
        }
        registerBackend(counting)
        val t = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
        t.trsv(doubleArrayOf(2.0, 4.0), lower = true)
        assertEquals(1, calls, "the extension must forward to the seam")
    }
}
