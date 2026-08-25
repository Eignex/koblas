package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.lu
import com.eignex.koblas.sparse.gemv
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Paths the shape-driven suites step over: a destination that aliases its source, the storage combinations
 * the generic entry points dispatch on, and each rejection the container constructors make.
 */
class BranchCoverageTest {

    @Test
    fun `an LU solve into its own right-hand side matches one into a fresh vector`() {
        val rng = Random(20260820)
        for (n in intArrayOf(1, 2, 6, 11)) {
            val lu = wellConditioned(n, rng).lu()
            for (transpose in booleanArrayOf(false, true)) {
                val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
                val fresh = koblas.solveInto(lu, b, DoubleArray(n), transpose)
                val aliased = b.copyOf()
                koblas.solveInto(lu, aliased, aliased, transpose)
                assertClose(fresh, aliased, "aliased LU solve n=$n t=$transpose", tolerance = 1e-12)
            }
        }
    }

    @Test
    fun `a blocked LU solve into its own right-hand side matches one into a fresh matrix`() {
        val rng = Random(20260821)
        val n = 6
        for (nrhs in intArrayOf(1, 5)) {
            val lu = wellConditioned(n, rng).lu()
            for (transpose in booleanArrayOf(false, true)) {
                val b = F64DenseMatrix(n, nrhs, DoubleArray(n * nrhs) { rng.nextDouble(-1.0, 1.0) })
                val fresh = koblas.solveInto(lu, b, F64DenseMatrix.zero(n, nrhs), transpose)
                val aliased = F64DenseMatrix(n, nrhs, b.data.copyOf())
                koblas.solveInto(lu, aliased, aliased, transpose)
                assertClose(fresh, aliased, "aliased blocked solve nrhs=$nrhs t=$transpose", tolerance = 1e-12)
            }
        }
    }

    /** The transposed sparse product and the beta scalings, which the untransposed suites never select. */
    @Test
    fun `the sparse product covers both orientations and every beta`() {
        val rng = Random(20260822)
        val n = 5
        val a = F64SparseMatrix.ofTriplets(
            rows = n,
            cols = n,
            rowIdx = intArrayOf(0, 2, 1, 3, 4, 2, 0),
            colIdx = intArrayOf(0, 0, 1, 1, 2, 3, 4),
            values = doubleArrayOf(2.0, -1.0, 3.0, 0.5, 4.0, -2.0, 1.5),
        )
        val dense = F64DenseMatrix(n, n)
        for (j in 0 until n) a.forEachInColumn(j) { i, v -> dense[i, j] = v }
        for (transpose in booleanArrayOf(false, true)) {
            for (alpha in doubleArrayOf(0.0, 1.0, -0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                    val x = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
                    val y0 = DoubleArray(n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = y0.copyOf()
                    koblas.gemv(alpha, dense, x, beta, expected, transpose)
                    val actual = y0.copyOf()
                    koblas.gemv(alpha, a, x, beta, actual, transpose)
                    assertClose(expected, actual, "sparse gemv t=$transpose a=$alpha b=$beta", tolerance = 1e-12)
                }
            }
        }
    }

    /** [times] dispatches on both operands' storage, and only the dense pair is otherwise exercised. */
    @Test
    fun `matrix vector product agrees across every storage combination`() {
        val rows = 4
        val cols = 3
        val entries = arrayOf(
            doubleArrayOf(1.0, 0.0, 2.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(-3.0, 4.0, 0.0),
            doubleArrayOf(0.0, 0.5, 1.5),
        )
        val dense = F64DenseMatrix.of(entries)
        val sparse = F64SparseMatrix.ofTriplets(
            rows,
            cols,
            intArrayOf(0, 0, 2, 2, 3, 3),
            intArrayOf(0, 2, 0, 1, 1, 2),
            doubleArrayOf(1.0, 2.0, -3.0, 4.0, 0.5, 1.5),
        )
        // A F64MatrixLike koblas does not know, so multiplication takes its entry-by-entry path.
        val foreign = object : F64MatrixLike {
            override val rows: Int get() = dense.rows
            override val cols: Int get() = dense.cols
            override fun get(i: Int, j: Int): Double = dense[i, j]
            override fun toArray(): Array<DoubleArray> = dense.toArray()
        }
        val denseX = F64DenseVector.of(doubleArrayOf(2.0, -1.0, 0.5))
        // The same vector with its middle entry dropped, so the two expectations differ in rows 2 and 3.
        val sparseX = F64SparseVector.of(3, intArrayOf(0, 2), doubleArrayOf(2.0, 0.5))
        val expectedDense = doubleArrayOf(3.0, 0.0, -10.0, 0.25)
        val expectedSparse = doubleArrayOf(3.0, 0.0, -6.0, 0.75)
        for ((name, matrix) in listOf<Pair<String, F64MatrixLike>>(
            "dense" to dense,
            "sparse" to sparse,
            "foreign" to foreign,
        )) {
            assertClose(expectedDense, (matrix * denseX).data, "$name x dense", tolerance = 1e-12)
            assertClose(expectedSparse, (matrix * sparseX).data, "$name x sparse", tolerance = 1e-12)
        }
    }

    @Test
    fun `the CSC constructor rejects each way the arrays can disagree`() {
        val cases = listOf<Pair<String, () -> Unit>>(
            "negative rows" to { F64SparseMatrix.wrap(-1, 1, intArrayOf(0, 0), IntArray(0), DoubleArray(0)) },
            "negative cols" to { F64SparseMatrix.wrap(1, -1, intArrayOf(0), IntArray(0), DoubleArray(0)) },
            "short colPtr" to { F64SparseMatrix.wrap(2, 2, intArrayOf(0, 1), IntArray(0), DoubleArray(0)) },
            "misaligned values" to { F64SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 1), intArrayOf(0), DoubleArray(0)) },
            "nonzero colPtr head" to {
                F64SparseMatrix.wrap(2, 2, intArrayOf(1, 1, 1), intArrayOf(0), doubleArrayOf(1.0))
            },
            "colPtr tail is not nnz" to {
                F64SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 1), intArrayOf(0), doubleArrayOf(1.0, 2.0))
            },
            "colPtr not monotonic" to {
                F64SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 1), intArrayOf(0), doubleArrayOf(1.0))
            },
            "row out of range" to {
                F64SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 1), intArrayOf(5), doubleArrayOf(1.0))
            },
            "rows not ascending" to {
                F64SparseMatrix.wrap(2, 1, intArrayOf(0, 2), intArrayOf(1, 0), doubleArrayOf(1.0, 2.0))
            },
            "rows repeated" to {
                F64SparseMatrix.wrap(2, 1, intArrayOf(0, 2), intArrayOf(1, 1), doubleArrayOf(1.0, 2.0))
            },
        )
        for ((name, build) in cases) {
            assertFailsWith<IllegalArgumentException>("$name should be rejected") { build() }
        }
    }

    @Test
    fun `the sparse vector constructor rejects each way the arrays can disagree`() {
        assertFailsWith<IllegalArgumentException>("misaligned") {
            F64SparseVector.wrap(3, intArrayOf(0, 1), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException>("negative index") {
            F64SparseVector.wrap(3, intArrayOf(-1), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException>("index past the end") {
            F64SparseVector.wrap(3, intArrayOf(3), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException>("not ascending") {
            F64SparseVector.wrap(3, intArrayOf(1, 0), doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<IllegalArgumentException>("repeated") {
            F64SparseVector.wrap(3, intArrayOf(1, 1), doubleArrayOf(1.0, 2.0))
        }
    }

    /** `of` repairs what `wrap` rejects: it sorts by index and sums the duplicates. */
    @Test
    fun `the sparse vector factory sorts and sums where wrap refuses`() {
        val v = F64SparseVector.of(4, intArrayOf(3, 0, 3), doubleArrayOf(1.0, 2.0, 0.5))
        assertEquals(listOf(0, 3), v.indices.toList())
        assertEquals(listOf(2.0, 1.5), v.values.toList())
        assertEquals(0.0, v[1], "an unstored position reads as zero")
        assertFailsWith<IllegalArgumentException> { F64SparseVector.of(4, intArrayOf(0), doubleArrayOf(1.0, 2.0)) }
    }

    /** The condition estimate's sweeps, including the ill-conditioned case its safeguard exists for. */
    @Test
    fun `rcond spans well and ill conditioned matrices`() {
        val identity = F64DenseMatrix.diagonal(6)
        assertTrue(
            abs(koblas.rcond(identity.lu(), identity.norm1()) - 1.0) < 1e-9,
            "the identity should estimate a reciprocal condition of 1",
        )
        // A Hilbert-like matrix, ill conditioned enough that the estimate has to iterate.
        val n = 8
        val hilbert = F64DenseMatrix(n, n)
        for (i in 0 until n) for (j in 0 until n) hilbert[i, j] = 1.0 / (i + j + 1.0)
        val estimate = koblas.rcond(hilbert.lu(), hilbert.norm1())
        assertTrue(estimate > 0.0 && estimate < 1e-6, "an ill-conditioned estimate should be tiny, got $estimate")
        // A zero matrix is singular, and a zero norm reports zero instead of dividing.
        assertEquals(0.0, koblas.rcond(F64DenseMatrix.zero(3, 3).lu(), 0.0))
        assertEquals(1.0, koblas.rcond(F64DenseMatrix.zero(0, 0).lu(), 0.0), "an empty matrix is perfectly conditioned")
    }
}
