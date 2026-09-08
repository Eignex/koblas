package com.eignex.koblas

import com.eignex.koblas.*
import kotlin.random.Random
import kotlin.test.*

/**
 * Paths the shape-driven suites step over: a destination that aliases its source, the storage combinations
 * the generic entry points dispatch on, and each rejection the container constructors make.
 */
class BranchCoverageTest {

    /** The transposed sparse product and the beta scalings, which the untransposed suites never select. */
    @Test
    fun `the sparse product covers both orientations and every beta`() {
        val rng = Random(20260822)
        val n = 5
        val a = SparseMatrix.ofTriplets(
            rows = n,
            cols = n,
            rowIdx = intArrayOf(0, 2, 1, 3, 4, 2, 0),
            colIdx = intArrayOf(0, 0, 1, 1, 2, 3, 4),
            values = doubleArrayOf(2.0, -1.0, 3.0, 0.5, 4.0, -2.0, 1.5),
        )
        val dense = DenseMatrix(n, n)
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
        val dense = DenseMatrix.of(entries)
        val sparse = SparseMatrix.ofTriplets(
            rows,
            cols,
            intArrayOf(0, 0, 2, 2, 3, 3),
            intArrayOf(0, 2, 0, 1, 1, 2),
            doubleArrayOf(1.0, 2.0, -3.0, 4.0, 0.5, 1.5),
        )
        // A MatrixLike koblas does not know, so multiplication takes its entry-by-entry path.
        val foreign = object : MatrixLike {
            override val rows: Int get() = dense.rows
            override val cols: Int get() = dense.cols
            override fun get(i: Int, j: Int): Double = dense[i, j]
            override fun toArray(): Array<DoubleArray> = dense.toArray()
        }
        val denseX = DenseVector.of(doubleArrayOf(2.0, -1.0, 0.5))
        // The same vector with its middle entry dropped, so the two expectations differ in rows 2 and 3.
        val sparseX = SparseVector.of(3, intArrayOf(0, 2), doubleArrayOf(2.0, 0.5))
        val expectedDense = doubleArrayOf(3.0, 0.0, -10.0, 0.25)
        val expectedSparse = doubleArrayOf(3.0, 0.0, -6.0, 0.75)
        for ((name, matrix) in listOf(
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
            "negative rows" to { SparseMatrix.wrap(-1, 1, intArrayOf(0, 0), IntArray(0), DoubleArray(0)) },
            "negative cols" to { SparseMatrix.wrap(1, -1, intArrayOf(0), IntArray(0), DoubleArray(0)) },
            "short colPtr" to { SparseMatrix.wrap(2, 2, intArrayOf(0, 1), IntArray(0), DoubleArray(0)) },
            "misaligned values" to { SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 1), intArrayOf(0), DoubleArray(0)) },
            "nonzero colPtr head" to {
                SparseMatrix.wrap(2, 2, intArrayOf(1, 1, 1), intArrayOf(0), doubleArrayOf(1.0))
            },
            "colPtr tail is not nnz" to {
                SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 1), intArrayOf(0), doubleArrayOf(1.0, 2.0))
            },
            "colPtr not monotonic" to {
                SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 1), intArrayOf(0), doubleArrayOf(1.0))
            },
            "row out of range" to {
                SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 1), intArrayOf(5), doubleArrayOf(1.0))
            },
            "rows not ascending" to {
                SparseMatrix.wrap(2, 1, intArrayOf(0, 2), intArrayOf(1, 0), doubleArrayOf(1.0, 2.0))
            },
            "rows repeated" to {
                SparseMatrix.wrap(2, 1, intArrayOf(0, 2), intArrayOf(1, 1), doubleArrayOf(1.0, 2.0))
            },
        )
        for ((name, build) in cases) {
            assertFailsWith<IllegalArgumentException>("$name should be rejected") { build() }
        }
    }

    @Test
    fun `the sparse vector constructor rejects each way the arrays can disagree`() {
        assertFailsWith<IllegalArgumentException>("misaligned") {
            SparseVector.wrap(3, intArrayOf(0, 1), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException>("negative index") {
            SparseVector.wrap(3, intArrayOf(-1), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException>("index past the end") {
            SparseVector.wrap(3, intArrayOf(3), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException>("not ascending") {
            SparseVector.wrap(3, intArrayOf(1, 0), doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<IllegalArgumentException>("repeated") {
            SparseVector.wrap(3, intArrayOf(1, 1), doubleArrayOf(1.0, 2.0))
        }
    }

    /** `of` repairs what `wrap` rejects: it sorts by index and sums the duplicates. */
    @Test
    fun `the sparse vector factory sorts and sums where wrap refuses`() {
        val v = SparseVector.of(4, intArrayOf(3, 0, 3), doubleArrayOf(1.0, 2.0, 0.5))
        assertEquals(listOf(0, 3), v.indices.toList())
        assertEquals(listOf(2.0, 1.5), v.values.toList())
        assertEquals(0.0, v[1], "an unstored position reads as zero")
        assertFailsWith<IllegalArgumentException> { SparseVector.of(4, intArrayOf(0), doubleArrayOf(1.0, 2.0)) }
    }
}
