@file:Suppress("PropertyName") // math convention: single-letter matrices in tests

package com.eignex.koblas

import kotlin.random.Random
import kotlin.test.*

@Suppress("VariableNaming") // single-letter matrix and vector names track math conventions
class MatrixOpsTest {
    private fun dense(vararg values: Double) = DenseVector.of(values)
    private fun sparse(size: Int, vararg entries: Pair<Int, Double>) = SparseVector.of(
        size,
        entries.map { it.first }.toIntArray(),
        entries.map { it.second }.toDoubleArray(),
    )

    @Test
    fun `the gemv overload computes A x for dense and sparse x`() {
        val A = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0),
                doubleArrayOf(3.0, 4.0),
                doubleArrayOf(5.0, 6.0),
            ),
        )
        val xDense = dense(1.0, -1.0)
        val xSparse = sparse(2, 0 to 1.0, 1 to -1.0)
        val expected = dense(1.0 * 1 + 2 * -1, 3.0 * 1 + 4 * -1, 5.0 * 1 + 6 * -1)
        assertEquals(expected, A * xDense)
        assertEquals(expected, A * xSparse)
    }

    @Test
    fun `sparse and dense gemv agree on a random example`() {
        val rng = Random(7)
        val n = 8
        val A = randomMatrix(n, n, rng)
        val nz = (0 until n).filter { rng.nextBoolean() }
        val xv = DoubleArray(n)
        for (i in nz) xv[i] = rng.nextDouble(-1.0, 1.0)
        val xSparse = SparseVector.of(n, nz.toIntArray(), nz.map { xv[it] }.toDoubleArray())
        assertClose((A * DenseVector.of(xv)).data, (A * xSparse).data, "matrix vector dense vs sparse")
    }

    @Test
    fun `gemvInto agrees with the allocating product over every storage pairing`() {
        val rng = Random(11)
        val n = 6
        val dense = randomMatrix(n, n, rng)
        val sparseMatrix = SparseMatrix.ofTriplets(
            rows = n,
            cols = n,
            rowIdx = IntArray(n) { it },
            colIdx = IntArray(n) { it },
            values = DoubleArray(n) { it + 1.0 },
        )
        val values = randomVector(n, rng)
        val xs = listOf<VectorLike>(
            DenseVector.of(values),
            SparseVector.of(n, intArrayOf(1, 4), doubleArrayOf(values[1], values[4])),
            StridedVectorView(DoubleArray(2 * n) { values[it / 2] }, 0, n, 2),
            ForeignRampVector(n),
        )
        for (A in listOf<MatrixLike>(dense, sparseMatrix, ForeignSpdMatrix(n))) {
            for (x in xs) {
                val out = DoubleArray(n)
                A.gemvInto(x, out)
                assertClose((A * x).data, out, "gemvInto ${A::class.simpleName} ${x::class.simpleName}")
            }
        }
    }

    @Test
    fun `gemvInto applies alpha and beta like dgemv`() {
        val A = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val xs = listOf<VectorLike>(dense(1.0, -1.0), sparse(2, 0 to 1.0, 1 to -1.0))
        for (x in xs) {
            val context = x::class.simpleName
            // beta == 0 must overwrite rather than accumulate, so a poisoned destination stays clean.
            val fresh = doubleArrayOf(Double.NaN, Double.NaN)
            A.gemvInto(2.0, x, 0.0, fresh)
            assertClose(doubleArrayOf(-2.0, -2.0), fresh, "alpha scaled, beta zero, $context")

            val accumulated = doubleArrayOf(1.0, 1.0)
            A.gemvInto(1.0, x, 2.0, accumulated)
            assertClose(doubleArrayOf(1.0, 1.0), accumulated, "alpha one, beta two, $context")

            val scaledOnly = doubleArrayOf(1.0, -2.0)
            A.gemvInto(0.0, x, 3.0, scaledOnly)
            assertClose(doubleArrayOf(3.0, -6.0), scaledOnly, "alpha zero leaves beta y, $context")
        }
    }

    @Test
    fun `symvInto reads only the selected triangle`() {
        val n = 5
        val rng = Random(13)
        val full = randomMatrix(n, n, rng)
        for (j in 0 until n) for (i in 0 until j) full[i, j] = full[j, i]
        val values = randomVector(n, rng)
        val xs = listOf<VectorLike>(
            DenseVector.of(values),
            SparseVector.of(n, intArrayOf(0, 3), doubleArrayOf(values[0], values[3])),
        )
        for (lower in listOf(true, false)) {
            // Only the named triangle may be read, so the other one holds NaN: any read poisons the result.
            val poisoned = DenseMatrix.wrap(n, n, full.data.copyOf())
            for (j in 0 until n) {
                for (i in 0 until n) {
                    val unread = if (lower) i < j else i > j
                    if (unread) poisoned[i, j] = Double.NaN
                }
            }
            for (x in xs) {
                val out = DoubleArray(n)
                poisoned.symvInto(x, out, lower)
                assertClose((full * x).data, out, "symvInto lower=$lower ${x::class.simpleName}")
            }
        }
    }

    @Test
    fun `symvInto applies alpha and beta like dsymv`() {
        val A = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        for (x in listOf<VectorLike>(dense(1.0, -1.0), sparse(2, 0 to 1.0, 1 to -1.0))) {
            val context = x::class.simpleName
            val fresh = doubleArrayOf(Double.NaN, Double.NaN)
            A.symvInto(2.0, x, 0.0, fresh)
            assertClose(doubleArrayOf(2.0, -4.0), fresh, "alpha scaled, beta zero, $context")

            val accumulated = doubleArrayOf(1.0, 1.0)
            A.symvInto(1.0, x, 2.0, accumulated)
            assertClose(doubleArrayOf(3.0, 0.0), accumulated, "alpha one, beta two, $context")

            val scaledOnly = doubleArrayOf(1.0, -2.0)
            A.symvInto(0.0, x, 3.0, scaledOnly)
            assertClose(doubleArrayOf(3.0, -6.0), scaledOnly, "alpha zero leaves beta y, $context")
        }
    }

    @Test
    fun `symvInto over one triangle agrees with the full ger sweep`() {
        // The point of the routine: a caller can keep its matrix with syr, which writes one triangle, and
        // still take products against it.
        val n = 4
        val rng = Random(17)
        val x = randomVector(n, rng)
        val viaSyr = DenseMatrix.zero(n, n)
        viaSyr.syr(1.5, DenseVector.of(x))
        val viaGer = DenseMatrix.zero(n, n)
        viaGer.ger(1.5, DenseVector.of(x), DenseVector.of(x))
        val probe = randomVector(n, rng)
        val fromSyr = DoubleArray(n)
        viaSyr.symvInto(DenseVector.of(probe), fromSyr)
        assertClose((viaGer * DenseVector.of(probe)).data, fromSyr, "syr-maintained symv")
    }

    @Test
    fun `the matvec destinations reject aliasing and mismatched shapes`() {
        val A = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val x = dense(1.0, -1.0)
        assertFailsWith<IllegalArgumentException> { A.gemvInto(x, x.data) }
        assertFailsWith<IllegalArgumentException> { A.symvInto(x, x.data) }
        val square = DenseMatrix.of(arrayOf(doubleArrayOf(1.0)))
        assertFailsWith<IllegalArgumentException> { square.gemvInto(dense(1.0), square.data) }
        assertFailsWith<DimensionMismatch> { A.gemvInto(dense(1.0, 2.0, 3.0), DoubleArray(2)) }
        assertFailsWith<DimensionMismatch> { A.gemvInto(x, DoubleArray(3)) }
        assertFailsWith<DimensionMismatch> { A.symvInto(x, DoubleArray(3)) }
        assertFailsWith<DimensionMismatch> {
            DenseMatrix.zero(2, 3).symvInto(dense(1.0, 2.0, 3.0), DoubleArray(2))
        }
    }

    @Test
    fun `ger updates a matrix with alpha x y_transpose`() {
        val M = DenseMatrix.diagonal(2, 1.0)
        M.ger(0.5, dense(1.0, 2.0), dense(3.0, 4.0))
        assertEquals(1.0 + 0.5 * 3, M[0, 0], 1e-12)
        assertEquals(0.5 * 4, M[0, 1], 1e-12)
        assertEquals(0.5 * 6, M[1, 0], 1e-12)
        assertEquals(1.0 + 0.5 * 8, M[1, 1], 1e-12)
    }

    @Test
    fun `ger with sparse operands only touches nonzero rows and cols`() {
        val M = DenseMatrix.diagonal(3, 0.0)
        M.ger(1.0, sparse(3, 1 to 2.0), sparse(3, 0 to 3.0, 2 to 4.0))
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                val expected = when {
                    i == 1 && j == 0 -> 6.0
                    i == 1 && j == 2 -> 8.0
                    else -> 0.0
                }
                assertEquals(expected, M[i, j], 1e-12, "M[$i,$j]")
            }
        }
    }

    @Test
    fun `alpha zero makes axpy and ger no-ops`() {
        val y = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))
        y.axpy(0.0, DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0)))
        assertTrue(y.toDoubleArray().contentEquals(doubleArrayOf(1.0, 2.0, 3.0)))
        val M = DenseMatrix.diagonal(2, 1.0)
        M.ger(0.0, DenseVector.of(doubleArrayOf(1.0, 1.0)), DenseVector.of(doubleArrayOf(1.0, 1.0)))
        for (i in 0 until 2) {
            for (j in 0 until 2) assertEquals(if (i == j) 1.0 else 0.0, M[i, j])
        }
    }

    @Test
    fun `ger skips zero entries on both carriers`() {
        val dense = DenseMatrix.diagonal(3, 0.0)
        dense.ger(1.0, dense(0.0, 2.0, 0.0), dense(1.0, 1.0, 1.0))
        for (i in 0 until 3) {
            for (j in 0 until 3) assertEquals(if (i == 1) 2.0 else 0.0, dense[i, j], 1e-12, "dense[$i,$j]")
        }
        val sparse = DenseMatrix.diagonal(3, 0.0)
        sparse.ger(1.0, sparse(3, 0 to 0.0, 1 to 1.0), sparse(3, 2 to 5.0))
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(if (i == 1 && j == 2) 5.0 else 0.0, sparse[i, j], "sparse[$i,$j]")
            }
        }
    }

    @Test
    fun `syr and syr2 match the equivalent ger sweeps`() {
        val rng = Random(20260807)
        val n = 6
        val x = DenseVector.of(randomVector(n, rng))
        val y = DenseVector.of(randomVector(n, rng))

        val viaSyr = DenseMatrix(n, n)
        viaSyr.syr(1.5, x)
        val viaGer = DenseMatrix(n, n)
        viaGer.ger(1.5, x, x)
        assertLowerTriangleClose(viaGer, viaSyr, "syr against ger")

        val viaSyr2 = DenseMatrix(n, n)
        viaSyr2.syr2(-0.75, x, y)
        val viaGer2 = DenseMatrix(n, n)
        viaGer2.ger(-0.75, x, y)
        viaGer2.ger(-0.75, y, x)
        assertLowerTriangleClose(viaGer2, viaSyr2, "syr2 against two gers")
    }

    /**
     * A sparse operand updates the outer product of its stored positions and leaves the rest of A alone, so
     * it has to land exactly where the same vector densified does.
     */
    @Test
    fun `syr and syr2 over sparse operands match their dense equivalents`() {
        val n = 8
        val xSparse = sparse(n, 1 to 2.0, 4 to -3.0, 7 to 0.5)
        val ySparse = sparse(n, 0 to 1.5, 4 to 2.5)
        val xDense = DenseVector.of(xSparse.toDoubleArray())
        val yDense = DenseVector.of(ySparse.toDoubleArray())

        for (lower in booleanArrayOf(true, false)) {
            val sparseSyr = DenseMatrix(n, n)
            sparseSyr.syr(1.5, xSparse, lower)
            val denseSyr = DenseMatrix(n, n)
            denseSyr.syr(1.5, xDense, lower)
            assertClose(denseSyr, sparseSyr, "syr lower=$lower")

            // Mixed storages as well, since only one operand of syr2 need be sparse.
            for (pair in listOf(xSparse to ySparse, xSparse to yDense, xDense to ySparse)) {
                val sparseSyr2 = DenseMatrix(n, n)
                sparseSyr2.syr2(-0.75, pair.first, pair.second, lower)
                val denseSyr2 = DenseMatrix(n, n)
                denseSyr2.syr2(-0.75, xDense, yDense, lower)
                assertClose(denseSyr2, sparseSyr2, "syr2 lower=$lower")
            }
        }
    }

    /** A sparse vector may store a zero, which contributes nothing and must not be mistaken for a position. */
    @Test
    fun `syr over a sparse operand storing a zero matches the dense update`() {
        val n = 5
        val stored = SparseVector.of(n, intArrayOf(0, 2, 3), doubleArrayOf(2.0, 0.0, -1.0))
        val densified = DenseVector.of(stored.toDoubleArray())

        val fromSparse = DenseMatrix(n, n)
        fromSparse.syr(1.0, stored)
        val fromDense = DenseMatrix(n, n)
        fromDense.syr(1.0, densified)

        assertClose(fromDense, fromSparse, "syr over a stored zero")
    }

    @Test
    fun `the symmetric updates write only the selected triangle`() {
        val rng = Random(20260808)
        val n = 5
        val x = DenseVector.of(randomVector(n, rng))

        val lower = DenseMatrix(n, n)
        lower.syr(1.0, x, lower = true)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i < j) assertEquals(0.0, lower[i, j], "syr wrote the upper triangle at [$i,$j]")
                if (i >= j) assertEquals(x[i] * x[j], lower[i, j], 1e-12, "[$i,$j]")
            }
        }
    }

    private fun assertLowerTriangleClose(expected: DenseMatrix, actual: DenseMatrix, context: String) {
        for (j in 0 until expected.cols) {
            for (i in j until expected.rows) assertEquals(expected[i, j], actual[i, j], 1e-12, context)
        }
    }

    @Test
    fun `the ops reject mismatched sizes and shapes`() {
        assertFailsWith<IllegalArgumentException> { dense(1.0) dot dense(1.0, 2.0) }
        assertFailsWith<IllegalArgumentException> { dense(1.0).axpy(1.0, dense(1.0, 2.0)) }
        assertFailsWith<IllegalArgumentException> {
            DenseMatrix(2, 2).ger(1.0, dense(1.0, 2.0, 3.0), dense(1.0, 2.0))
        }
        assertFailsWith<IllegalArgumentException> { DenseMatrix(2, 3) * dense(1.0, 2.0) }
    }

    @Test
    fun `zeroStrictUpper clears above the diagonal and keeps the rest`() {
        val M = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
                doubleArrayOf(7.0, 8.0, 9.0),
            ),
        )
        M.zeroStrictUpper()
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                val expected = if (i < j) 0.0 else (i * 3 + j + 1).toDouble()
                assertEquals(expected, M[i, j], "($i,$j)")
            }
        }
        // Column-major, so the cleared entries are the leading run of each column after the first.
        assertClose(
            doubleArrayOf(1.0, 4.0, 7.0, 0.0, 5.0, 8.0, 0.0, 0.0, 9.0),
            M.data,
            "backing buffer",
        )
    }

    @Test
    fun `zeroStrictUpper zeroes whole columns past the last row`() {
        val wide = DenseMatrix.wrap(2, 4, DoubleArray(8) { it + 1.0 })
        wide.zeroStrictUpper()
        for (j in 0 until 4) {
            for (i in 0 until 2) {
                val expected = if (i < j) 0.0 else (j * 2 + i + 1).toDouble()
                assertEquals(expected, wide[i, j], "($i,$j)")
            }
        }
    }

    @Test
    fun `zeroStrictUpper keeps a tall matrix intact below the diagonal`() {
        val tall = DenseMatrix.wrap(4, 2, DoubleArray(8) { it + 1.0 })
        val before = tall.data.copyOf()
        tall.zeroStrictUpper()
        // Only (0,1) sits above the diagonal here, so every other entry survives.
        assertEquals(0.0, tall[0, 1], "(0,1)")
        for (j in 0 until 2) {
            for (i in 0 until 4) {
                if (i < j) continue
                assertEquals(before[i + j * 4], tall[i, j], "($i,$j)")
            }
        }
    }

    @Test
    fun `transpose round-trips and maps entries`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
            ),
        )
        val t = a.transpose()
        assertEquals(3, t.rows)
        assertEquals(2, t.cols)
        for (i in 0 until a.rows) for (j in 0 until a.cols) assertEquals(a[i, j], t[j, i])
        assertEquals(a, t.transpose())
        assertEquals(DenseMatrix(0, 0), DenseMatrix(0, 0).transpose())
        assertEquals(0, DenseMatrix(0, 5).transpose().cols)
        assertEquals(5, DenseMatrix(0, 5).transpose().rows)
    }

    @Test
    fun `transpose agrees with the gemm transpose flag`() {
        val rng = Random(20260804)
        val a = randomMatrix(4, 6, rng)
        val b = randomMatrix(4, 3, rng)
        val viaMaterialized = a.transpose() * b
        val viaFlag = DenseMatrix(6, 3)
        koblas.gemm(1.0, a, true, b, false, 0.0, viaFlag)
        assertClose(viaMaterialized, viaFlag, "transpose flag vs materialized")
    }

    @Test
    fun `sparse against sparse dot matches the dense answer over merge shapes`() {
        val n = 8
        val patterns = listOf(
            intArrayOf(0, 2, 4, 6) to intArrayOf(1, 3, 5, 7), // disjoint, fully interleaved
            intArrayOf(0, 1, 2) to intArrayOf(0, 1, 2), // identical
            intArrayOf(0, 7) to intArrayOf(3, 4), // disjoint, nested inside the first span
            intArrayOf(0, 1, 2, 3) to intArrayOf(3), // one side exhausts early
            intArrayOf(5) to intArrayOf(0, 1, 2, 3, 4), // the other side does
            IntArray(0) to intArrayOf(0, 4), // empty against non-empty
            IntArray(0) to IntArray(0),
        )
        for ((ia, ib) in patterns) {
            val a = SparseVector.of(n, ia, DoubleArray(ia.size) { it + 1.5 })
            val b = SparseVector.of(n, ib, DoubleArray(ib.size) { it + 2.5 })
            val expected = DenseVector.of(a.toDoubleArray()) dot DenseVector.of(b.toDoubleArray())
            assertEquals(expected, a dot b, "pattern ${ia.toList()} vs ${ib.toList()}")
            assertEquals(expected, b dot a, "dot should be symmetric for ${ia.toList()} vs ${ib.toList()}")
        }
    }

    @Test
    fun `mixed sparse and dense dot agrees in both operand orders`() {
        val sparse = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))
        val dense = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expected = 2.0 * 2.0 + -3.0 * 5.0
        assertEquals(expected, sparse dot dense)
        assertEquals(expected, dense dot sparse)
    }

    @Test
    fun `gemvInto honours beta on a matrix with no columns`() {
        // The seams quick-return on a zero-extent operand before scaling, so the dense fast path used to
        // leave the destination untouched while a generic MatrixLike of the same shape returned zeros.
        val dense = DenseMatrix.zero(3, 0)
        val destination = doubleArrayOf(Double.NaN, Double.NaN, Double.NaN)

        dense.gemvInto(1.0, DenseVector(DoubleArray(0)), 0.0, destination)

        assertContentEquals(doubleArrayOf(0.0, 0.0, 0.0), destination)
    }

    @Test
    fun `gemvInto scales by beta on a sparse matrix with no columns`() {
        val sparse = SparseMatrix.ofColumns(3, 0, emptyList())
        val destination = doubleArrayOf(2.0, 4.0, 6.0)

        sparse.gemvInto(1.0, DenseVector(DoubleArray(0)), 0.5, destination)

        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0), destination)
    }
}
