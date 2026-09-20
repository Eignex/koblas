@file:Suppress("PropertyName") // math convention: single-letter matrices in tests

package com.eignex.koblas

import com.eignex.koblas.dense.MatrixStructure
import kotlin.random.Random
import kotlin.test.*

@Suppress("VariableNaming") // single-letter matrix and vector names track math conventions
class MatrixOpsTest {
    private fun dense(vararg values: Double) = DenseVector.of(values)

    @Test
    fun `the gemv overload computes A x`() {
        val A = DenseMatrix.ofRows(
            arrayOf(
                doubleArrayOf(1.0, 2.0),
                doubleArrayOf(3.0, 4.0),
                doubleArrayOf(5.0, 6.0),
            ),
        )
        val x = dense(1.0, -1.0)
        val expected = dense(1.0 * 1 + 2 * -1, 3.0 * 1 + 4 * -1, 5.0 * 1 + 6 * -1)
        assertEquals(expected, A * x)
    }

    @Test
    fun `matrix vector operations match hand results over every dense spacing`() {
        val rows = 4
        val entries = arrayOf(
            doubleArrayOf(1.0, 0.0, 2.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(-3.0, 4.0, 0.0),
            doubleArrayOf(0.0, 0.5, 1.5),
        )
        val dense = DenseMatrix.ofRows(entries)
        // Adjacent and every second entry: the spacing is all a vendor is told apart from the pointer.
        val vectors = listOf<Pair<DenseVector, DoubleArray>>(
            DenseVector.of(doubleArrayOf(2.0, -1.0, 0.5)) to doubleArrayOf(3.0, 0.0, -10.0, 0.25),
            StridedVector(doubleArrayOf(2.0, 99.0, -1.0, 99.0, 0.5), 0, 3, 2) to
                doubleArrayOf(3.0, 0.0, -10.0, 0.25),
        )
        for ((x, expected) in vectors) {
            assertClose(expected, (dense * x).values, "product ${x::class.simpleName}")
            val out = DoubleArray(rows)
            dense.gemvInto(x, out)
            assertClose(expected, out, "gemvInto ${x::class.simpleName}")
        }
    }

    @Test
    fun `gemvInto applies alpha and beta like dgemv`() {
        val A = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val xs = listOf(dense(1.0, -1.0), StridedVector(doubleArrayOf(1.0, 9.0, -1.0), 0, 2, 2))
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
    fun `gemvInto transposes the matrix like the sparse extension of the same name`() {
        val rng = Random(20260920)
        val A = randomMatrix(4, 3, rng)
        val x = randomVector(4, rng)
        val expected = randomVector(3, rng)
        val actual = expected.copyOf()

        koblas.gemv(0.75, A, x, -0.5, expected, transpose = true)
        A.gemvInto(0.75, DenseVector.wrap(x), -0.5, actual, transpose = true)

        assertClose(expected, actual, "transposed gemvInto")
        val fresh = DoubleArray(3)
        A.gemvInto(DenseVector.wrap(x), fresh, transpose = true)
        koblas.gemv(1.0, A, x, 0.0, expected, transpose = true)
        assertClose(expected, fresh, "transposed gemvInto shorthand")
    }

    @Test
    fun `symvInto reads only the selected triangle`() {
        val n = 5
        val rng = Random(13)
        val full = randomMatrix(n, n, rng)
        for (j in 0 until n) for (i in 0 until j) full[i, j] = full[j, i]
        val values = randomVector(n, rng)
        val xs = listOf(DenseVector.of(values), StridedVector(values, 0, n, 1))
        for (lower in listOf(true, false)) {
            // Only the named triangle may be read, so the other one holds NaN: any read poisons the result.
            val poisoned = DenseMatrix.wrap(n, n, full.values.copyOf())
            for (j in 0 until n) {
                for (i in 0 until n) {
                    val unread = if (lower) i < j else i > j
                    if (unread) poisoned[i, j] = Double.NaN
                }
            }
            for (x in xs) {
                val out = DoubleArray(n)
                poisoned.symvInto(x, out, lower)
                assertClose((full * x).values, out, "symvInto lower=$lower ${x::class.simpleName}")
            }
        }
    }

    @Test
    fun `symvInto applies alpha and beta like dsymv`() {
        val A = DenseMatrix.ofRows(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        for (x in listOf(dense(1.0, -1.0), StridedVector(doubleArrayOf(1.0, 9.0, -1.0), 0, 2, 2))) {
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
        // A caller can keep its matrix with syr, which writes one triangle, and still take products with it.
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
        assertClose((viaGer * DenseVector.of(probe)).values, fromSyr, "syr-maintained symv")
    }

    @Test
    fun `the matvec destinations reject mismatched shapes`() {
        val A = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val x = dense(1.0, -1.0)
        assertFailsWith<DimensionMismatch> { A.gemvInto(dense(1.0, 2.0, 3.0), DoubleArray(2)) }
        assertFailsWith<DimensionMismatch> { A.gemvInto(x, DoubleArray(3)) }
        assertFailsWith<DimensionMismatch> { A.symvInto(x, DoubleArray(3)) }
        assertFailsWith<DimensionMismatch> {
            DenseMatrix.zero(2, 3).symvInto(dense(1.0, 2.0, 3.0), DoubleArray(2))
        }
    }

    @Test
    fun `matvec adapters snapshot vector aliases before writing`() {
        val matrix = DenseMatrix.diagonal(3)
        for (reversed in booleanArrayOf(false, true)) {
            for (symmetric in booleanArrayOf(false, true)) {
                val destination = doubleArrayOf(1.0, 2.0, 3.0)
                val original = destination.copyOf()
                // A call that scaled the destination before reading its input would see the scaled values.
                val source =
                    if (reversed) StridedVector(destination, 2, 3, -1) else StridedVector(destination, 0, 3, 1)
                val expected = DoubleArray(3) { source[it] + 0.5 * original[it] }

                if (symmetric) {
                    matrix.symvInto(1.0, source, 0.5, destination)
                } else {
                    matrix.gemvInto(1.0, source, 0.5, destination)
                }

                assertContentEquals(expected, destination)
            }
        }
    }

    @Test
    fun `gemvInto snapshots matrix aliases before writing`() {
        val destination = doubleArrayOf(1.0, 2.0, 3.0)
        val matrix = DenseMatrix.wrap(3, 1, destination)

        matrix.gemvInto(1.0, dense(2.0), 0.5, destination)

        assertContentEquals(doubleArrayOf(2.5, 5.0, 7.5), destination)
    }

    @Test
    fun `symvInto snapshots its matrix alias before writing`() {
        val destination = doubleArrayOf(2.0)
        val matrix = DenseMatrix.wrap(1, 1, destination)

        matrix.symvInto(1.0, dense(3.0), 0.5, destination)

        assertContentEquals(doubleArrayOf(7.0), destination)
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
    fun `ger contributes nothing where an operand is zero`() {
        val a = DenseMatrix.diagonal(3, 0.0)

        a.ger(1.0, dense(0.0, 2.0, 0.0), dense(1.0, 1.0, 1.0))

        for (i in 0 until 3) {
            for (j in 0 until 3) assertEquals(if (i == 1) 2.0 else 0.0, a[i, j], 1e-12, "a[$i,$j]")
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
    fun `matrix operations reject mismatched shapes`() {
        assertFailsWith<DimensionMismatch> {
            DenseMatrix(2, 2).ger(1.0, dense(1.0, 2.0, 3.0), dense(1.0, 2.0))
        }
        assertFailsWith<DimensionMismatch> { DenseMatrix(2, 3) * dense(1.0, 2.0) }
    }

    @Test
    fun `masking to a lower triangle clears above the diagonal and keeps the rest`() {
        val M = DenseMatrix.ofRows(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
                doubleArrayOf(7.0, 8.0, 9.0),
            ),
        )
        M.maskTo(MatrixStructure.TriangularLower)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                val expected = if (i < j) 0.0 else (i * 3 + j + 1).toDouble()
                assertEquals(expected, M[i, j], "($i,$j)")
            }
        }
        // Column-major, so the cleared entries are the leading run of each column after the first.
        assertClose(
            doubleArrayOf(1.0, 4.0, 7.0, 0.0, 5.0, 8.0, 0.0, 0.0, 9.0),
            M.values,
            "backing buffer",
        )
    }

    @Test
    fun `masking a wide matrix zeroes whole columns past the last row`() {
        val wide = DenseMatrix.wrap(2, 4, DoubleArray(8) { it + 1.0 })
        wide.maskTo(MatrixStructure.TriangularLower)
        for (j in 0 until 4) {
            for (i in 0 until 2) {
                val expected = if (i < j) 0.0 else (j * 2 + i + 1).toDouble()
                assertEquals(expected, wide[i, j], "($i,$j)")
            }
        }
    }

    @Test
    fun `masking a tall matrix keeps it intact below the diagonal`() {
        val tall = DenseMatrix.wrap(4, 2, DoubleArray(8) { it + 1.0 })
        val before = tall.values.copyOf()
        tall.maskTo(MatrixStructure.TriangularLower)
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
        val a = DenseMatrix.ofRows(
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
    fun `gemvInto honours beta on a matrix with no columns`() {
        val dense = DenseMatrix.zero(3, 0)
        val destination = doubleArrayOf(Double.NaN, Double.NaN, Double.NaN)

        dense.gemvInto(1.0, DenseVector.zero(0), 0.0, destination)

        assertContentEquals(doubleArrayOf(0.0, 0.0, 0.0), destination)
    }
}
