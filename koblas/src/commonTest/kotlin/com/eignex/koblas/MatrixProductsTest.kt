package com.eignex.koblas

import com.eignex.koblas.sparse.ReferenceSparseBlas
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.fail

class MatrixProductsTest {

    /** A [Matrix] from outside the library, which has only [Matrix.get] to be read through. */
    private class ForeignMatrix(private val backing: DenseMatrix) : Matrix {
        override val rows: Int get() = backing.rows
        override val cols: Int get() = backing.cols
        override fun get(i: Int, j: Int): Double = backing[i, j]
        override fun toArray(): Array<DoubleArray> = backing.toArray()
    }

    /** A [Matrix] whose entries cannot be read at all, for the calls whose contract says they are not. */
    private class PoisonMatrix(override val rows: Int, override val cols: Int) : Matrix {
        override fun get(i: Int, j: Int): Double = fail("a matrix this call must not read was read at ($i, $j)")
        override fun toArray(): Array<DoubleArray> = fail("a matrix this call must not read was materialised")
    }

    @Test
    fun `generic dense product returns dense storage`() {
        val left: Matrix = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val right: Matrix = DenseMatrix.ofRows(arrayOf(doubleArrayOf(5.0), doubleArrayOf(6.0)))

        val result = left * right

        assertEquals(DenseMatrix::class, result::class)
        assertContentEquals(doubleArrayOf(17.0, 39.0), (result as DenseMatrix).values)
    }

    @Test
    fun `generic product supports transpose beta alias and reusable workspace`() {
        val backing = doubleArrayOf(1.0, 3.0, 2.0, 4.0)
        val left: Matrix = DenseMatrix.wrap(2, 2, backing)
        val right: Matrix = DenseMatrix.diagonal(2, 2.0)
        val destination = DenseMatrix.wrap(2, 2, backing)

        left.gemmInto(1.0, true, right, false, 0.5, destination, Workspace())

        assertContentEquals(doubleArrayOf(2.5, 5.5, 7.0, 10.0), destination.values)
    }

    @Test
    fun `fresh result storage follows the runtime storage pair`() {
        val dense: Matrix = DenseMatrix.diagonal(3, 2.0)
        val sparse: Matrix = SparseMatrix.ofColumns(3, 3, List(3) { j -> listOf(j to 3.0) })

        assertEquals(DenseMatrix::class, (dense * dense)::class, "dense by dense")
        assertEquals(DenseMatrix::class, (dense * sparse)::class, "dense by sparse")
        assertEquals(DenseMatrix::class, (sparse * dense)::class, "sparse by dense")
        assertEquals(SparseMatrix::class, (sparse * sparse)::class, "sparse by sparse")
        assertEquals(3, ((sparse * sparse) as SparseMatrix).nnz, "sparse result keeps CSC support")
    }

    @Test
    fun `every pairing agrees with the storage-specific call through variables typed as Matrix`() {
        val rng = Random(20260924)
        val denseLeft = randomMatrix(4, 3, rng)
        val denseRight = randomMatrix(3, 5, rng)
        val sparseLeft = sparseOf(4, 3, rng)
        val sparseRight = sparseOf(3, 5, rng)

        assertClose(
            koblas.gemm(denseLeft, denseRight),
            (denseLeft as Matrix).gemm(denseRight) as DenseMatrix,
            "dense by dense",
        )
        assertClose(
            koblas.gemm(sparseLeft, denseRight),
            (sparseLeft as Matrix).gemm(denseRight) as DenseMatrix,
            "sparse by dense",
        )
        val mirrored = DenseMatrix.zero(4, 5)
        koblas.gemm(1.0, sparseRight, false, denseLeft, false, 0.0, mirrored, right = true)
        assertClose(mirrored, (denseLeft as Matrix).gemm(sparseRight) as DenseMatrix, "dense by sparse")

        assertEquals(
            ReferenceSparseBlas.gemm(1.0, sparseLeft, false, sparseRight, false),
            (sparseLeft as Matrix).gemm(sparseRight) as SparseMatrix,
            "sparse by sparse",
        )
    }

    @Test
    fun `every transpose pair reaches the same result as the storage-specific call`() {
        val rng = Random(20260925)
        for (transpose in booleanArrayOf(false, true)) {
            for (transposeOther in booleanArrayOf(false, true)) {
                // op(S) is 4x3 and op(D) is 3x5, so the sparse operand is on the left of a 4x5 destination.
                val sparse = sparseOf(if (transpose) 3 else 4, if (transpose) 4 else 3, rng)
                val dense = randomMatrix(if (transposeOther) 5 else 3, if (transposeOther) 3 else 5, rng)
                val expected = randomMatrix(4, 5, rng)
                val actual = DenseMatrix.wrap(4, 5, expected.values.copyOf())

                koblas.gemm(0.75, sparse, transpose, dense, transposeOther, -0.5, expected)
                (sparse as Matrix).gemmInto(0.75, transpose, dense, transposeOther, -0.5, actual)

                assertClose(expected, actual, "sparse-dense transpose=$transpose other=$transposeOther")

                // The mirror: op(D) is 5x4 and op(S) is 4x3, so the sparse operand is on the right of 5x3.
                val leftDense = randomMatrix(if (transpose) 4 else 5, if (transpose) 5 else 4, rng)
                val rightSparse = sparseOf(if (transposeOther) 3 else 4, if (transposeOther) 4 else 3, rng)
                val mirroredExpected = randomMatrix(5, 3, rng)
                val mirroredActual = DenseMatrix.wrap(5, 3, mirroredExpected.values.copyOf())

                koblas.gemm(
                    0.75,
                    rightSparse,
                    transposeOther,
                    leftDense,
                    transpose,
                    -0.5,
                    mirroredExpected,
                    right = true,
                )
                (leftDense as Matrix).gemmInto(0.75, transpose, rightSparse, transposeOther, -0.5, mirroredActual)

                assertClose(
                    mirroredExpected,
                    mirroredActual,
                    "dense-sparse transpose=$transpose other=$transposeOther",
                )
            }
        }
    }

    @Test
    fun `a generic product stages aliases in either operand position`() {
        val backing = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
        val sparse = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), backing)
        val identity: Matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
        val destination = DenseMatrix.wrap(2, 2, backing)

        (sparse as Matrix).gemmInto(1.0, false, identity, false, 0.0, destination, Workspace())

        assertContentEquals(doubleArrayOf(2.0, 3.0, 5.0, 7.0), destination.values, "left operand alias")

        val denseBacking = doubleArrayOf(1.0, 0.0, 0.0, 1.0)
        val dense = DenseMatrix.wrap(2, 2, denseBacking)
        val denseDestination = DenseMatrix.wrap(2, 2, denseBacking)
        val scale: Matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 2.0)))

        (dense as Matrix).gemmInto(1.0, false, scale, false, 0.0, denseDestination, Workspace())

        assertContentEquals(doubleArrayOf(2.0, 0.0, 0.0, 2.0), denseDestination.values, "right operand alias")
    }

    @Test
    fun `a zero alpha never reads either operand`() {
        val destination = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        PoisonMatrix(2, 2).gemmInto(0.0, false, PoisonMatrix(2, 2), false, 2.0, destination)

        assertContentEquals(doubleArrayOf(2.0, 4.0, 6.0, 8.0), destination.values)
    }

    @Test
    fun `a zero alpha and a zero beta overwrite without reading anything`() {
        val destination = DenseMatrix.wrap(1, 1, doubleArrayOf(Double.NaN))

        PoisonMatrix(1, 1).gemmInto(0.0, false, DenseMatrix.diagonal(1) as Matrix, false, 0.0, destination)

        assertContentEquals(doubleArrayOf(0.0), destination.values)
    }

    @Test
    fun `an invalid shape is rejected before the destination is touched`() {
        val destination = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val before = destination.values.copyOf()
        val sparse: Matrix = SparseMatrix.ofColumns(2, 3, List(3) { emptyList() })

        assertFailsWith<DimensionMismatch> {
            sparse.gemmInto(1.0, false, DenseMatrix.diagonal(2) as Matrix, false, 0.0, destination)
        }
        assertFailsWith<DimensionMismatch> { sparse.gemm(DenseMatrix.diagonal(2)) }

        assertContentEquals(before, destination.values)
    }

    @Test
    fun `a matrix from outside the library is read through its logical accessor`() {
        val rng = Random(20260926)
        val backing = randomMatrix(3, 2, rng)
        val foreign: Matrix = ForeignMatrix(backing)
        val sparse = sparseOf(2, 4, rng)

        val expected = DenseMatrix.zero(3, 4)
        koblas.gemm(1.0, sparse, false, backing, false, 0.0, expected, right = true)
        val actual = DenseMatrix.zero(3, 4)
        foreign.gemmInto(1.0, false, sparse, false, 0.0, actual)

        assertClose(expected, actual, "foreign matrix beside a sparse operand")
        assertEquals(DenseMatrix::class, (foreign * (backing.transpose() as Matrix))::class)
    }

    private fun sparseOf(rows: Int, cols: Int, rng: Random): SparseMatrix = SparseMatrix.ofColumns(
        rows,
        cols,
        List(cols) { j ->
            (0 until rows).mapNotNull { i ->
                if ((i + j) % 2 == 0) i to rng.nextDouble(-1.0, 1.0) else null
            }
        },
    )
}
