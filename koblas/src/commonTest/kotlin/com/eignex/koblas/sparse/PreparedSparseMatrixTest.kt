package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.PreparedSparseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.gemm
import com.eignex.koblas.gemmInto
import com.eignex.koblas.koblas
import com.eignex.koblas.prepare
import com.eignex.koblas.times
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PreparedSparseMatrixTest {
    private fun matrix(): SparseMatrix = SparseMatrix.ofColumns(
        3,
        2,
        listOf(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0)),
    )

    @Test
    fun `a prepared matrix is an immutable snapshot`() {
        val source = matrix()
        val prepared = koblas.prepare(source)
        source.values.fill(100.0)

        val actual = DoubleArray(3)
        prepared.gemvInto(1.0, doubleArrayOf(4.0, 5.0), 0.0, actual)

        assertContentEquals(doubleArrayOf(8.0, 15.0, -4.0), actual)
    }

    // Deriving the opposite orientation is the other way this could go, and the calibration measured it
    // slower, so the prepared route is the one-shot route and asking for it builds nothing.
    @Test
    fun `a prepared transposed product reports the schedule it actually runs`() {
        val source = SparseMatrix.ofColumns(
            3,
            2,
            listOf(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0)),
        )
        val prepared = koblas.prepare(source)
        // A transposed dense operand as well, so the reduction has its right-hand sides adjacent and reaches
        // a panel rather than being written out by the traversal.
        val call = SparseCall(
            source,
            0.875,
            -0.25,
            destinationElements = 2 * 16,
            depth = 3,
            rightHandSides = 16,
            transposeSparse = true,
            transposeDense = true,
        )

        val oneShot = koblas.routeOf(SparseMatrixOperation.GemmDense, call)
        val reused = prepared.routeOf(SparseMatrixOperation.GemmDense, call)

        assertTrue(
            oneShot.components.any { it.endsWith("/sparse-rhs-gather") },
            "a transposed one-shot product did not reduce its columns: $oneShot",
        )
        assertEquals(oneShot.components, reused.components, "a prepared product named another traversal")
        assertEquals(oneShot.executionGroup, reused.executionGroup)
        assertFalse(prepared.orientationDerived, "a prepared sparse-dense product derived an orientation")
    }

    @Test
    fun `a prepared transposed gemv keeps the orientation of the snapshot`() {
        val source = matrix()
        val prepared = koblas.prepare(source)
        val call = SparseCall(
            source,
            0.875,
            -0.25,
            destinationElements = 2,
            depth = 3,
            transposeSparse = true,
        )

        val route = prepared.routeOf(SparseMatrixOperation.GemvTransposed, call)

        assertFalse(prepared.orientationDerived, "a prepared transposed gemv derived an orientation")
        assertEquals(
            koblas.routeOf(SparseMatrixOperation.GemvTransposed, call).components,
            route.components,
            "a prepared transposed gemv reported something other than the call it makes",
        )
    }

    // The second sparse operand decides the schedule and the call facts do not carry it.
    @Test
    fun `a prepared transposed sparse product reports that its schedule is undecided`() {
        val prepared = matrix().prepare()

        val route = prepared.routeOf(
            SparseMatrixOperation.GemmSparse,
            SparseCall(matrix(), 1.0, transposeSparse = true, depth = 3),
        )

        assertEquals(false, route.resolved, route.toString())
        assertTrue(route.reason.orEmpty().contains("do not carry"), route.toString())
        assertFalse(prepared.orientationDerived, "an undecided route derived an orientation anyway")
    }

    /** A fact the call's own facts leave open stays open against the snapshot as well. */
    @Test
    fun `a prepared route keeps an unresolved right hand side count`() {
        val source = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0)))
        val prepared = koblas.prepare(source)
        val call = SparseCall(source, 0.875, -0.25, destinationElements = 8, depth = 3, transposeSparse = true)

        val route = prepared.routeOf(SparseMatrixOperation.GemmDense, call)

        assertEquals(koblas.routeOf(SparseMatrixOperation.GemmDense, call).resolved, route.resolved)
        assertEquals(false, route.resolved, route.toString())
        assertFalse(prepared.orientationDerived, "an unresolved route derived an orientation anyway")
    }

    // The extent still affects scaling and no-work attribution, so each route is compared with the same
    // supplied facts.
    @Test
    fun `a prepared transposed product names the one shot traversal whether or not its extent is given`() {
        val source = matrix()
        val prepared = koblas.prepare(source)
        val withoutExtent = SparseCall(source, 0.875, -0.25, depth = 3, rightHandSides = 8, transposeSparse = true)
        val withExtent = SparseCall(
            source,
            0.875,
            -0.25,
            destinationElements = 2 * 8,
            depth = 3,
            rightHandSides = 8,
            transposeSparse = true,
        )

        val open = prepared.routeOf(SparseMatrixOperation.GemmDense, withoutExtent)
        val settled = prepared.routeOf(SparseMatrixOperation.GemmDense, withExtent)

        assertEquals(
            koblas.routeOf(SparseMatrixOperation.GemmDense, withoutExtent).toString(),
            open.toString(),
        )
        assertEquals(
            koblas.routeOf(SparseMatrixOperation.GemmDense, withExtent).toString(),
            settled.toString(),
        )
        assertEquals(true, settled.resolved, settled.toString())
        assertFalse(prepared.orientationDerived, "a prepared sparse-dense route derived an orientation")
    }

    @Test
    fun `a prepared transposed product with an empty destination orients nothing`() {
        val prepared = matrix().prepare()

        val route = prepared.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(
                matrix(),
                0.875,
                -0.25,
                destinationElements = 0,
                depth = 3,
                rightHandSides = 8,
                transposeSparse = true,
            ),
        )

        assertEquals(RouteKind.NoWork, route.kind, route.toString())
        assertTrue(route.resolved, "a call with nothing to do has nothing undecided about it")
        assertFalse(prepared.orientationDerived, "an empty destination derived an orientation")
    }

    @Test
    fun `an unsettled prepared route keeps the facts its call does carry`() {
        val prepared = matrix().prepare()
        val destination = 3 * 4

        val scaled = prepared.routeOf(
            SparseMatrixOperation.GemmSparseDense,
            SparseCall(
                matrix(),
                1.0,
                -0.25,
                destinationElements = destination,
                depth = 3,
                transposeSparse = true,
            ),
        )
        val empty = prepared.routeOf(
            SparseMatrixOperation.GemmSparseDense,
            SparseCall(matrix(), 1.0, -0.25, destinationElements = 0, depth = 3, transposeSparse = true),
        )

        assertEquals(false, scaled.resolved, scaled.toString())
        assertTrue(scaled.components.any { it.endsWith("/scale") }, scaled.toString())
        assertEquals(RouteKind.NoWork, empty.kind, "a destination with no elements is still no work")
        assertTrue(empty.resolved, "a call with nothing to do has nothing undecided about it")
        assertFalse(prepared.orientationDerived, "an undecided route derived an orientation anyway")
    }

    /** A call with no work reads nothing, so asking about it derives no orientation either. */
    @Test
    fun `a prepared route for a call with no work derives no orientation`() {
        val prepared = matrix().prepare()

        prepared.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(
                matrix(),
                0.0,
                1.0,
                destinationElements = 8,
                depth = 3,
                rightHandSides = 4,
                transposeSparse = true,
            ),
        )

        assertFalse(prepared.orientationDerived, "a route for a zero multiplier derived an orientation")
    }

    @Test
    fun `the prepared shape reports the snapshot`() {
        val prepared = matrix().prepare()

        assertEquals(3, prepared.rows)
        assertEquals(2, prepared.cols)
        assertEquals(3, prepared.nnz)
    }

    @Test
    fun `a prepared matrix reads as the snapshot it copied`() {
        val source = matrix()

        val prepared = source.prepare()

        assertEquals(source[2, 0], prepared[2, 0])
        assertEquals(0.0, prepared[1, 0], "an absent position")
        assertContentEquals(source.toArray()[1], prepared.toArray()[1])
    }

    @Test
    fun `a prepared operand on the right of a dense one agrees with the one-shot call`() {
        val source = matrix()
        val prepared = source.prepare()
        val dense = DenseMatrix.wrap(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expected = DenseMatrix.zero(2, 2)
        koblas.gemm(1.0, source, false, dense, false, 0.0, expected, right = true)

        val actual = DenseMatrix.zero(2, 2)
        dense.gemmInto(1.0, false, prepared, false, 0.0, actual)

        assertClose(expected, actual, "dense by prepared")
    }

    @Test
    fun `two prepared operands give the sparse product of their snapshots`() {
        val source = matrix()
        val b = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to -1.0), listOf(1 to 4.0)))

        val product = source.prepare() * b.prepare()

        assertEquals(koblas.gemm(source, b), product)
    }

    // Naming an engine to prepare with settles which kernels the products of that snapshot reach, so the
    // common product surface asks it rather than the default engine.
    @Test
    fun `a prepared product runs on the engine that prepared the snapshot`() {
        val recording = RecordingSparseBlas(koblas)
        val prepared = PreparedSparseMatrix(matrix(), recording)
        val destination = DenseMatrix.zero(3, 2)

        prepared.gemmInto(1.0, false, DenseMatrix.diagonal(2), false, 0.0, destination)

        assertEquals(1, recording.denseProducts)
    }

    @Test
    fun `all prepared products agree with the one-shot calls`() {
        val source = matrix()
        val prepared = koblas.prepare(source)
        val dense = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val expectedDense = koblas.gemm(source, dense)
        val actualDense = DenseMatrix.zero(3, 2)
        prepared.gemmInto(1.0, false, dense, false, 0.0, actualDense)
        assertClose(expectedDense, actualDense, "dense product")

        val right = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 2.0, 1 to -1.0)))
        assertEquals(koblas.gemm(source, right), prepared.gemm(right))

        val transposedDense = DenseMatrix.wrap(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expectedTransposed = DenseMatrix.zero(2, 2)
        koblas.gemm(1.0, source, true, transposedDense, true, 0.0, expectedTransposed, right = false)
        val actualTransposed = DenseMatrix.zero(2, 2)
        prepared.gemmInto(1.0, true, transposedDense, true, 0.0, actualTransposed)
        assertClose(expectedTransposed, actualTransposed, "full dense product")

        val sparseDense = DenseMatrix.zero(3, 1)
        prepared.gemmInto(2.0, false, right, false, 0.0, sparseDense)
        val expectedSparseDense = DenseMatrix.zero(3, 1)
        koblas.gemm(2.0, source, false, right, false, 0.0, expectedSparseDense)
        assertClose(expectedSparseDense, sparseDense, "direct sparse dense result")
    }

    @Test
    fun `a prepared transposed product agrees with the one-shot transpose`() {
        val source = matrix()
        val prepared = source.prepare()
        val b = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.5, 2 to -2.0), listOf(1 to 0.5)))

        // Twice, because the second call is the one that reuses the derived transpose rather than deriving it.
        repeat(2) {
            assertEquals(koblas.gemm(1.0, source, true, b, false), prepared.gemm(1.0, true, b, false))
        }
    }

    @Test
    fun `prepared symmetric products retain snapshot semantics`() {
        val source = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 3.0), listOf(1 to 5.0)))
        val prepared = koblas.prepare(source)
        source.values.fill(Double.NaN)
        val y = DoubleArray(2)
        prepared.symvInto(1.0, doubleArrayOf(7.0, 11.0), 0.0, y)
        assertContentEquals(doubleArrayOf(47.0, 76.0), y)

        val b = DenseMatrix.wrap(2, 1, doubleArrayOf(7.0, 11.0))
        val c = DenseMatrix.zero(2, 1)
        prepared.symmInto(1.0, b, 0.0, c)
        assertContentEquals(y, c.values)
    }

    @Test
    fun `a prepared snapshot keeps its own structural arrays`() {
        val pointers = intArrayOf(0, 1, 2)
        val rows = intArrayOf(0, 1)
        val values = doubleArrayOf(2.0, 3.0)
        val source = SparseMatrix.wrap(2, 2, pointers, rows, values)
        val prepared = source.prepare()

        values[0] = Double.NaN
        val y = DoubleArray(2)
        prepared.gemvInto(1.0, doubleArrayOf(1.0, 1.0), 0.0, y)

        assertContentEquals(doubleArrayOf(2.0, 3.0), y)
    }

    // The two families differ here: this one runs the transposed traversal over the stored orientation
    // however often it is called, where the sparse-sparse one above derives the opposite orientation once.
    @Test
    fun `a repeated prepared transposed dense product agrees and derives nothing`() {
        val source = matrix()
        val prepared = source.prepare()
        val b = DenseMatrix.wrap(3, 2, doubleArrayOf(1.5, -2.0, 0.5, 4.0, -1.0, 2.5))
        val expected = DenseMatrix.zero(2, 2)
        koblas.gemm(0.875, source, true, b, false, -0.25, expected)

        repeat(2) {
            val actual = DenseMatrix.zero(2, 2)
            prepared.gemmInto(0.875, true, b, false, -0.25, actual)
            assertContentEquals(expected.values, actual.values)
        }

        assertFalse(prepared.orientationDerived, "a repeated prepared dense product built the transpose cache")
    }
}

/** An engine that counts the products reaching it, to see which one a prepared operand hands its call to. */
private class RecordingSparseBlas(private val delegate: SparseBlas) : SparseBlas by delegate {
    var denseProducts: Int = 0

    @Suppress("LongParameterList") // the delegated dgemm signature
    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        right: Boolean,
        workspace: Workspace?,
    ) {
        denseProducts++
        delegate.gemm(alpha, a, transposeA, b, transposeB, beta, c, right, workspace)
    }
}
