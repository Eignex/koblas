package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.prepare
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
        prepared.gemv(1.0, doubleArrayOf(4.0, 5.0), 0.0, actual)

        assertContentEquals(doubleArrayOf(8.0, 15.0, -4.0), actual)
    }

    /**
     * A prepared transposed product against a dense block runs the transposed traversal over the snapshot.
     *
     * Deriving the opposite orientation and running the untransposed schedule over it is the other way this
     * could go, and the calibration measured it slower: the transposed traversal reduces each stored column
     * into one destination row, and the oriented one spreads each index across the destination. So the
     * prepared route is the one-shot route, and asking for it builds nothing.
     */
    @Test
    fun `a prepared transposed product reports the schedule it actually runs`() {
        val source = SparseMatrix.ofColumns(
            3,
            2,
            listOf(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0)),
        )
        val prepared = koblas.prepare(source)
        // A transposed dense operand as well, so the reduction has its right-hand sides adjacent and reaches
        // a panel: what this is about is which traversal the route names, and a reduction over a strided
        // block is written out by the traversal instead.
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

    /**
     * A prepared transposed matrix-vector product reduces the snapshot's own columns, so asking about it
     * neither reports an oriented schedule nor builds one.
     */
    @Test
    fun `a prepared transposed gemv keeps the snapshot's own orientation`() {
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

    /**
     * A transposed product against a second sparse operand is decided by that operand, which the call facts
     * do not carry, so the route says so rather than choosing one of the two schedules.
     */
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

    /**
     * Prepared and one-shot routes retain the same uncertainty when destination facts are omitted.
     *
     * The extent can still affect scaling and no-work attribution, so each route is compared using the
     * same supplied facts. Neither query should initialize a transpose for a sparse-dense product.
     */
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

    /**
     * A transposed prepared product with a destination of no elements is decided, and decided the other
     * way: there is nothing to write, so nothing is oriented.
     */
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

    /**
     * An undecided orientation does not make the rest of a call undecided: a destination multiplier still
     * scales a destination, and a call with nothing to do still has nothing to do.
     */
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
    fun `all prepared products agree with the one-shot calls`() {
        val source = matrix()
        val prepared = koblas.prepare(source)
        val dense = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val expectedDense = koblas.gemm(source, dense)
        val actualDense = DenseMatrix.zero(3, 2)
        prepared.gemm(1.0, false, dense, 0.0, actualDense)
        assertClose(expectedDense, actualDense, "dense product")

        val right = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 2.0, 1 to -1.0)))
        assertEquals(koblas.gemm(source, right), prepared.gemm(right))

        val transposedDense = DenseMatrix.wrap(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expectedTransposed = DenseMatrix.zero(2, 2)
        koblas.gemm(1.0, source, true, transposedDense, true, 0.0, expectedTransposed, right = false)
        val actualTransposed = DenseMatrix.zero(2, 2)
        prepared.gemm(1.0, true, transposedDense, true, 0.0, actualTransposed, false)
        assertClose(expectedTransposed, actualTransposed, "full dense product")

        val sparseDense = DenseMatrix.zero(3, 1)
        prepared.gemm(2.0, false, right, false, 0.0, sparseDense)
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
        prepared.symv(1.0, doubleArrayOf(7.0, 11.0), 0.0, y)
        assertContentEquals(doubleArrayOf(47.0, 76.0), y)

        val b = DenseMatrix.wrap(2, 1, doubleArrayOf(7.0, 11.0))
        val c = DenseMatrix.zero(2, 1)
        prepared.symm(1.0, b, 0.0, c)
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
        prepared.gemv(1.0, doubleArrayOf(1.0, 1.0), 0.0, y)

        assertContentEquals(doubleArrayOf(2.0, 3.0), y)
    }

    /**
     * A repeated transposed product against a dense block agrees with the one-shot call and builds nothing.
     *
     * The two families of prepared product differ here, and each is pinned: this one runs the transposed
     * traversal over the stored orientation however many times it is called, and the sparse-sparse one
     * above derives the opposite orientation once. Both sides of the split are checked against the one-shot
     * answer, so the choice is a schedule rather than a different result.
     */
    @Test
    fun `a repeated prepared transposed dense product agrees and derives nothing`() {
        val source = matrix()
        val prepared = source.prepare()
        val b = DenseMatrix.wrap(3, 2, doubleArrayOf(1.5, -2.0, 0.5, 4.0, -1.0, 2.5))
        val expected = DenseMatrix.zero(2, 2)
        koblas.gemm(0.875, source, true, b, false, -0.25, expected)

        repeat(2) {
            val actual = DenseMatrix.zero(2, 2)
            prepared.gemm(0.875, transposeA = true, b = b, beta = -0.25, c = actual)
            assertContentEquals(expected.values, actual.values)
        }

        assertFalse(prepared.orientationDerived, "a repeated prepared dense product built the transpose cache")
    }
}
