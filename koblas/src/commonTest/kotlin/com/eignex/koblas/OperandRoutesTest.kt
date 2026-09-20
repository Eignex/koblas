package com.eignex.koblas

import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.sparse.SparseCall
import com.eignex.koblas.sparse.SparseMatrixRoute
import com.eignex.koblas.sparse.SparseMatrixOperation
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.test.fail

class OperandRoutesTest {

    /** A [Matrix] whose entries cannot be read at all, since describing a call reads none of them. */
    private class PoisonMatrix(override val rows: Int, override val cols: Int) : Matrix {
        override fun get(i: Int, j: Int): Double = fail("a route read an operand at ($i, $j)")
        override fun toArray(): Array<DoubleArray> = fail("a route materialised an operand")
    }

    private fun dense(rows: Int, cols: Int): DenseMatrix =
        DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { it + 1.0 })

    private fun sparse(rows: Int, cols: Int): SparseMatrix = SparseMatrix.ofColumns(
        rows,
        cols,
        List(cols) { j -> List(rows) { i -> i to (i + j + 1.0) } },
    )

    @Test
    fun `a dense product names the same route as the call described by hand`() {
        val route = koblas.routeOf(ALPHA, dense(4, 3), false, dense(3, 5), false, BETA, dense(4, 5))

        assertEquals(
            koblas.routeOf(DenseMatrixOperation.Gemm, DenseCall(4, 5, ALPHA, BETA, depth = 3)).toString(),
            route.toString(),
        )
    }

    @Test
    fun `a sparse operand keeps the side it sits on and the transpose that is its own`() {
        val a = sparse(4, 3)

        val left = koblas.routeOf(ALPHA, a, true, dense(4, 6), false, BETA, dense(3, 6))
        val right = koblas.routeOf(ALPHA, dense(5, 4), false, a, false, BETA, dense(5, 3))

        assertEquals(
            koblas.routeOf(
                SparseMatrixOperation.GemmDense,
                SparseCall(
                    a,
                    ALPHA,
                    BETA,
                    destinationElements = 18,
                    depth = 4,
                    rightHandSides = 6,
                    transposeSparse = true,
                ),
            ).toString(),
            left.toString(),
        )
        assertEquals(
            koblas.routeOf(
                SparseMatrixOperation.GemmDenseRight,
                SparseCall(
                    a,
                    ALPHA,
                    BETA,
                    destinationElements = 15,
                    depth = 4,
                    updateRun = 5,
                    rightHandSides = 5,
                ),
            ).toString(),
            right.toString(),
        )
    }

    @Test
    fun `two sparse operands name the fresh result or the dense destination the call was given`() {
        val a = sparse(4, 3)
        val b = sparse(3, 2)

        assertEquals("spgemm", koblas.routeOf(1.0, a, false, b, false).entryPoint)
        assertEquals("spgemm-dense", koblas.routeOf(1.0, a, false, b, false, 0.0, dense(4, 2)).entryPoint)
    }

    @Test
    fun `an operand from outside this library is the dense product its staged copy feeds`() {
        val route = koblas.routeOf(ALPHA, PoisonMatrix(4, 3), false, dense(3, 5), false, BETA, dense(4, 5))

        assertEquals("gemm", route.entryPoint)
    }

    @Test
    fun `a prepared product against a dense block is the route its snapshot runs`() {
        val source = sparse(4, 3)
        val prepared = koblas.prepare(source)
        val b = dense(3, 6)
        val c = dense(4, 6)

        val route = koblas.routeOf(ALPHA, prepared, false, b, false, BETA, c)

        assertEquals(koblas.routeOf(ALPHA, source, false, b, false, BETA, c).toString(), route.toString())
        assertFalse(prepared.orientationDerived, "describing a prepared dense product derived an orientation")
    }

    @Test
    fun `a prepared transposed sparse product names the orientation its second operand settles`() {
        val prepared = koblas.prepare(sparse(4, 3))

        val route = koblas.routeOf(1.0, prepared, true, sparse(4, 2), false) as SparseMatrixRoute

        assertTrue(route.resolved, route.toString())
        assertTrue(route.reason.orEmpty().contains("derived on first use"), route.toString())
        assertFalse(prepared.orientationDerived, "describing the call derived the orientation it describes")
    }

    @Test
    fun `a prepared transposed product against an empty operand keeps the stored orientation`() {
        val prepared = koblas.prepare(sparse(4, 3))
        val empty = SparseMatrix.ofColumns(4, 2, List(2) { emptyList<Pair<Int, Double>>() })

        val route = koblas.routeOf(1.0, prepared, true, empty, false) as SparseMatrixRoute

        assertTrue(route.resolved, route.toString())
        assertTrue(route.reason.orEmpty().contains("no prepared orientation is derived"), route.toString())
        assertFalse(prepared.orientationDerived, "a call reaching no position derived an orientation")
    }

    private companion object {
        const val ALPHA = 0.875
        const val BETA = -0.25
    }
}
