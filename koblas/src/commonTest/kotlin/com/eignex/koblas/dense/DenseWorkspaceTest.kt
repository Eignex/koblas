package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * That a workspace handed to a dense routine is used, and that using it changes nothing.
 *
 * Two questions, and they are separate. The result has to be the one [ReferenceBlas] computes from operands
 * that were never aliased, which is what catches a staging bug that both the lent and the unlent path would
 * otherwise share. And the loan has to actually happen, which is visible only as the buffers the workspace
 * holds afterwards: a routine that took the parameter and allocated anyway would pass the first question.
 */
class DenseWorkspaceTest {
    private val blas: DenseBlas get() = testDenseBlas

    /**
     * A matrix-vector product whose input is also its destination, against the oracle's answer for the same
     * numbers held apart.
     */
    @Test
    fun `a staged vector gives the answer the reference computes from unaliased operands`() {
        val rng = Random(20260927)
        val n = 7
        val workspace = Workspace()
        for (transpose in booleanArrayOf(false, true)) {
            val a = randomMatrix(n, n, rng)
            val start = randomVector(n, rng)
            val expected = start.copyOf()
            ReferenceBlas.gemv(0.875, a, start.copyOf(), -0.25, expected, transpose)
            val aliased = start.copyOf()

            blas.gemv(0.875, a, aliased, -0.25, aliased, transpose, workspace)

            assertClose(expected, aliased, "gemv transpose=$transpose")
        }
    }

    /**
     * A product whose two operands are both its destination, which is two staging loans of one length at
     * once. A loan that overwrote the other would show up as the wrong answer rather than as a leak.
     */
    @Test
    fun `a product with both operands sharing the destination stages each of them`() {
        val rng = Random(20260928)
        val n = 6
        val workspace = Workspace()
        val start = randomMatrix(n, n, rng)
        val expected = copyOf(start)
        ReferenceBlas.gemm(0.875, copyOf(start), false, copyOf(start), true, -0.25, expected)
        val aliased = copyOf(start)

        blas.gemm(0.875, aliased, false, aliased, true, -0.25, aliased, workspace)

        assertClose(expected.values, aliased.values, "gemm with both operands aliased")
        assertEquals(2, workspace.available(n * n), "the two staged operands did not take a loan each")
    }

    /**
     * A solve whose triangle shares the block it solves, which holds the staged triangle and the gathered
     * right-hand side at the same time and at two different lengths.
     */
    @Test
    fun `a triangular solve stages its triangle and gathers its side from the same workspace`() {
        val rng = Random(20260929)
        val n = 5
        val workspace = Workspace()
        val start = triangle(randomMatrix(n, n, rng), n)
        val expected = copyOf(start)
        ReferenceBlas.trsm(copyOf(start), expected, lower = true)
        val aliased = copyOf(start)

        blas.trsm(aliased, aliased, lower = true, workspace = workspace)

        assertClose(expected.values, aliased.values, "trsm with its own triangle", tolerance = 1e-9)
        assertEquals(1, workspace.available(n * n), "the triangle was not staged from the workspace")
        assertEquals(1, workspace.available(n), "the right-hand side was not gathered from the workspace")
    }

    /** A second call over the same shape reuses the first call's buffers rather than asking for more. */
    @Test
    fun `a repeated call over one shape reuses the same scratch`() {
        val rng = Random(20260930)
        val order = 5
        val workspace = Workspace()
        val triangle = triangle(randomMatrix(order, order, rng), order)
        val b = randomMatrix(order, 3, rng)

        blas.trmm(triangle, b, lower = true, workspace = workspace)
        val afterOne = workspace.available(order)
        repeat(3) { blas.trmm(triangle, b, lower = true, workspace = workspace) }

        // A triangular multiply holds the gathered side and the copy of it the product reads, both of the
        // triangle's order, so one call leaves two behind and any number of them leave the same two.
        assertEquals(2, afterOne, "a triangular multiply did not hold its two loans at once")
        assertEquals(afterOne, workspace.available(order), "a repeated call kept asking for new buffers")
    }

    /** A call whose contract stops before the arithmetic takes no loan, because it stages nothing. */
    @Test
    fun `a call with nothing to compute never reaches the workspace`() {
        val rng = Random(20260931)
        val n = 4
        val workspace = Workspace()
        val a = randomMatrix(n, n, rng)
        val c = randomMatrix(n, n, rng)

        blas.gemm(0.0, a, false, a, false, -0.25, c, workspace)
        blas.gemv(0.0, a, DoubleArray(n), -0.25, DoubleArray(n), false, workspace)

        assertEquals(0, workspace.idleLengths(), "a no-work call borrowed scratch it had no use for")
    }

    /** A triangle with a dominant diagonal, so a solve over it is well conditioned. */
    private fun triangle(a: DenseMatrix, n: Int): DenseMatrix {
        val t = copyOf(a)
        for (i in 0 until n) t.values[i + i * n] = 3.0 + i % 2
        return t
    }

    private fun copyOf(a: DenseMatrix) = DenseMatrix.wrap(a.rows, a.cols, a.values.copyOf())
}
