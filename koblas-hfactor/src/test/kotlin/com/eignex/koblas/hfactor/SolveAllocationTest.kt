package com.eignex.koblas.hfactor

import com.eignex.koblas.AllocationGuarantee
import com.eignex.koblas.SparseMatrix
import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory
import kotlin.test.*

/**
 * What a solve through a native binding allocates, against what it says it allocates.
 *
 * The wrapper costs a little on every call whatever the size: the ownership anchor builds closures, and the
 * binding wraps the arrays it hands over. That is why these declare the size-independent guarantee and not
 * the stronger one, and this is what holds them to it.
 */
class SolveAllocationTest {
    private val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean
    private var sink: Any? = null

    private fun bytesPerSolve(n: Int): Double {
        val diagonal = SparseMatrix.ofColumns(n, n, (0 until n).map { j -> listOf(j to (n + 10.0)) })
        val factorization = BundledHfactor().factor(diagonal)
        val b = DoubleArray(n) { 1.0 + it }
        val out = DoubleArray(n)
        repeat(500) { sink = factorization.solveInto(b, out) }
        val id = Thread.currentThread().threadId()
        var best = Double.MAX_VALUE
        repeat(5) {
            val before = bean.getThreadAllocatedBytes(id)
            repeat(5000) { sink = factorization.solveInto(b, out) }
            val after = bean.getThreadAllocatedBytes(id)
            best = minOf(best, (after - before).toDouble() / 5000)
        }
        return best
    }

    /**
     * The guarantee is that the cost is bounded independently of the size, not that it is identical: the
     * measured figure moves a little as the JIT settles differently at each size. Eight times the order is
     * what makes the test mean something, since a cost that tracked the problem would rise with it.
     */
    @Test
    fun `a solve allocates no more for a larger problem`() {
        val small = bytesPerSolve(64)
        val large = bytesPerSolve(512)

        assertTrue(
            large <= small * 2.0,
            "eight times the order should not cost more to solve: $small at 64 against $large at 512",
        )
        assertTrue(large < 1024.0, "and the bound is a constant, not a function of n: $large at 512")
    }

    @Test
    fun `a solve declares the guarantee it keeps`() {
        val n = 64
        val diagonal = SparseMatrix.ofColumns(n, n, (0 until n).map { j -> listOf(j to (n + 10.0)) })
        val declared = BundledHfactor().factor(diagonal).solveAllocation(aliasing = false, transpose = false)

        assertEquals(
            AllocationGuarantee.NO_SIZE_DEPENDENT_MANAGED,
            declared.guarantee,
            "the solve allocates per call, so it cannot promise NO_MANAGED",
        )
    }
}
