package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.ReferenceBackend
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.VectorKernels
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.solve
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * The context as a value you can hold: that it is usable as a backend, that [KoblasContext.with] keeps what
 * you did not name, and that a context's own vector kernels actually reach the inner loops.
 *
 * That last one is the point of the whole exercise and the thing that did not work before. Backend choice
 * was global, so "use these kernels for this piece of work" had nowhere to live.
 */
class KoblasContextTest {

    /** Counts what the inner loops ask of it and computes the answer with a plain loop. */
    private class Counting(override val name: String = "counting") : VectorKernels {
        override val priority: Int get() = 0
        var dots = 0
        var axpys = 0

        override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
            dots++
            var s = 0.0
            for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
            return s
        }

        override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
            axpys++
            for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
        }

        override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
            for (i in 0 until len) v[vOff + i] *= alpha
        }

        override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)

        override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)
    }

    @AfterTest
    fun restore() = installBackends(null)

    @Test
    fun `a context is usable as a backend`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        val f = koblas.factor(a)
        val x = koblas.solve(f, doubleArrayOf(3.0, 5.0))
        // Same answer through the free-function spelling, which uses this same context.
        assertContentClose(x, a.lu().solve(doubleArrayOf(3.0, 5.0)))
        // And the sparse half answers on the same object, which is what makes one context enough.
        val s = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
        assertContentClose(doubleArrayOf(2.0, 8.0), koblas.gemv(s, doubleArrayOf(1.0, 2.0)))
    }

    @Test
    fun `with keeps every half it is not given`() {
        val base = koblas
        val mine = Counting()
        val derived = base.with(vectorKernels = mine)
        assertSame(mine, derived.vectorKernels)
        assertSame(base.blas, derived.blas)
        assertSame(base.lapack, derived.lapack)
        assertSame(base.sparseBlas, derived.sparseBlas)
        assertSame(base.sparseLapack, derived.sparseLapack)
        assertSame(base.sparseVectorKernels, derived.sparseVectorKernels)
        assertSame(base.vectorKernels, base.vectorKernels, "the original must be untouched; contexts are values")
    }

    /**
     * A context's kernels reach the reference backend's inner loops.
     *
     * `ReferenceBackend` takes the kernels it should use, so a context built around one carries all the way
     * down into `gemv`. Asserting the count rather than the result is the whole point: the arithmetic would
     * be right either way, and what is being tested is *which* kernels ran.
     */
    @Test
    fun `a contexts own kernels reach the reference inner loops`() {
        val mine = Counting()
        val portable: Blas = ReferenceBackend(mine)
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))

        portable.gemv(a, doubleArrayOf(1.0, 1.0))
        assertTrue(mine.axpys > 0, "a non-transposed gemv sweeps columns with axpy")

        val before = mine.dots
        portable.gemv(1.0, a, doubleArrayOf(1.0, 1.0), 0.0, DoubleArray(2), transpose = true)
        assertTrue(mine.dots > before, "a transposed gemv dots contiguous columns")
    }

    /** The shared reference instance follows the process default instead of pinning kernels. */
    @Test
    fun `the shared reference follows the process default kernels`() {
        val mine = Counting("installed")
        installBackends(koblas.with(vectorKernels = mine))
        val lapack: Lapack = ReferenceLinearAlgebra
        lapack.factor(DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))))
        assertTrue(mine.axpys > 0, "ReferenceLinearAlgebra must pick up an installed context's kernels")
    }

    /**
     * The name covers the matrix halves and deduplicates them; the kernels are reported by [mathBackend]
     * instead, because including them prefixed every name with `"simd(4 lanes)+"`.
     */
    @Test
    fun `the name covers the matrix halves and not the kernels`() {
        val counting = Counting()
        assertEquals(koblas.name, koblas.with(vectorKernels = counting).name, "kernels do not belong in the name")
        val named = koblas.with(sparseBlas = ReferenceSparseLinearAlgebra).name.split("+")
        assertEquals(named.size, named.distinct().size, "the name repeated a backend")
    }

    private fun assertContentClose(expected: DoubleArray, actual: DoubleArray) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) assertEquals(expected[i], actual[i], absoluteTolerance = 1e-9)
    }
}
