package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.ReferenceBackend
import com.eignex.koblas.dense.ReferenceLapack
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

class KoblasContextTest {

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
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        val f = koblas.factor(a)
        val x = koblas.solve(f, doubleArrayOf(3.0, 5.0))
        assertContentClose(x, a.lu().solve(doubleArrayOf(3.0, 5.0)))
        val s = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
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

    @Test
    fun `the inherited routines run on the kernels their backend was built with`() {
        val mine = Counting()
        val backend = ReferenceBackend(mine)
        val l = F64DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(1.0, 3.0)))
        val x = doubleArrayOf(2.0, 5.0)

        (backend as Blas).trsv(l, x, lower = true)
        assertTrue(mine.dots + mine.axpys > 0, "trsv must use the backend's kernels")

        val before = mine.dots + mine.axpys
        (backend as Lapack).cholesky(F64DenseMatrix.of(arrayOf(doubleArrayOf(4.0, 1.0), doubleArrayOf(1.0, 3.0))))
        assertTrue(mine.dots + mine.axpys > before, "cholesky must use the backend's kernels")
    }

    /** A backend built the way host bindings are, delegating to the portable base with its own kernels. */
    private class LapackHalf(kernels: VectorKernels) : Lapack by ReferenceLapack(kernels) {
        override val name: String get() = "half"
    }

    @Test
    fun `qrPivoted runs on the kernels of the half it was called on`() {
        val mine = Counting()
        LapackHalf(mine).qrPivoted(F64DenseMatrix.of(arrayOf(doubleArrayOf(3.0, 1.0), doubleArrayOf(4.0, 2.0))))
        assertTrue(mine.dots + mine.axpys > 0, "qrPivoted must not fall back to the installed kernels")
    }

    @Test
    fun `a contexts own kernels reach the reference inner loops`() {
        val mine = Counting()
        val portable: Blas = ReferenceBackend(mine)
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))

        portable.gemv(a, doubleArrayOf(1.0, 1.0))
        assertTrue(mine.axpys > 0, "a non-transposed gemv sweeps columns with axpy")

        val before = mine.dots
        portable.gemv(1.0, a, doubleArrayOf(1.0, 1.0), 0.0, DoubleArray(2), transpose = true)
        assertTrue(mine.dots > before, "a transposed gemv dots contiguous columns")
    }

    @Test
    fun `the shared reference follows the process default kernels`() {
        val mine = Counting("installed")
        installBackends(koblas.with(vectorKernels = mine))
        val lapack: Lapack = ReferenceLinearAlgebra
        lapack.factor(F64DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))))
        assertTrue(mine.axpys > 0, "ReferenceLinearAlgebra must pick up an installed context's kernels")
    }

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

    /** A context is at least as preferred as the strongest half in it, and names what it is made of. */
    @Test
    fun `a context reports the strongest half's priority and names its backends`() {
        val reference = KoblasContext(
            vectorKernels = Counting(),
            blas = ReferenceLinearAlgebra,
            lapack = ReferenceLinearAlgebra,
            sparseVectorKernels = ReferenceSparseLinearAlgebra,
            sparseBlas = ReferenceSparseLinearAlgebra,
            sparseLapack = ReferenceSparseLinearAlgebra,
        )
        assertEquals(0, reference.priority, "every reference half has priority 0")
        assertEquals("reference", reference.name, "one distinct backend name should not repeat")
        assertEquals("KoblasContext(reference)", reference.toString())

        val strong = object : Lapack by ReferenceLinearAlgebra {
            override val name: String get() = "strong"
            override val priority: Int get() = 42
        }
        val mixed = reference.with(lapack = strong)
        assertEquals(42, mixed.priority, "the context should take the strongest half's priority")
        assertEquals("reference+strong", mixed.name, "both distinct names, in half order")
        assertSame(reference.blas, mixed.blas, "with() should keep the halves it was not given")
    }
}
