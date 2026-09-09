package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.sparse.ReferenceSparseDecompositions
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLinearAlgebra
import kotlin.test.*

class KoblasContextTest {

    private class Counting(override val name: String = "counting") : Kernels {
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

        override fun sum(v: DoubleArray, vOff: Int, len: Int): Double {
            var s = 0.0
            for (i in 0 until len) s += v[vOff + i]
            return s
        }

        override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
            var s = 0.0
            for (i in 0 until len) {
                val d = a[aOff + i] - b[bOff + i]
                s += d * d
            }
            return s
        }

        override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
            for (i in 0 until len) {
                val t = a[aOff + i]
                a[aOff + i] = b[bOff + i]
                b[bOff + i] = t
            }
        }

        override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens =
            portableRotmg(d1, d2, x1, y1)

        @Suppress("LongParameterList")
        override fun rotm(
            x: DoubleArray,
            xOff: Int,
            xStride: Int,
            y: DoubleArray,
            yOff: Int,
            yStride: Int,
            len: Int,
            transformation: ModifiedGivens,
        ) = portableRotm(x, xOff, xStride, y, yOff, yStride, len, transformation)

        @Suppress("LongParameterList")
        override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
            portableRot(x, xOff, y, yOff, len, c, s)
    }

    @AfterTest
    fun restore() = installBackends(null)

    @Test
    fun `a context is usable as a backend`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        assertContentClose(doubleArrayOf(4.0, 7.0), koblas.gemv(a, doubleArrayOf(1.0, 2.0)))
        val s = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
        assertContentClose(doubleArrayOf(2.0, 8.0), koblas.gemv(s, doubleArrayOf(1.0, 2.0)))
    }

    /** The roles are read out of a composition once, so a new one has to be read and the old one kept. */
    @Test
    fun `with rereads the factorization roles only from a new composition`() {
        val base = koblas
        val replacement = ReferenceSparseDecompositions(equilibrate = true)

        val rederived = base.with(sparseDecompositions = replacement)
        val kept = base.with(kernels = Counting())

        assertSame(replacement, rederived.generalSparseLu)
        assertSame(replacement, rederived.sparseCholesky)
        assertSame(replacement, rederived.quasiDefiniteLdl)
        assertSame(replacement, rederived.sparseQr)
        assertSame(base.generalSparseLu, kept.generalSparseLu)
        assertSame(base.sparseQr, kept.sparseQr)
    }

    /** A composition filling none of the basis half leaves that role to koblas's own. */
    @Test
    fun `a composition that fills no basis role falls back to the reference`() {
        val replacement = ReferenceSparseDecompositions()

        val context = koblas.with(sparseDecompositions = replacement)

        assertSame(ReferenceSparseLinearAlgebra, context.basisFactorizations)
    }

    /** Reading roles out of a composition is the one place a partial one is caught, and it says which role. */
    @Test
    fun `a composition filling no QR role is rejected at construction`() {
        val partial = object : SparseLapack by ReferenceSparseLinearAlgebra {
            override val name: String get() = "partial"
        }

        val failure = assertFailsWith<IllegalStateException> {
            KoblasContext(
                kernels = koblas.kernels,
                blas = ReferenceBlas,
                sparseKernels = ReferenceSparseLinearAlgebra,
                sparseBlas = ReferenceSparseLinearAlgebra,
                sparseDecompositions = partial,
                basisSolvers = ReferenceSparseLinearAlgebra,
            )
        }

        assertEquals("partial fills no general sparse LU role", failure.message)
    }

    @Test
    fun `with keeps every half it is not given`() {
        val base = koblas
        val mine = Counting()
        val derived = base.with(kernels = mine)
        assertSame(mine, derived.kernels)
        assertSame(base.blas, derived.blas)
        assertSame(base.sparseBlas, derived.sparseBlas)
        assertEquals(base.sparseDecompositions.name, derived.sparseDecompositions.name)
        assertSame(base.sparseKernels, derived.sparseKernels)
        assertSame(base.kernels, base.kernels, "the original must be untouched; contexts are values")
    }

    @Test
    fun `sparse linear algebra exposes its vector kernels`() {
        val context: SparseLinearAlgebra = koblas

        assertSame(koblas.sparseKernels, context.sparseKernels)
    }

    @Test
    fun `the inherited routines run on the kernels their backend was built with`() {
        val mine = Counting()
        val backend = ReferenceBackend(mine)
        val l = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(1.0, 3.0)))
        val x = doubleArrayOf(2.0, 5.0)

        backend.trsv(l, x, lower = true)

        assertTrue(mine.dots + mine.axpys > 0, "trsv must use the backend's kernels")
    }

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

    @Test
    fun `the shared reference follows the process default kernels`() {
        val mine = Counting("installed")
        installBackends(koblas.with(kernels = mine))
        val blas: Blas = ReferenceBlas
        blas.gemv(
            DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))),
            doubleArrayOf(1.0, 1.0),
        )
        assertTrue(mine.axpys > 0, "ReferenceBlas must pick up an installed context's kernels")
    }

    @Test
    fun `the name covers the matrix halves and not the kernels`() {
        val counting = Counting()
        assertEquals(koblas.name, koblas.with(kernels = counting).name, "kernels do not belong in the name")
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
            kernels = Counting(),
            blas = ReferenceBlas,
            sparseKernels = ReferenceSparseLinearAlgebra,
            sparseBlas = ReferenceSparseLinearAlgebra,
            sparseDecompositions = ReferenceSparseLinearAlgebra,
            basisSolvers = ReferenceSparseLinearAlgebra,
        )
        assertEquals(0, reference.priority, "every reference half has priority 0")
        assertEquals("reference", reference.name, "one distinct backend name should not repeat")
        assertEquals("KoblasContext(reference)", reference.toString())

        val strong = object : Blas by ReferenceBlas {
            override val name: String get() = "strong"
            override val priority: Int get() = 42
        }
        val mixed = reference.with(blas = strong)
        assertEquals(42, mixed.priority, "the context should take the strongest half's priority")
        assertEquals("strong+reference", mixed.name, "both distinct names, in half order")
        assertSame(reference.sparseBlas, mixed.sparseBlas, "with() should keep the halves it was not given")
    }
}
