package com.eignex.koblas.sparse

import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.asum
import com.eignex.koblas.axpy
import com.eignex.koblas.copy
import com.eignex.koblas.dot
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.norm2
import com.eignex.koblas.registerBackend
import com.eignex.koblas.resetBackends
import com.eignex.koblas.withCleanBackends
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * The three sparse seams: that the public sparse operations reach a registered backend, that ranking and
 * overriding work as they do on the dense side, and that the composed pair behaves.
 *
 * The point of a seam is that it can be replaced, so these tests replace it. A counting backend proves the
 * traffic actually arrives — which for the sparse vector tier could not be asserted at all before, because
 * those operations were `forEachStored` loops inlined into `Ops.kt` with nothing to substitute.
 */
class SparseSeamTest {

    private class CountingVectorKernels(override val priority: Int = 50) : SparseVectorKernels {
        override val name: String get() = "counting"
        var dots = 0
        var axpys = 0
        var scatters = 0
        var nrm2s = 0
        var asums = 0

        override fun dot(x: SparseVector, y: DoubleArray): Double {
            dots++
            return super.dot(x, y)
        }

        override fun dot(x: SparseVector, y: SparseVector): Double {
            dots++
            return super.dot(x, y)
        }

        override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
            axpys++
            super.axpy(y, alpha, x)
        }

        override fun scatter(x: SparseVector, out: DoubleArray) {
            scatters++
            super.scatter(x, out)
        }

        override fun nrm2(x: SparseVector): Double {
            nrm2s++
            return super.nrm2(x)
        }

        override fun asum(x: SparseVector): Double {
            asums++
            return super.asum(x)
        }
    }

    private class CountingSparseBlas(override val priority: Int = 50) : SparseBlas {
        override val name: String get() = "counting-blas"
        var gemvs = 0

        @Suppress("LongParameterList")
        override fun gemv(
            alpha: Double,
            a: SparseMatrix,
            x: DoubleArray,
            beta: Double,
            y: DoubleArray,
            transpose: Boolean,
        ) {
            gemvs++
            super.gemv(alpha, a, x, beta, y, transpose)
        }
    }

    private class CountingSparseLapack(override val priority: Int = 50) : SparseLapack {
        override val name: String get() = "counting-lapack"
        var factors = 0

        override fun factor(a: SparseMatrix, equilibrate: Boolean, dropTolerance: Double): SparseFactorization {
            factors++
            return super.factor(a, equilibrate, dropTolerance)
        }
    }

    // Restoration is withCleanBackends' job, per test, so an eagerly-registered platform backend survives.

    private fun sparse() = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))

    @Test
    fun `every public sparse vector operation reaches the registered kernels`() = withCleanBackends {
        val kernels = CountingVectorKernels()
        registerBackend(kernels)
        val x = sparse()
        val dense = DenseVector.of(DoubleArray(6) { it + 1.0 })

        assertEquals(2.0 * 2.0 + -3.0 * 5.0, x dot dense)
        assertEquals(2.0 * 2.0 + -3.0 * 5.0, dense dot x)
        assertEquals(4.0 + 9.0, x dot x)
        assertEquals(3, kernels.dots, "all three dot combinations should route")

        axpy(DenseVector.of(DoubleArray(6)), 2.0, x)
        assertEquals(1, kernels.axpys)

        copy(x, DenseVector.of(DoubleArray(6)))
        assertEquals(1, kernels.scatters, "copy from a sparse source is a scatter")

        norm2(x)
        asum(x)
        assertEquals(1, kernels.nrm2s)
        assertEquals(1, kernels.asums)
    }

    @Test
    fun `a dense-only operation does not reach the sparse kernels`() = withCleanBackends {
        val kernels = CountingVectorKernels()
        registerBackend(kernels)
        val a = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = DenseVector.of(doubleArrayOf(3.0, 4.0))
        assertEquals(11.0, a dot b)
        norm2(a)
        asum(a)
        assertEquals(0, kernels.dots + kernels.nrm2s + kernels.asums, "dense work must stay on the dense kernels")
    }

    @Test
    fun `the matrix product and the factorization reach their halves`() = withCleanBackends {
        val blas = CountingSparseBlas()
        val lapack = CountingSparseLapack()
        registerBackend(blas)
        registerBackend(lapack)
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))

        assertTrue(doubleArrayOf(2.0, 8.0).contentEquals(a.gemv(doubleArrayOf(1.0, 2.0))))
        assertEquals(1, blas.gemvs, "SparseMatrix.gemv should forward to the seam")

        val f = a.lu()
        assertEquals(1, lapack.factors, "SparseMatrix.lu should forward to the seam")
        assertTrue(!f.singular)
        assertEquals(8.0, f.determinant(), absoluteTolerance = 1e-12)
    }

    @Test
    fun `the halves land in the context independently`() = withCleanBackends {
        registerBackend(CountingSparseBlas())
        registerBackend(CountingSparseLapack())
        assertEquals("counting-blas", koblas.sparseBlas.name)
        assertEquals("counting-lapack", koblas.sparseLapack.name)
        // One object registered for both halves lands in both, with no composite in between.
        resetBackends()
        registerBackend(ReferenceSparseLinearAlgebra)
        assertSame(ReferenceSparseLinearAlgebra, koblas.sparseBlas)
        assertSame(ReferenceSparseLinearAlgebra, koblas.sparseLapack)
    }

    @Test
    fun `registration keeps the highest priority and install overrides both`() = withCleanBackends {
        val weak = CountingVectorKernels(priority = 10)
        val strong = CountingVectorKernels(priority = 200)
        registerBackend(strong)
        registerBackend(weak)
        assertSame(strong, koblas.sparseVectorKernels, "a weaker registration displaced a stronger one")
        val override = CountingVectorKernels(priority = 0)
        installBackends(koblas.with(sparseVectorKernels = override))
        assertSame(override, koblas.sparseVectorKernels, "install must win regardless of priority")
        installBackends(null)
        assertSame(strong, koblas.sparseVectorKernels, "clearing the override falls back to registration")
        resetBackends()
        assertSame(
            ReferenceSparseLinearAlgebra,
            koblas.sparseVectorKernels,
            "an empty registry means the reference",
        )
    }

    @Test
    fun `an empty registry resolves to the reference on all three sparse halves`() = withCleanBackends {
        assertSame(ReferenceSparseLinearAlgebra, koblas.sparseBlas)
        assertSame(ReferenceSparseLinearAlgebra, koblas.sparseLapack)
        assertSame(ReferenceSparseLinearAlgebra, koblas.sparseVectorKernels)
    }
}
