package com.eignex.koblas.sparse

import com.eignex.koblas.F64DenseVector
import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.F64SparseVector
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

class SparseSeamTest {

    private class CountingVectorKernels(override val priority: Int = 50) : F64SparseVectorKernels {
        override val name: String get() = "counting"
        var dots = 0
        var axpys = 0
        var scatters = 0
        var nrm2s = 0
        var asums = 0

        override fun dot(x: F64SparseVector, y: DoubleArray): Double {
            dots++
            return F64ReferenceSparseLinearAlgebra.dot(x, y)
        }

        override fun dot(x: F64SparseVector, y: F64SparseVector): Double {
            dots++
            return F64ReferenceSparseLinearAlgebra.dot(x, y)
        }

        override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector) {
            axpys++
            F64ReferenceSparseLinearAlgebra.axpy(y, alpha, x)
        }

        override fun scatter(x: F64SparseVector, out: DoubleArray) {
            scatters++
            F64ReferenceSparseLinearAlgebra.scatter(x, out)
        }

        override fun nrm2(x: F64SparseVector): Double {
            nrm2s++
            return F64ReferenceSparseLinearAlgebra.nrm2(x)
        }

        override fun asum(x: F64SparseVector): Double {
            asums++
            return F64ReferenceSparseLinearAlgebra.asum(x)
        }
    }

    private class CountingSparseBlas(override val priority: Int = 50) : F64SparseBlas {
        override val name: String get() = "counting-blas"
        var gemvs = 0

        @Suppress("LongParameterList")
        override fun gemv(
            alpha: Double,
            a: F64SparseMatrix,
            x: DoubleArray,
            beta: Double,
            y: DoubleArray,
            transpose: Boolean,
        ) {
            gemvs++
            F64ReferenceSparseLinearAlgebra.gemv(alpha, a, x, beta, y, transpose)
        }

        override fun trsv(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
            F64ReferenceSparseLinearAlgebra.trsv(a, x, lower, transpose, unitDiag)
    }

    private class CountingSparseLapack(override val priority: Int = 50) : F64SparseLapack {
        override val name: String get() = "counting-lapack"
        var factors = 0

        override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): F64SparseFactorization {
            factors++
            return F64ReferenceSparseLinearAlgebra.factor(a, equilibrate, dropTolerance)
        }

        override fun analyze(a: F64SparseMatrix, ordering: SparseOrdering) =
            F64ReferenceSparseLinearAlgebra.analyze(a, ordering)

        override fun ldl(a: F64SparseMatrix, policy: SparseLdlPolicy, ordering: SparseOrdering) =
            F64ReferenceSparseLinearAlgebra.ldl(a, policy, ordering)
    }

    private fun sparse() = F64SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))

    @Test
    fun `every public sparse vector operation reaches the registered kernels`() = withCleanBackends {
        val kernels = CountingVectorKernels()
        registerBackend(kernels)
        val x = sparse()
        val dense = F64DenseVector.of(DoubleArray(6) { it + 1.0 })

        assertEquals(2.0 * 2.0 + -3.0 * 5.0, x dot dense)
        assertEquals(2.0 * 2.0 + -3.0 * 5.0, dense dot x)
        assertEquals(4.0 + 9.0, x dot x)
        assertEquals(3, kernels.dots, "all three dot combinations should route")

        axpy(F64DenseVector.of(DoubleArray(6)), 2.0, x)
        assertEquals(1, kernels.axpys)

        copy(x, F64DenseVector.of(DoubleArray(6)))
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
        val a = F64DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = F64DenseVector.of(doubleArrayOf(3.0, 4.0))
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
        val a = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))

        assertTrue(doubleArrayOf(2.0, 8.0).contentEquals(a.gemv(doubleArrayOf(1.0, 2.0))))
        assertEquals(1, blas.gemvs, "F64SparseMatrix.gemv should forward to the seam")

        val f = a.lu()
        assertEquals(1, lapack.factors, "F64SparseMatrix.lu should forward to the seam")
        assertTrue(!f.singular)
        assertEquals(8.0, f.determinant(), absoluteTolerance = 1e-12)
    }

    @Test
    fun `the halves land in the context independently`() = withCleanBackends {
        registerBackend(CountingSparseBlas())
        registerBackend(CountingSparseLapack())
        assertEquals("counting-blas", koblas.sparseBlas.name)
        assertEquals("counting-lapack", koblas.sparseLapack.name)
        resetBackends()
        registerBackend(F64ReferenceSparseLinearAlgebra)
        assertSame(F64ReferenceSparseLinearAlgebra, koblas.sparseBlas)
        assertSame(F64ReferenceSparseLinearAlgebra, koblas.sparseLapack)
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
            F64PlatformSparseVectorKernels,
            koblas.sparseVectorKernels,
            "an empty registry means the compiled-in kernels for this target",
        )
    }

    @Test
    fun `an empty registry resolves to the portable implementation on all three sparse halves`() = withCleanBackends {
        assertSame(F64ReferenceSparseLinearAlgebra, koblas.sparseBlas)
        assertSame(F64ReferenceSparseLinearAlgebra, koblas.sparseLapack)
        assertSame(F64PlatformSparseVectorKernels, koblas.sparseVectorKernels)
    }
}
