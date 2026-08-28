package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.*
import com.eignex.koblas.sparse.F64SparseCholeskyFactorization
import com.eignex.koblas.sparse.F64SparseLdlFactorization
import com.eignex.koblas.sparse.F64SparseLuFactorization
import kotlin.test.*

class SparseSeamTest {

    private class CountingKernels(override val priority: Int = 50) : F64SparseKernels {
        override val name: String get() = "counting"
        var dots = 0
        var axpys = 0
        var scatters = 0
        var gathers = 0
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

        override fun gather(x: F64SparseVector, from: DoubleArray) {
            gathers++
            F64ReferenceSparseLinearAlgebra.gather(x, from)
        }

        override fun gatherZero(x: F64SparseVector, from: DoubleArray) {
            gathers++
            F64ReferenceSparseLinearAlgebra.gatherZero(x, from)
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
        var gemms = 0
        var sparseProducts = 0
        var trsms = 0
        var transposes = 0

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

        override fun transpose(a: F64SparseMatrix): F64SparseMatrix {
            transposes++
            return F64ReferenceSparseLinearAlgebra.transpose(a)
        }

        @Suppress("LongParameterList")
        override fun gemm(
            alpha: Double,
            a: F64SparseMatrix,
            transposeA: Boolean,
            b: F64DenseMatrix,
            transposeB: Boolean,
            beta: Double,
            c: F64DenseMatrix,
            right: Boolean,
        ) {
            gemms++
            F64ReferenceSparseLinearAlgebra.gemm(alpha, a, transposeA, b, transposeB, beta, c, right)
        }

        override fun gemm(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix {
            sparseProducts++
            return F64ReferenceSparseLinearAlgebra.gemm(a, b)
        }

        @Suppress("LongParameterList")
        override fun trsm(
            a: F64SparseMatrix,
            b: F64DenseMatrix,
            lower: Boolean,
            transpose: Boolean,
            unitDiag: Boolean,
            right: Boolean,
            alpha: Double,
        ) {
            trsms++
            F64ReferenceSparseLinearAlgebra.trsm(a, b, lower, transpose, unitDiag, right, alpha)
        }
    }

    private class CountingSparseLu(override val priority: Int = 50) :
        F64SparseDecompositions,
        F64GeneralSparseLu,
        F64SparseCholesky,
        F64SparseLdl,
        F64SparseQr {
        override val name: String get() = "counting-decompositions"
        var factors = 0
        var choleskys = 0
        var ldls = 0
        var qrs = 0

        override fun factor(a: F64SparseMatrix): F64SparseLuFactorization {
            factors++
            return F64ReferenceSparseLinearAlgebra.factor(a)
        }

        override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization {
            choleskys++
            return F64ReferenceSparseLinearAlgebra.cholesky(a)
        }

        override fun ldl(a: F64SparseMatrix): F64SparseLdlFactorization {
            ldls++
            return F64ReferenceSparseLinearAlgebra.ldl(a)
        }

        override fun qr(a: F64SparseMatrix): F64SparseQrFactorization {
            qrs++
            return F64ReferenceSparseLinearAlgebra.qr(a)
        }
    }

    private fun sparse() = F64SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))

    @Test
    fun `every public sparse vector operation reaches the registered kernels`() = withCleanBackends {
        val kernels = CountingKernels()
        registerBackend(kernels)
        val x = sparse()
        val dense = F64DenseVector.of(DoubleArray(6) { it + 1.0 })

        assertEquals(2.0 * 2.0 + -3.0 * 5.0, x dot dense)
        assertEquals(2.0 * 2.0 + -3.0 * 5.0, dense dot x)
        assertEquals(4.0 + 9.0, x dot x)
        assertEquals(3, kernels.dots, "all three dot combinations should route")

        F64DenseVector.of(DoubleArray(6)).axpy(2.0, x)
        assertEquals(1, kernels.axpys)

        copy(x, F64DenseVector.of(DoubleArray(6)))
        assertEquals(1, kernels.scatters, "copy from a sparse source is a scatter")

        gather(sparse(), dense)
        gatherZero(sparse(), F64DenseVector.of(DoubleArray(6)))
        assertEquals(2, kernels.gathers, "both gathers should route")

        x.norm2()
        x.asum()
        assertEquals(1, kernels.nrm2s)
        assertEquals(1, kernels.asums)
    }

    @Test
    fun `a dense-only operation does not reach the sparse kernels`() = withCleanBackends {
        val kernels = CountingKernels()
        registerBackend(kernels)
        val a = F64DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = F64DenseVector.of(doubleArrayOf(3.0, 4.0))
        assertEquals(11.0, a dot b)
        a.norm2()
        a.asum()
        assertEquals(0, kernels.dots + kernels.nrm2s + kernels.asums, "dense work must stay on the dense kernels")
    }

    @Test
    fun `the matrix product and the factorization reach their halves`() = withCleanBackends {
        val blas = CountingSparseBlas()
        val decompositions = CountingSparseLu()
        registerBackend(blas)
        registerBackend(decompositions)
        val a = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))

        assertTrue(doubleArrayOf(2.0, 8.0).contentEquals(koblas.gemv(a, doubleArrayOf(1.0, 2.0))))
        assertEquals(1, blas.gemvs, "the context gemv should forward to the seam")

        a * F64DenseMatrix.diagonal(2)
        assertEquals(1, blas.gemms, "the product operator should forward to the seam")

        a.trsm(F64DenseMatrix.diagonal(2), lower = true)
        assertEquals(1, blas.trsms, "F64SparseMatrix.trsm should forward to the seam")

        a.transpose()
        assertEquals(1, blas.transposes, "F64SparseMatrix.transpose should forward to the seam")

        a * (a)
        assertEquals(1, blas.sparseProducts, "the sparse product should forward to the seam")

        val f = a.lu()
        assertEquals(1, decompositions.factors, "F64SparseMatrix.lu should forward to the seam")
        assertTrue(!f.singular)

        a.cholesky()
        assertEquals(1, decompositions.choleskys, "F64SparseMatrix.cholesky should forward to the seam")

        a.ldl()
        assertEquals(1, decompositions.ldls, "F64SparseMatrix.ldl should forward to the seam")
    }

    /**
     * The portable factorizations transpose on the way in, and the portable answer is the definition every
     * binding is compared against. Reaching the seam for it would route that definition through whichever
     * backend happens to be registered.
     */
    @Test
    fun `the portable Cholesky transposes without reaching the seam`() = withCleanBackends {
        val blas = CountingSparseBlas()
        registerBackend(blas)
        val spd = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 9.0)))

        F64ReferenceSparseLinearAlgebra.cholesky(spd)

        assertEquals(0, blas.transposes, "the portable factorization went through the registered backend")
    }

    @Test
    fun `the halves land in the context independently`() = withCleanBackends {
        registerBackend(CountingSparseBlas())
        registerBackend(CountingSparseLu())
        assertEquals("counting-blas", koblas.sparseBlas.name)
        assertEquals("counting-decompositions", koblas.sparseDecompositions.name)
        resetBackends()
        registerBackend(F64ReferenceSparseLinearAlgebra)
        assertSame(F64ReferenceSparseLinearAlgebra, koblas.sparseBlas)
        assertEquals("reference", koblas.sparseDecompositions.name)
    }

    @Test
    fun `registration keeps the highest priority and install overrides both`() = withCleanBackends {
        val weak = CountingKernels(priority = 10)
        val strong = CountingKernels(priority = 200)
        registerBackend(strong)
        registerBackend(weak)
        assertSame(strong, koblas.sparseKernels, "a weaker registration displaced a stronger one")
        val override = CountingKernels(priority = 0)
        installBackends(koblas.with(sparseKernels = override))
        assertSame(override, koblas.sparseKernels, "install must win regardless of priority")
        installBackends(null)
        assertSame(strong, koblas.sparseKernels, "clearing the override falls back to registration")
        resetBackends()
        assertSame(
            F64PlatformSparseKernels,
            koblas.sparseKernels,
            "an empty registry means the compiled-in kernels for this target",
        )
    }

    @Test
    fun `an empty registry resolves to the portable implementation on all three sparse halves`() = withCleanBackends {
        assertSame(F64ReferenceSparseLinearAlgebra, koblas.sparseBlas)
        assertEquals("reference", koblas.sparseDecompositions.name)
        assertSame(F64PlatformSparseKernels, koblas.sparseKernels)
    }
}
