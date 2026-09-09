package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.sparse.QuasiDefiniteLdlFactorization
import com.eignex.koblas.sparse.SparseCholeskyFactorization
import com.eignex.koblas.sparse.SparseLuFactorization
import kotlin.test.*

class SparseSeamTest {

    private class CountingKernels(override val priority: Int = 50) : SparseKernels {
        override val name: String get() = "counting"
        var dots = 0
        var axpys = 0
        var scatters = 0
        var gathers = 0
        var nrm2s = 0
        var asums = 0

        override fun dot(x: SparseVector, y: DoubleArray): Double {
            dots++
            return ReferenceSparseLinearAlgebra.dot(x, y)
        }

        override fun dot(x: SparseVector, y: SparseVector): Double {
            dots++
            return ReferenceSparseLinearAlgebra.dot(x, y)
        }

        override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
            axpys++
            ReferenceSparseLinearAlgebra.axpy(y, alpha, x)
        }

        override fun scatter(x: SparseVector, out: DoubleArray) {
            scatters++
            ReferenceSparseLinearAlgebra.scatter(x, out)
        }

        override fun gather(x: SparseVector, from: DoubleArray) {
            gathers++
            ReferenceSparseLinearAlgebra.gather(x, from)
        }

        override fun gatherZero(x: SparseVector, from: DoubleArray) {
            gathers++
            ReferenceSparseLinearAlgebra.gatherZero(x, from)
        }

        override fun nrm2(x: SparseVector): Double {
            nrm2s++
            return ReferenceSparseLinearAlgebra.nrm2(x)
        }

        override fun asum(x: SparseVector): Double {
            asums++
            return ReferenceSparseLinearAlgebra.asum(x)
        }
    }

    private class CountingSparseBlas(override val priority: Int = 50) : SparseBlas {
        override val name: String get() = "counting-blas"
        var gemvs = 0
        var gemms = 0
        var sparseProducts = 0
        var trsms = 0
        var trmms = 0
        var transposes = 0

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
            ReferenceSparseLinearAlgebra.gemv(alpha, a, x, beta, y, transpose)
        }

        override fun trsv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
            ReferenceSparseLinearAlgebra.trsv(a, x, lower, transpose, unitDiag)

        override fun trmv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
            ReferenceSparseLinearAlgebra.trmv(a, x, lower, transpose, unitDiag)

        override fun transpose(a: SparseMatrix): SparseMatrix {
            transposes++
            return ReferenceSparseLinearAlgebra.transpose(a)
        }

        @Suppress("LongParameterList")
        override fun gemm(
            alpha: Double,
            a: SparseMatrix,
            transposeA: Boolean,
            b: DenseMatrix,
            transposeB: Boolean,
            beta: Double,
            c: DenseMatrix,
            right: Boolean,
            workspace: Workspace?,
        ) {
            gemms++
            ReferenceSparseLinearAlgebra.gemm(alpha, a, transposeA, b, transposeB, beta, c, right, workspace)
        }

        override fun gemm(a: SparseMatrix, b: SparseMatrix): SparseMatrix {
            sparseProducts++
            return ReferenceSparseLinearAlgebra.gemm(a, b)
        }

        @Suppress("LongParameterList")
        override fun trsm(
            a: SparseMatrix,
            b: DenseMatrix,
            lower: Boolean,
            transpose: Boolean,
            unitDiag: Boolean,
            right: Boolean,
            alpha: Double,
            workspace: Workspace?,
        ) {
            trsms++
            ReferenceSparseLinearAlgebra.trsm(a, b, lower, transpose, unitDiag, right, alpha, workspace)
        }

        @Suppress("LongParameterList")
        override fun trmm(
            a: SparseMatrix,
            b: DenseMatrix,
            lower: Boolean,
            transpose: Boolean,
            unitDiag: Boolean,
            right: Boolean,
            alpha: Double,
        ) {
            trmms++
            ReferenceSparseLinearAlgebra.trmm(a, b, lower, transpose, unitDiag, right, alpha)
        }
    }

    private class CountingSparseLu(override val priority: Int = 50) :
        SparseLapack,
        GeneralSparseLu,
        SparseCholesky,
        QuasiDefiniteLdl,
        SparseQr {
        override val name: String get() = "counting-decompositions"
        var factors = 0
        var choleskys = 0
        var ldls = 0
        var qrs = 0

        override fun factor(a: SparseMatrix): SparseLuFactorization {
            factors++
            return ReferenceSparseLinearAlgebra.factor(a)
        }

        override fun cholesky(a: SparseMatrix): SparseCholeskyFactorization {
            choleskys++
            return ReferenceSparseLinearAlgebra.cholesky(a)
        }

        override fun quasiDefiniteLdl(a: SparseMatrix): QuasiDefiniteLdlFactorization {
            ldls++
            return ReferenceSparseLinearAlgebra.quasiDefiniteLdl(a)
        }

        override fun qr(a: SparseMatrix): SparseQrFactorization {
            qrs++
            return ReferenceSparseLinearAlgebra.qr(a)
        }
    }

    private fun sparse() = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))

    @Test
    fun `every public sparse vector operation reaches the registered kernels`() = withCleanBackends {
        val kernels = CountingKernels()
        registerBackend(kernels)
        val x = sparse()
        val dense = DenseVector.of(DoubleArray(6) { it + 1.0 })

        assertEquals(2.0 * 2.0 + -3.0 * 5.0, x dot dense)
        assertEquals(2.0 * 2.0 + -3.0 * 5.0, dense dot x)
        assertEquals(4.0 + 9.0, x dot x)
        assertEquals(3, kernels.dots, "all three dot combinations should route")

        DenseVector.of(DoubleArray(6)).axpy(2.0, x)
        assertEquals(1, kernels.axpys)

        copy(x, DenseVector.of(DoubleArray(6)))
        assertEquals(1, kernels.scatters, "copy from a sparse source is a scatter")

        gather(sparse(), dense)
        gatherZero(sparse(), DenseVector.of(DoubleArray(6)))
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
        val a = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = DenseVector.of(doubleArrayOf(3.0, 4.0))
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
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))

        assertTrue(doubleArrayOf(2.0, 8.0).contentEquals(koblas.gemv(a, doubleArrayOf(1.0, 2.0))))
        assertEquals(1, blas.gemvs, "the context gemv should forward to the seam")

        a * DenseMatrix.diagonal(2)
        assertEquals(1, blas.gemms, "the product operator should forward to the seam")

        a.trsm(DenseMatrix.diagonal(2), lower = true)
        assertEquals(1, blas.trsms, "SparseMatrix.trsm should forward to the seam")

        a.trmm(DenseMatrix.diagonal(2), lower = true)
        assertEquals(1, blas.trmms, "SparseMatrix.trmm should forward to the seam")

        a.transpose()
        assertEquals(1, blas.transposes, "SparseMatrix.transpose should forward to the seam")

        a * a
        assertEquals(1, blas.sparseProducts, "the sparse product should forward to the seam")

        val f = a.lu()
        assertEquals(1, decompositions.factors, "SparseMatrix.lu should forward to the seam")
        assertTrue(!f.singular)

        a.cholesky()
        assertEquals(1, decompositions.choleskys, "SparseMatrix.cholesky should forward to the seam")

        a.quasiDefiniteLdl()
        assertEquals(1, decompositions.ldls, "SparseMatrix.quasiDefiniteLdl should forward to the seam")

        @Suppress("DEPRECATION")
        a.quasiDefiniteLdl()
        assertEquals(2, decompositions.ldls, "the LDL alias should remain compatible")
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
        val spd = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 9.0)))

        ReferenceSparseLinearAlgebra.cholesky(spd)

        assertEquals(0, blas.transposes, "the portable factorization went through the registered backend")
    }

    @Test
    fun `the halves land in the context independently`() = withCleanBackends {
        registerBackend(CountingSparseBlas())
        registerBackend(CountingSparseLu())
        assertEquals("counting-blas", koblas.sparseBlas.name)
        assertEquals("counting-decompositions", koblas.sparseDecompositions.name)
        resetBackends()
        registerBackend(ReferenceSparseLinearAlgebra)
        assertSame(ReferenceSparseLinearAlgebra, koblas.sparseBlas)
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
            PlatformSparseKernels,
            koblas.sparseKernels,
            "an empty registry means the compiled-in kernels for this target",
        )
    }

    @Test
    fun `an empty registry resolves to the portable implementation on all three sparse halves`() = withCleanBackends {
        assertSame(ReferenceSparseLinearAlgebra, koblas.sparseBlas)
        assertEquals("reference", koblas.sparseDecompositions.name)
        assertSame(PlatformSparseKernels, koblas.sparseKernels)
    }
}
