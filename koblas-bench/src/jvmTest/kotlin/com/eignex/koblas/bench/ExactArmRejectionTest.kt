package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.vendor.Vendor
import kotlin.test.*

class ExactArmRejectionTest {
    private fun backend(operations: Set<EngineImplementation>) = object : KoblasEngine() {
        override val name = "openblas"
        override val vendor = Vendor.OpenBlas
        override val implementations = operations
    }

    @Test
    fun `exported operations with work qualify for exact timing`() {
        val binding = backend(setOf(EngineImplementation.Gemm))
        val matrix = DenseMatrix.zero(2, 2)
        assertNull(exactArmRejection(binding, EngineImplementation.Gemm, listOf(matrix, matrix, matrix)))
    }

    @Test
    fun `missing exports are rejected before timing`() {
        val binding = backend(emptySet())
        assertNotNull(exactArmRejection(binding, EngineImplementation.Gemmt))
    }

    @Test
    fun `empty destinations and vectors perform no vendor work`() {
        val binding = backend(setOf(EngineImplementation.Gemm, EngineImplementation.Dot))
        assertNotNull(exactArmRejection(binding, EngineImplementation.Gemm, listOf(DenseMatrix.zero(0, 0))))
        assertNotNull(exactArmRejection(binding, EngineImplementation.Dot, vectors = listOf(DenseVector.wrap(DoubleArray(0)))))
    }

    @Test
    fun `empty product depth uses shared destination scaling without vendor work`() {
        val binding = backend(setOf(EngineImplementation.Gemm))
        assertNotNull(exactArmRejection(binding, EngineImplementation.Gemm,
            listOf(DenseMatrix.zero(2, 0), DenseMatrix.zero(0, 2), DenseMatrix.zero(2, 2))))
    }
}
