package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The product block contract, on the portable backend and on whichever one this platform selected.
 *
 * Both are checked against the written-out definition and against packed panels built from the contract's
 * own index formula, so a backend that agrees with the portable one only because it delegates to it still
 * has to be right.
 */
class DenseProductKernelsTest {
    @Test
    fun `the portable tile agrees with the written out definition`() {
        assertProductBlockAgreesWithReference(PortableProductKernels)
    }

    @Test
    fun `the selected tile agrees with the written out definition`() {
        assertProductBlockAgreesWithReference(koblas.productKernels)
    }

    @Test
    fun `the candidate tile agrees with the written out definition`() {
        val candidate = BuiltinEngines.simd ?: return skipped()
        assertProductBlockAgreesWithReference(candidate.productKernels)
    }

    @Test
    fun `the portable tile keeps the empty and no read rules`() {
        assertEmptyProductBlockReadsNothing(PortableProductKernels)
        assertZeroBetaOverwritesPoison(PortableProductKernels)
    }

    @Test
    fun `the selected tile keeps the empty and no read rules`() {
        assertEmptyProductBlockReadsNothing(koblas.productKernels)
        assertZeroBetaOverwritesPoison(koblas.productKernels)
    }

    @Test
    fun `the candidate tile keeps the empty and no read rules`() {
        val candidate = BuiltinEngines.simd ?: return skipped()
        assertEmptyProductBlockReadsNothing(candidate.productKernels)
        assertZeroBetaOverwritesPoison(candidate.productKernels)
    }

    @Test
    fun `the portable tile accumulates depth slices of a retained panel`() {
        assertDepthSlicesAccumulate(PortableProductKernels)
    }

    @Test
    fun `the selected tile accumulates depth slices of a retained panel`() {
        assertDepthSlicesAccumulate(koblas.productKernels)
    }

    @Test
    fun `the candidate tile accumulates depth slices of a retained panel`() {
        val candidate = BuiltinEngines.simd ?: return skipped()
        assertDepthSlicesAccumulate(candidate.productKernels)
    }

    /** A tile geometry is a positive rectangle, whatever a backend chose it to be. */
    @Test
    fun `every tile geometry is positive`() {
        for (kernels in listOf(PortableProductKernels, koblas.productKernels)) {
            assertTrue(kernels.tileRows > 0 && kernels.tileColumns > 0, "${kernels.name} has no tile")
            assertTrue(kernels.name.isNotEmpty(), "a tile with no name cannot be attributed")
        }
    }

    /**
     * A product too small or too thin to hide a copy is not packed, and a large square one is.
     *
     * The rule is the backend's, so what is asserted is its shape rather than its constants: an extent below
     * one tile cannot fill the tiles it would pack into, and a matrix-vector shaped call has no arithmetic to
     * amortise a copy over however large its other extent is.
     */
    @Test
    fun `packing is declined where there is nothing to amortise it over`() {
        for (kernels in listOf(PortableProductKernels, koblas.productKernels)) {
            assertTrue(!kernels.packsProduct(1, 1, 1), "${kernels.name} packed a single product")
            assertTrue(!kernels.packsProduct(4096, 1, 4096), "${kernels.name} packed a matrix-vector product")
            assertTrue(!kernels.packsProduct(1, 4096, 4096), "${kernels.name} packed a vector-matrix product")
            assertTrue(kernels.packsProduct(256, 256, 256), "${kernels.name} declined a large square product")
        }
    }

    private fun skipped() {
        println("SKIPPED: no Vector API product candidate on this runtime; its tiles were not exercised")
    }
}
