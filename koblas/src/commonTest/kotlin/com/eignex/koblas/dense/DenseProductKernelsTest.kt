package com.eignex.koblas.dense

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
    fun `each product block agrees with the scalar oracle`() = withProductKernels {
        assertProductBlockAgreesWithReference(it)
    }

    @Test
    fun `an empty product block reads nothing`() = withProductKernels {
        assertEmptyProductBlockReadsNothing(it)
    }

    @Test
    fun `a zero beta overwrites a poisoned product block`() = withProductKernels {
        assertZeroBetaOverwritesPoison(it)
    }

    @Test
    fun `each product block accumulates retained depth slices`() = withProductKernels {
        assertDepthSlicesAccumulate(it)
    }

    /** A tile geometry is a positive rectangle, whatever a backend chose it to be. */
    @Test
    fun `every tile geometry is positive`() {
        withProductKernels { kernels ->
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
        withProductKernels { kernels ->
            assertTrue(!kernels.packsProduct(1, 1, 1), "${kernels.name} packed a single product")
            assertTrue(!kernels.packsProduct(4096, 1, 4096), "${kernels.name} packed a matrix-vector product")
            assertTrue(!kernels.packsProduct(1, 4096, 4096), "${kernels.name} packed a vector-matrix product")
            assertTrue(kernels.packsProduct(256, 256, 256), "${kernels.name} declined a large square product")
        }
    }
}
