package com.eignex.koblas.internal.kernels.scalar.dense

import com.eignex.koblas.dense.assertDepthBlocksAccumulate
import com.eignex.koblas.dense.assertEmptyProductBlockReadsNothing
import com.eignex.koblas.dense.assertProductBlockAgreesWithReference
import com.eignex.koblas.dense.assertZeroBetaOverwritesPoison
import com.eignex.koblas.dense.withProductKernels
import com.eignex.koblas.koblas
import kotlin.test.Test

/**
 * The product block contract, on the portable backend and on whichever one this platform selected.
 *
 * Both are checked against the written-out definition and against packed panels built from the contract's
 * own index formula, so a backend that agrees with the portable one only because it delegates to it still
 * has to be right.
 */
class ProductKernelsTest {
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
    fun `each product block accumulates transient depth blocks`() = withProductKernels {
        assertDepthBlocksAccumulate(it)
    }
}
