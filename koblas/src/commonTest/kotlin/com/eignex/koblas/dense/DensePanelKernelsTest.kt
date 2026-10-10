package com.eignex.koblas.dense

import kotlin.test.Test

/**
 * The panel contract, on the portable backend and on whichever one this platform selected.
 *
 * The two are checked against the same written-out definitions rather than against each other, so a backend
 * that agrees with the portable one only because it delegates to it still has to be right.
 */
class DensePanelKernelsTest {
    @Test
    fun `each panel implementation agrees with the scalar oracle`() = withPanelKernels {
        assertPanelKernelsAgreeWithReference(it)
    }

    @Test
    fun `each panel implementation preserves no read and zero evaluation rules`() = withPanelKernels {
        assertPanelContractHolds(it)
    }

    @Test
    fun `an empty panel reads nothing`() = withPanelKernels {
        assertEmptyExtentsReadNothing(it)
    }

    @Test
    fun `each panel implementation stays inside its windows`() = withPanelKernels {
        assertPanelsStayInsideTheirWindows(it)
    }

    @Test
    fun `each panel implementation recommends a usable grouping`() = withPanelKernels {
        assertExecutionGroupIsUsable(it)
    }

    /**
     * An address past what a word of twenty-one bits holds, on every backend.
     *
     * One test rather than three, because it allocates buffers wide enough to reach past that bound and the
     * arithmetic in it is a handful of entries.
     */
    @Test
    fun `the indexed panels address past a packed bound`() {
        withPanelKernels(::assertIndexedPanelsAddressPastAPackedBound)
    }
}
