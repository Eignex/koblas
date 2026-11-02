@file:Suppress("LongParameterList") // test panels implement the array window seam

package com.eignex.koblas.dense

import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.dense.PortablePanelKernels
import kotlin.test.Test
import kotlin.test.assertTrue

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
    fun `each panel implementation preserves no read and sparse structure rules`() = withPanelKernels {
        assertPanelContractHolds(it)
    }

    @Test
    fun `zero skipping dense panels satisfy the numerical contract`() {
        val kernels = ZeroSkippingPanels()

        assertPanelKernelsAgreeWithReference(kernels)
        assertPanelContractHolds(kernels)
        assertEmptyExtentsReadNothing(kernels)
        assertPanelsStayInsideTheirWindows(kernels)

        assertTrue(kernels.skippedCoefficients > 0, "the conformance checks exercised no zero coefficient")
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

    /** A valid dense zero shortcut; indexed leaves retain their separate stored-position behavior. */
    private class ZeroSkippingPanels : KoblasEngine() {
        override val name = "zero-skipping"
        override val implementations = setOf(EngineImplementation.Panels)
        var skippedCoefficients: Int = 0
            private set

        override fun columnUpdate(
            alpha: Double,
            a: DoubleArray,
            aOffset: Int,
            lda: Int,
            x: DoubleArray,
            xOffset: Int,
            xStride: Int,
            rows: Int,
            columns: Int,
            y: DoubleArray,
            yOffset: Int,
            yStride: Int,
        ) {
            if (rows <= 0 || columns <= 0) return
            for (column in 0 until columns) {
                if (x[xOffset + column * xStride] == 0.0) {
                    skippedCoefficients++
                } else {
                    PortablePanelKernels.columnUpdate(
                        alpha, a, aOffset + column * lda, lda, x, xOffset + column * xStride, xStride,
                        rows, 1, y, yOffset, yStride,
                    )
                }
            }
        }

        override fun rankUpdate(
            alpha: Double,
            a: DoubleArray,
            aOffset: Int,
            lda: Int,
            x: DoubleArray,
            xOffset: Int,
            xStride: Int,
            rows: Int,
            columns: Int,
            coefficients: DoubleArray,
            coefficientOffset: Int,
            coefficientStride: Int,
        ) {
            if (rows <= 0 || columns <= 0) return
            for (column in 0 until columns) {
                if (coefficients[coefficientOffset + column * coefficientStride] == 0.0) {
                    skippedCoefficients++
                } else {
                    PortablePanelKernels.rankUpdate(
                        alpha, a, aOffset + column * lda, lda, x, xOffset, xStride, rows, 1,
                        coefficients, coefficientOffset + column * coefficientStride, coefficientStride,
                    )
                }
            }
        }
    }
}
