package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The compiled-in dense defaults, pinned so that collecting them in one file cannot quietly move a number
 * that a benchmark chose. Each is the value its routine used before the collection.
 *
 * These read the resolved values, so a run whose environment sets one of the tuning keys will fail here.
 * That is the intended reading: the suite measures the library as built, and a box that has retuned it is
 * not testing the same library.
 */
class DenseTuningTest {
    @Test
    fun `the level three block keeps its measured shape`() {
        assertEquals(256, DenseTuning.level3BlockRows)
        assertEquals(8, DenseTuning.level3BlockColumns)
        assertEquals(128, DenseTuning.level3BlockDepth)
        assertEquals(64, DenseTuning.triangularBlock)
        assertEquals(32, DenseTuning.transposeBlock)
        assertEquals(16, DenseTuning.trmmPackedMinOrder)
        assertEquals(32, DenseTuning.trmmPackedMinRows)
    }

    @Test
    fun `every bundled C crossover keeps its measured length`() {
        assertEquals(128, DenseTuning.jvmCDotCrossover)
        assertEquals(128, DenseTuning.jvmCSumCrossover)
        assertEquals(256, DenseTuning.jvmCSsqdCrossover)
        assertEquals(128, DenseTuning.jvmCNrm2Crossover)
        assertEquals(128, DenseTuning.jvmCAsumCrossover)
        assertEquals(512, DenseTuning.jvmCDot4Crossover)
        assertEquals(64, DenseTuning.jvmCAxpy4Crossover)
        assertEquals(256, DenseTuning.jvmCDotAxpyCrossover)
        assertEquals(16, DenseTuning.jvmCGemmTileCrossover)
        assertEquals(16, DenseTuning.jvmCGemmTrsmTileCrossover)
    }

    @Test
    fun `the compiled in kernel thresholds keep their measured lengths`() {
        assertEquals(512, DenseTuning.symvFourColumnCrossover)
        assertEquals(32, DenseTuning.simdUnrollMinVectors)
        assertEquals(48, DenseTuning.nativeCMinLength)
    }

    @Test
    fun `the triangular block still fits the mask that indexes it`() {
        requireTriangularBlockFitsMask()

        assertEquals(TRIANGULAR_BLOCK, DenseTuning.triangularBlock)
    }

    @Test
    fun `the routines read the values the collection resolved`() {
        assertEquals(DenseTuning.level3BlockRows, LEVEL3_BLOCK_ROWS)
        assertEquals(DenseTuning.level3BlockColumns, LEVEL3_BLOCK_COLUMNS)
        assertEquals(DenseTuning.level3BlockDepth, LEVEL3_BLOCK_DEPTH)
    }
}
