package com.eignex.koblas.sparse

import com.eignex.koblas.sparse.factorization.lu.MAX_CANDIDATE_COLS
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The compiled-in sparse defaults, pinned for the reason the dense ones are: collecting them must not move
 * a number a benchmark chose. A run whose environment sets one of the keys will fail here, which is the
 * intended reading.
 */
class SparseTuningTest {
    @Test
    fun `the sparse crossovers keep their measured values`() {
        assertEquals(256, SparseTuning.dotDenseCCrossover)
        assertEquals(4, SparseTuning.luMaxCandidateColumns)
        assertEquals(0.1, SparseTuning.reachableFtranMaxDensity)
    }

    @Test
    fun `the elimination reads the candidate width the collection resolved`() {
        assertEquals(SparseTuning.luMaxCandidateColumns, MAX_CANDIDATE_COLS)
    }
}
