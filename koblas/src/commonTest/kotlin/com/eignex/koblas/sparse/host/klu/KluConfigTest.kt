package com.eignex.koblas.sparse.host.klu

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs

class KluConfigTest {

    @Test
    fun `rejects an invalid pivot tolerance`() {
        assertFailsWith<IllegalArgumentException> { KluConfig(pivotTolerance = 1.1) }
    }

    @Test
    fun `rejects a nonpositive memory growth`() {
        assertFailsWith<IllegalArgumentException> { KluConfig(memoryGrowth = 0.0) }
    }

    @Test
    fun `a null KLU factor is singular only when the status says so`() {
        // KLU answers a singular matrix and an out-of-memory the same way, with a null pointer, so the
        // status is the only thing separating them. The native binding read neither and called every
        // failure singular, telling callers a well-formed matrix was singular when KLU had simply failed.
        assertEquals(
            KluFactorOutcome.Singular,
            kluFactorOutcome(symbolicNull = false, numericNull = true, status = KLU_SINGULAR),
        )
        assertEquals(
            KluFactorOutcome.Singular,
            kluFactorOutcome(symbolicNull = true, numericNull = true, status = KLU_SINGULAR),
        )
        assertEquals(
            KluFactorOutcome.Factored,
            kluFactorOutcome(symbolicNull = false, numericNull = false, status = 0),
        )

        val analyzeFailed = kluFactorOutcome(symbolicNull = true, numericNull = true, status = -2)
        val factorFailed = kluFactorOutcome(symbolicNull = false, numericNull = true, status = -2)

        assertIs<KluFactorOutcome.Failed>(analyzeFailed)
        assertIs<KluFactorOutcome.Failed>(factorFailed)
        assertEquals("klu_analyze failed with status -2", analyzeFailed.message)
        assertEquals("klu_factor failed with status -2", factorFailed.message)
    }
}
