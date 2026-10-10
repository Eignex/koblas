package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SparseFactoryTest {
    @Test
    fun `triplet construction verifies its result and is measured on the portable arm`() {
        for (operation in listOf("spbuild-triplets", "spbuild-duplicates")) {
            val case = Cases.parse("$operation+17x9+sparse-uniform+density=0.2+support=empty").single()
            val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work)

            assertEquals("construction", work.timingMode)
            assertEquals("portable-csc/ofTriplets", work.kernel)
            assertTrue(work.run().isFinite())
            BuiltinEngines.simd?.let { assertNull(sparseArm(case, it)?.work) }
        }
    }
}
