package com.eignex.koblas.bench

import com.eignex.lab.bench.BenchRequest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class VendorComparisonAdapterTest {
    @Test
    fun `the plan reserves requested cores for the capture`() {
        val request = BenchRequest(mapOf("vendor" to "onemkl", "reservation-cpus" to "16"))

        val case = VendorComparisonSuite.plan(request).single()

        assertEquals(16, case.resources.cpus)
    }

    @Test
    fun `nonpositive and invalid reservations are rejected`() {
        for (value in listOf("0", "-1", "many", "2147483648")) {
            val request = BenchRequest(mapOf("vendor" to "onemkl", "reservation-cpus" to value))

            assertFailsWith<IllegalArgumentException> { VendorComparisonSuite.plan(request) }
        }
    }
}
