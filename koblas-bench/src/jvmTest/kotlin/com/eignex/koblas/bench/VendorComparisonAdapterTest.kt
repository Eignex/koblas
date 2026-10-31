package com.eignex.koblas.bench

import com.eignex.lab.bench.BenchRequest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class VendorComparisonAdapterTest {
    @Test fun `shards plan one case each on the same host`() {
        val cases = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "shards" to "3")))

        assertEquals(listOf("comparison/onemkl/all/1-of-3", "comparison/onemkl/all/2-of-3", "comparison/onemkl/all/3-of-3"),
            cases.map { it.id })
        assertEquals(listOf("1", "2", "3"), cases.map { it.parameters["shard"] })
        assertEquals(setOf(setOf("platform:linux", "arch:x86_64")), cases.map { it.resources.requiredTags }.toSet())
    }

    @Test fun `a shard outside its count is rejected`() {
        assertFailsWith<IllegalArgumentException> {
            VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "shards" to "2", "shard" to "3")))
        }
    }

    @Test fun `an unsharded request keeps its single case id`() {
        val case = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "accelerate"))).single()

        assertEquals("comparison/accelerate/all", case.id)
    }
}
