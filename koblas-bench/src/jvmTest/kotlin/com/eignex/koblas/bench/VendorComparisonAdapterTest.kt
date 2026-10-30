package com.eignex.koblas.bench

import com.eignex.lab.bench.BenchRequest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class VendorComparisonAdapterTest {
    @Test fun `openblas is placed on the requested architecture`() {
        for (arch in listOf("x86_64", "arm64")) {
            val case = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "openblas", "arch" to arch))).single()

            assertEquals(setOf("platform:linux", "arch:$arch"), case.resources.requiredTags)
            assertEquals("comparison/openblas-$arch/all", case.id)
        }
    }

    @Test fun `onemkl and accelerate keep their fixed hosts and case ids`() {
        val onemkl = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl"))).single()
        val accelerate = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "accelerate"))).single()

        assertEquals(setOf("platform:linux", "arch:x86_64"), onemkl.resources.requiredTags)
        assertEquals("comparison/onemkl/all", onemkl.id)
        assertEquals(setOf("platform:mac", "arch:arm64"), accelerate.resources.requiredTags)
    }

    @Test fun `only openblas takes an architecture`() {
        assertFailsWith<IllegalArgumentException> {
            VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "arch" to "arm64")))
        }
    }

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

    @Test fun `the native engine is an accepted comparison target`() {
        val request = BenchRequest(mapOf("vendor" to "openblas", "targets" to "jvm-simd,native,openblas,openblas-jvm"))

        VendorComparisonSuite.plan(request)
    }
}
