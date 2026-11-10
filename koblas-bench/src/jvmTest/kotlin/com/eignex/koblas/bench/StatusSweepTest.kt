package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.kernels.PACKED_PRODUCT_MINIMUM_WORK
import com.eignex.koblas.internal.scheduling.dense.*
import com.eignex.lab.bench.BenchRequest
import java.lang.reflect.Modifier
import java.security.MessageDigest
import kotlin.test.*

class StatusSweepTest {
    private val parameters = mapOf("status-sweep" to "true", "samples" to "3", "warmups" to "3",
        "target-ms" to "100", "forks" to "1", "rounds" to "1")

    private fun workloads(profile: String = "status-v3"): List<BenchCase> =
        KoblasLabSuite.plan(BenchRequest(parameters + ("sweep-profile" to profile)))
            .map { Cases.parse(it.parameters.getValue("row")).single() }

    @Test fun `core sweep retains routine families and existing case identities within a bounded inventory`() {
        val routines = KoblasEngine::class.java.declaredMethods
            .filter { !Modifier.isStatic(it.modifiers) && !it.name.startsWith("get") && '$' !in it.name }.map { it.name }.toSet()
        val cases = workloads()
        val names = mapOf("scale" to "scal", "scatter" to "spscatter-raw", "gather" to "spgather",
            "gatherZero" to "spgather-zero", "dotColumn" to "spdot-column")
        val covered = cases.map { it.operation }.toSet()

        assertTrue(routines.all { names.getOrElse(it) { it } in covered },
            "Review representative sweep membership when the engine adds a routine family")
        assertTrue(cases.size <= 150, "Keep the per commit change detector bounded")
        assertTrue(cases.map { it.id }.containsAll(workloads("status-v2").map { it.id }))
    }

    @Test fun `core sweep samples vector boundaries and different storage traversals`() {
        val cases = workloads()

        assertTrue(cases.filter { it.operation == "dot" }.map { it.dimension(0) }.containsAll(listOf(31, 32, 33)))
        assertTrue(cases.filter { it.operation == "iamax" }.map { it.dimension(0) }.containsAll(listOf(255, 256, 257)))
        assertTrue(cases.mapNotNull { it.options["stride"] }.toSet().containsAll(setOf("2", "-1", "-2")))
        val indexed = cases.filter { it.operation == "spdot-raw" }
            .map { Fixtures.storedCount(it.dimension(0), it.options.getValue("density").toDouble()) }
        assertTrue(indexed.containsAll(listOf(15, 16, 17)))
    }

    @Test fun `core sweep samples packing and schedule boundaries with alternate orientations`() {
        val cases = workloads()
        val products = cases.filter { it.operation == "gemm" }
        val work = products.map { it.dimensions.fold(1L) { product, extent -> product * extent } }

        assertTrue(work.any { it < PACKED_PRODUCT_MINIMUM_WORK })
        assertTrue(PACKED_PRODUCT_MINIMUM_WORK in work)
        assertTrue(work.any { it > PACKED_PRODUCT_MINIMUM_WORK })
        val crossing = products.filter { it.dimension(0) > PRODUCT_BLOCK_ROWS &&
            it.dimension(1) > PRODUCT_BLOCK_COLUMNS && it.dimension(2) > PRODUCT_BLOCK_DEPTH }
        assertTrue(crossing.any { it.options["layout"] == "borrowed" && "workspace" !in it.options })
        assertTrue(crossing.any { it.options["layout"] == "borrowed" && it.options["workspace"] == "retained" })
        for (operation in listOf("trsm", "trmm")) {
            assertTrue(cases.any { it.operation == operation && it.option("side", "L") == "L" &&
                it.dimension(0) > TRIANGULAR_DIAGONAL_BLOCK && it.dimension(1) < 16 })
            assertTrue(cases.any { it.operation == operation && it.option("side", "L") == "R" &&
                it.dimension(1) > TRIANGULAR_DIAGONAL_BLOCK && it.dimension(0) < 16 })
        }
        for ((option, value) in listOf("transA" to "T", "transB" to "T", "uplo" to "U", "diag" to "U")) {
            assertTrue(cases.any { !it.operation.startsWith("sp") && it.options[option] == value })
        }
    }

    @Test fun `core sweep samples sparse distributions and densities`() {
        val cases = workloads().filter { it.operation.startsWith("sp") }

        assertTrue(cases.map { it.option("support", "uniform") }.toSet()
            .containsAll(setOf("uniform", "banded", "skewed", "empty", "mixed")))
        assertTrue(cases.map { it.options.getValue("density").toDouble() }.toSet().containsAll(setOf(0.01, 0.125, 0.5)))
        assertTrue(cases.any { it.operation == "spmm-right" })
        assertTrue(cases.any { it.operation == "spmm" && it.options["transB"] == "T" })
    }

    @Test fun `extended profile retains core cases and adds larger working sets`() {
        val cases = workloads("extended-v1")

        assertTrue(cases.size <= 200)
        assertTrue(cases.map { it.id }.containsAll(workloads().map { it.id }))
        assertTrue(cases.any { it.dimension(0) > 4194304 })
        assertTrue(cases.any { it.dimension(0) > 1048576 && it.options["stride"] == "-2" })
    }

    @Test fun `versioned inventories have frozen canonical membership`() {
        for ((profile, digest) in mapOf(
            "status-v2" to "3981114dc3cef318040af672fdd0e3b0cba1d65e81a2b9ad33f505165db4605e",
            "status-v3" to "c4428dd54bf3b022e18550e48cd3a8426c9816fd82e4c9d50be78e8edfc22982",
            "extended-v1" to "7ca7cbe4f085166c69cfc1303c491c37058bc2eb4ca53b5fd3641a942c8a5be8",
        )) {
            val cases = workloads(profile)
            val bytes = cases.joinToString("\n") { it.id }.toByteArray()
            val actual = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            assertEquals(digest, actual, "Version the profile when changing frozen membership")
        }
    }

    @Test fun `status sweep keeps fixed timing controls and versioned regression series`() {
        val cases = KoblasLabSuite.plan(BenchRequest(parameters))

        for (case in cases) {
            assertEquals("jvm-default", case.parameters["targets"])
            assertEquals("status-v3", case.parameters["sweep-profile"])
            assertEquals("metric-v1", case.attributes["regressionPolicy"])
            assertEquals("koblas-status-v3", case.attributes["suite"])
            assertEquals(60L, case.resources.timeoutSeconds)
            assertTrue(case.inputs.none { it.role == "executable" })
        }
    }

    @Test fun `extended comparisons keep all requested arms on each workload`() {
        for ((platform, arch, targets) in listOf(Triple("mac", "arm64", "jvm-default,jvm-scalar,jvm-simd"),
            Triple("linux", "x86_64", "jvm-default,jvm-scalar,jvm-simd,native,native-default"))) {
            val cases = KoblasLabSuite.plan(BenchRequest(mapOf("sweep-profile" to "extended-v1",
                "platform" to platform, "arch" to arch, "targets" to targets)))

            assertTrue(cases.all { it.parameters["targets"] == targets && it.attributes.isEmpty() })
            assertTrue(cases.all { it.resources.requiredTags == setOf("platform:$platform", "arch:$arch") })
            assertEquals(platform == "linux", cases.all { it.inputs.any { input -> input.role == "executable" } })
        }
    }

    @Test fun `engine sweep keeps all engines in the workload and prepares the native executable`() {
        val targets = "jvm-default,jvm-scalar,jvm-simd,native,native-default"

        val cases = KoblasLabSuite.plan(BenchRequest(parameters + mapOf("targets" to targets,
            "platform" to "mac", "arch" to "arm64")))

        assertEquals(136, cases.size)
        for (case in cases) {
            assertEquals(targets, case.parameters["targets"])
            assertEquals("median_ns,median_ns/jvm-scalar,median_ns/jvm-simd,median_ns/native,median_ns/native-default",
                case.attributes["regressionMetrics"])
            assertEquals("koblas-status-v3-engines-v1", case.attributes["suite"])
            assertTrue(case.inputs.single { it.role == "executable" }.path.contains("macosArm64"))
        }
    }

    @Test fun `status sweep refuses controls that would mix measurement series`() {
        for (override in listOf(mapOf("rounds" to "2"), mapOf("samples" to "1"), mapOf("targets" to "jvm-scalar"),
            mapOf("smoke" to "true"), mapOf("cases" to "koblas-bench/cases.txt"),
            mapOf("sweep-profile" to "unknown"))) {
            assertFailsWith<IllegalArgumentException> { KoblasLabSuite.plan(BenchRequest(parameters + override)) }
        }
        assertFailsWith<IllegalArgumentException> {
            KoblasLabSuite.plan(BenchRequest(mapOf("sweep-profile" to "status-v3", "cases" to "koblas-bench/cases.txt")))
        }
    }
}
