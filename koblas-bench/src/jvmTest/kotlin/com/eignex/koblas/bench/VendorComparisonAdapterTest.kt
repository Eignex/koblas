package com.eignex.koblas.bench

import com.eignex.lab.bench.BenchRequest
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class VendorComparisonAdapterTest {
    @Test fun `openblas is placed on the requested architecture`() {
        for (arch in listOf("x86_64", "arm64")) {
            val cases = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "openblas", "arch" to arch)))

            assertEquals(setOf(setOf("platform:linux", "arch:$arch")), cases.map { it.resources.requiredTags }.toSet())
            assertTrue(cases.all { it.id.startsWith("comparison/openblas-$arch/") })
        }
    }

    @Test fun `onemkl and accelerate keep their fixed hosts`() {
        val onemkl = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl")))
        val accelerate = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "accelerate")))

        assertEquals(setOf(setOf("platform:linux", "arch:x86_64")), onemkl.map { it.resources.requiredTags }.toSet())
        assertEquals(setOf(setOf("platform:mac", "arch:arm64")), accelerate.map { it.resources.requiredTags }.toSet())
    }

    @Test fun `only openblas takes an architecture`() {
        assertFailsWith<IllegalArgumentException> {
            VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "arch" to "arm64")))
        }
    }

    @Test fun `every selected row lands in exactly one routine and size band case`() {
        val grid = "koblas-bench/lab/cases-parity-level2.txt"
        val selected = Cases.select(Cases.parse(repositoryFile(grid)))

        val cases = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "cases" to grid)))

        val rows = cases.flatMap { case -> case.parameters.getValue("rows").lines().map { case.id to it } }
        assertEquals(selected.size, rows.size)
        assertEquals(cases.size, cases.map { it.id }.distinct().size)
        for ((id, row) in rows) {
            val parsed = Cases.parse(row).single()
            assertEquals("comparison/onemkl/${parsed.operation}/${sizeBand(parsed)}", id, row)
        }
    }

    @Test fun `an operation selects only its own cells`() {
        val cases = VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "operation" to "dot")))

        assertTrue(cases.isNotEmpty() && cases.all { it.id.startsWith("comparison/onemkl/dot/") })
    }

    @Test fun `a request cannot supply its own rows`() {
        assertFailsWith<IllegalArgumentException> {
            VendorComparisonSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "rows" to "dot+4+uniform")))
        }
    }

    @Test fun `jvm options take only hotspot flags`() {
        val accepted = mapOf("vendor" to "onemkl", "operation" to "dot", "jvm-options" to "-XX:MaxVectorSize=32 -XX:-UseFMA")

        VendorComparisonSuite.plan(BenchRequest(accepted))

        for (rejected in listOf("-cp /tmp", "-XX:MaxVectorSize=32;rm", "-Dkoblas=1", "")) {
            assertFailsWith<IllegalArgumentException>(rejected) {
                VendorComparisonSuite.plan(BenchRequest(accepted + ("jvm-options" to rejected)))
            }
        }
    }

    @Test fun `size bands follow the parity thresholds`() {
        for ((line, band) in listOf(
            "dot+4095+uniform" to "small", "dot+4096+uniform" to "mid", "dot+65535+uniform" to "large",
            "gemv+127x127+uniform" to "small", "gemv+128x128+uniform" to "mid", "gemv+16384x4+uniform" to "large",
            "trsv+127+triangular+uplo=L+transA=N+diag=N" to "small",
            "trsv+256+triangular+uplo=L+transA=N+diag=N" to "large",
        )) {
            assertEquals(band, sizeBand(Cases.parse(line).single()), line)
        }
    }

    @Test fun `the native engine is an accepted comparison target`() {
        val request = BenchRequest(mapOf("vendor" to "openblas", "targets" to "jvm-simd,native,openblas,openblas-jvm"))

        VendorComparisonSuite.plan(request)
    }

    private fun repositoryFile(path: String): String =
        generateSequence(Path.of("").toAbsolutePath()) { it.parent }
            .map { it.resolve(path) }.first { Files.isRegularFile(it) }.let(Files::readString)
}
