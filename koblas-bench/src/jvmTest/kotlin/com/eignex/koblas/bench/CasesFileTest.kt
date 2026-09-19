package com.eignex.koblas.bench

import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CasesFileTest {
    @Test
    fun `dense sweeps share a default case and span bounded explicit sizes`() {
        val cases = Cases.parse(Files.readString(Path.of("cases.txt")))
        for (operation in SWEPT) {
            val defaults = Cases.select(cases, operation = operation).toSet()
            val sweep = Cases.select(cases, "sweep", operation)
            // Sharing a workload with the default suite is what keeps a sweep comparable to an ordinary
            // capture; a sweep of sizes nothing else runs would be its own incomparable scale. 4096 is the
            // width every operation carries in both suites, so it is the one that has to be shared.
            assertTrue(
                sweep.any { it.dimension(0) == 4096 && it in defaults },
                "$operation's sweep does not share its 4096 case with the default suite",
            )
            assertTrue(sweep.size in 12..28, "$operation swept ${sweep.size} sizes")
            assertTrue(sweep.minOf { it.dimension(0) } <= 8, "$operation's sweep starts above the call overhead")
            assertEquals(262144, sweep.maxOf { it.dimension(0) }, operation)
        }
    }

    @Test
    fun `capture selection agrees with Kotlin across suites and smoke limits`() {
        val cases = Cases.parse(Files.readString(Path.of("cases.txt")))
        for ((suite, operation) in listOf("default" to "all", "default" to "dot", "sweep" to "dot", "sweep" to "sum")) {
            for (smoke in listOf(false, true)) {
                val process = ProcessBuilder("awk", "-v", "suite=$suite", "-v", "operation=$operation", "-v", "smoke=$smoke",
                    "-f", "select-cases.awk", "cases.txt").redirectErrorStream(true).start()
                val output = process.inputStream.bufferedReader().readText()
                assertEquals(0, process.waitFor(), output)
                val expected = Cases.select(cases, suite, operation).let { if (smoke) it.take(3) else it }
                assertEquals(expected, Cases.parse(output))
            }
        }
    }

    @Test
    fun `capture rejects invalid and empty selections`() {
        // gemm has no sweep, so asking for one is the empty intersection this rejects.
        for ((suite, operation) in listOf("sweep" to "all", "unknown" to "dot", "sweep" to "gemm", "default" to "unknown")) {
            val process = ProcessBuilder("awk", "-v", "suite=$suite", "-v", "operation=$operation",
                "-f", "select-cases.awk", "cases.txt").redirectErrorStream(true).start()
            val output = process.inputStream.bufferedReader().readText()
            assertEquals(2, process.waitFor(), output)
        }
    }

    @Test
    fun `capture rejects malformed membership even after smoke limit`() {
        for (suffix in listOf("", "all", "default,default", "sweep,", ",sweep", "default,other", "default+suite=sweep")) {
            val process = ProcessBuilder("awk", "-v", "suite=default", "-v", "operation=all", "-v", "smoke=true",
                "-f", "select-cases.awk").redirectErrorStream(true).start()
            process.outputStream.bufferedWriter().use {
                it.write("dot+1+uniform\ndot+2+uniform\ndot+3+uniform\ndot+7+uniform+suite=$suffix\n")
            }
            val output = process.inputStream.bufferedReader().readText()
            assertEquals(2, process.waitFor(), output)
        }
    }

    @Test
    fun `default workload retains historical identities`() {
        val defaults = Cases.select(Cases.parse(Files.readString(Path.of("cases.txt"))))
        val ids = defaults.joinToString("\n") { it.id }
        val digest = MessageDigest.getInstance("SHA-256").digest(ids.toByteArray()).joinToString("") { "%02x".format(it) }

        assertEquals("cf1e00bdd07656d8ea6f70fcc7e5a45cf48c7504b2efc9e40413c071505aef99", digest)
    }

    @Test
    fun `shared workload is canonical and bounded`() {
        val cases = Cases.parse(Files.readString(Path.of("cases.txt")))

        assertEquals(SWEPT, cases.filter { "sweep" in it.suites }.map { it.operation }.toSet())
        assertEquals(168, cases.count { "sweep" in it.suites })
        assertEquals(256, cases.size)
        // The default suite is what an ordinary capture runs, and the sweeps did not enlarge it.
        assertEquals(104, Cases.select(cases).size)
        assertEquals(cases.size, cases.map { it.id }.toSet().size)
    }

    private companion object {
        /** Every Level 1 operation with a crossover to locate, which is every one a vendor implements. */
        val SWEPT = setOf("dot", "sum", "asum", "nrm2", "iamax", "axpy", "scal", "swap", "rot")
    }
}
