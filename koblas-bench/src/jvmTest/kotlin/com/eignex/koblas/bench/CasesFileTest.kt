package com.eignex.koblas.bench

import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CasesFileTest {
    @Test
    fun `dense sweeps share the default case and span bounded explicit sizes`() {
        val cases = Cases.parse(Files.readString(Path.of("cases.txt")))
        for (operation in listOf("dot", "sum", "asum")) {
            val defaults = Cases.select(cases, operation = operation)
            val sweep = Cases.select(cases, "sweep", operation)
            assertEquals(listOf(4096), defaults.map { it.dimension(0) })
            assertEquals(defaults, sweep.intersect(defaults.toSet()).toList())
            assertTrue(sweep.size in 12..15)
            assertEquals(1, sweep.minOf { it.dimension(0) })
            assertEquals(262144, sweep.maxOf { it.dimension(0) })
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
        for ((suite, operation) in listOf("sweep" to "all", "unknown" to "dot", "sweep" to "axpy", "default" to "unknown")) {
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

        assertEquals("ead2eb8b54f39a7e622f1e2403f7f3bd532cd96a6dc5b2c7b4528a1df0fd4597", digest)
    }

    @Test
    fun `shared workload is canonical and bounded`() {
        val cases = Cases.parse(Files.readString(Path.of("cases.txt")))

        assertEquals(setOf("dot", "sum", "asum"),
            cases.filter { "sweep" in it.suites }.map { it.operation }.toSet())
        assertEquals(39, cases.count { "sweep" in it.suites })
        assertEquals(108, cases.size)
        assertEquals(72, Cases.select(cases).size)
        assertEquals(cases.size, cases.map { it.id }.toSet().size)
    }
}
