package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RunnerTest {
    @Test
    fun `runner accepts an explicit targeted suite`() {
        val defaults = parseArguments(arrayOf("--mode=jvm-scalar"))
        val sweep = parseArguments(arrayOf("--mode=native", "--suite=sweep", "--operation=dot", "--warmups=0", "--samples=1", "--target-ms=1"))

        assertEquals("default", defaults.suite)
        assertEquals("sweep", sweep.suite)
        assertEquals("dot", sweep.operation)
        assertEquals(1_000_000L, sweep.targetNanos)
        for (args in listOf(arrayOf("--suite=sweep"), arrayOf("--suite=other"), arrayOf("--suite=default", "--suite=sweep"))) {
            assertFailsWith<IllegalArgumentException> { parseArguments(arrayOf("--mode=native") + args) }
        }
    }

    @Test
    fun `suite metadata does not change report comparison fields`() {
        for (id in listOf("dot+4096+uniform", "gemm-tile+3x2x31+uniform+packed=4x4")) {
            for (mode in listOf("jvm-scalar", "jvm-c-raw-scalar", "native-raw-scalar") +
                if ("packed=" in id) emptyList() else listOf("jvm-c", "native")) {
                val original = Cases.parse(id).single()
                val shared = Cases.parse("$id+suite=default,sweep").single()
                val settings = settings().copy(mode = mode)
                val before = measurement(original, settings, 1, 2.5, "ok", "direct", "arithmetic")
                val after = measurement(shared, settings.copy(suite = "sweep"), 1, 2.5, "ok", "direct", "arithmetic")

                assertEquals(reportCsv(listOf(before)), reportCsv(listOf(after)))
            }
        }
    }

    @Test
    fun `report summarizes samples across forks without run metadata`() {
        val case = Cases.parse("dot+4+uniform").single()
        val measurements = listOf(
            measurement(case, settings(), 1, 2.5, "ok", "direct", "arithmetic"),
            measurement(case, settings(), 2, 3.5, "ok", "direct", "arithmetic"),
        )

        val records = reportCsv(measurements).trim().lines()

        assertEquals(listOf(CSV_HEADER, "dot+4+uniform,ok,direct,arithmetic,policy,2,2,3.0,2.5,3.5"), records)
    }

    @Test
    fun `unsupported case has no timing`() {
        val case = Cases.parse("dot+4+uniform").single()
        val measurement = measurement(case, settings(), 0, null, "unsupported", "unsupported", "arithmetic")

        val records = reportCsv(listOf(measurement)).trim().lines()

        assertEquals(listOf(CSV_HEADER, "dot+4+uniform,unsupported,unsupported,arithmetic,unavailable,0,0,,,"), records)
    }

    @Test
    fun `packed case records omit derived metadata`() {
        val case = Cases.parse("gemm-tile+3x2x31+uniform+packed=4x4").single()
        val measurement = measurement(case, settings(), 1, 2.5, "ok", "partial", "raw-tile")

        val records = reportCsv(listOf(measurement)).trim().lines()

        assertEquals("gemm-tile+3x2x31+uniform+packed=4x4,ok,partial,raw-tile,portable-tile,1,1,2.5,2.5,2.5",
            records[1])
    }

    private fun settings() = Settings("jvm-scalar", "all", "cases.txt", "output.csv", 0, 1, 1_000_000, 2)
}
