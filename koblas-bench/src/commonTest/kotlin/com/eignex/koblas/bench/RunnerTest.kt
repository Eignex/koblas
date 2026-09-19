package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RunnerTest {
    @Test
    fun `runner rejects malformed numeric options instead of using defaults`() {
        for (option in listOf("warmups", "samples", "target-ms", "forks")) {
            for (value in listOf("", "many", "1.5", "999999999999999999999999")) {
                assertFailsWith<IllegalArgumentException>("$option=$value") {
                    parseArguments(arrayOf("--mode=native", "--$option=$value"))
                }
            }
        }
    }

    @Test
    fun `runner accepts an explicit targeted suite`() {
        val defaults = parseArguments(arrayOf("--mode=jvm-scalar"))
        val sweep = parseArguments(
            arrayOf("--mode=native", "--suite=sweep", "--operation=dot", "--warmups=0", "--samples=1", "--target-ms=1"),
        )

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
        for (id in listOf("dot+4096+uniform", "gemm+15x7x31+uniform")) {
            for (mode in listOf("jvm-scalar", "jvm-simd", "native")) {
                val original = Cases.parse(id).single()
                val shared = Cases.parse("$id+suite=default,sweep").single()
                val settings = settings().copy(mode = mode)
                val before = measurement(original, settings, 1, 2.5, "ok", "direct", "arithmetic", "scalar/dot")
                val after = measurement(
                    shared,
                    settings.copy(suite = "sweep"),
                    1,
                    2.5,
                    "ok",
                    "direct",
                    "arithmetic",
                    "scalar/dot",
                )

                assertEquals(reportCsv(listOf(before)), reportCsv(listOf(after)))
            }
        }
    }

    @Test
    fun `report summarizes samples across forks without run metadata`() {
        val case = Cases.parse("dot+4+uniform").single()
        val measurements = listOf(
            measurement(case, settings(), 1, 2.5, "ok", "direct", "arithmetic", "scalar/dot"),
            measurement(case, settings(), 2, 3.5, "ok", "direct", "arithmetic", "scalar/dot"),
        )

        val records = reportCsv(measurements).trim().lines()

        assertEquals(listOf(CSV_HEADER, "dot+4+uniform,ok,direct,arithmetic,scalar/dot,2,2,3.0,2.5,3.5"), records)
    }

    @Test
    fun `unsupported case has no timing`() {
        val case = Cases.parse("dot+4+uniform").single()
        val measurement = measurement(case, settings(), 0, null, "unsupported", "unsupported", "arithmetic")

        val records = reportCsv(listOf(measurement)).trim().lines()

        assertEquals(listOf(CSV_HEADER, "dot+4+uniform,unsupported,unsupported,arithmetic,unavailable,0,0,,,"), records)
    }

    /** A timed row that cannot say which call produced it is a bug, not a row with a missing field. */
    @Test
    fun `a timed record without a route is refused rather than reported`() {
        val case = Cases.parse("dot+4+uniform").single()

        val failure = assertFailsWith<IllegalStateException> {
            measurement(case, settings(), 1, 2.5, "ok", "direct", "arithmetic")
        }

        assertEquals(true, "carries no route" in failure.message.orEmpty(), failure.message)
    }

    private fun settings() = Settings("jvm-scalar", "all", "cases.txt", "output.csv", 0, 1, 1_000_000, 2)
}
