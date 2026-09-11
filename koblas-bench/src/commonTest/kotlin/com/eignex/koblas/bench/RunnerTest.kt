package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RunnerTest {
    @Test
    fun `report shares metadata and preserves samples across forks`() {
        val case = Cases.parse("dot+4+uniform").single()
        val settings = settings()
        val measurements = listOf(
            measurement(case, "jvm-scalar", settings, 1, 10, 25, "2.5", "ok", "direct", "arithmetic", "runtime, \"build\""),
            measurement(case, "jvm-scalar", settings, 2, 20, 70, "3.5", "ok", "direct", "arithmetic", "runtime, \"build\""),
        )

        val records = reportCsv(measurements).lines()

        assertEquals(1, records.count { it.startsWith("run,1,") })
        assertEquals(1, records.count { it.startsWith("case,1,") })
        assertTrue(records.single { it.startsWith("run,1,") }.contains("\"runtime, \"\"build\"\"\""))
        assertEquals(listOf("sample,1,1,1,10,25,2.5", "sample,1,2,2,20,70,3.5"), records.filter { it.startsWith("sample,1,") })
    }

    @Test
    fun `unsupported case has metadata without an empty sample`() {
        val case = Cases.parse("dot+4+uniform").single()
        val measurement = measurement(case, "jvm-scalar", settings(), 0, 0, 0, "", "unsupported", "unsupported", "arithmetic", "runtime")

        val records = reportCsv(listOf(measurement)).lines()

        assertTrue(records.single { it.startsWith("case,1,") }.contains(",unsupported,"))
        assertEquals(0, records.count { it.startsWith("sample,1,") })
    }

    @Test
    fun `packed case records omit derived metadata`() {
        val case = Cases.parse("gemm-tile+3x2x31+uniform+packed=4x4").single()
        val measurement = measurement(case, "jvm-scalar", settings(), 1, 10, 25, "2.5", "ok", "partial", "raw-tile", "runtime")

        val records = reportCsv(listOf(measurement)).lines()

        assertEquals("case,1,1,gemm-tile+3x2x31+uniform+packed=4x4,ok,partial,raw-tile,portable-tile",
            records.single { it.startsWith("case,1,") })
    }

    private fun settings() = Settings("jvm-scalar", "all", "cases.txt", "output.csv", 0, 1, 1_000_000, 2, "1", "source", "false")
}
