package com.eignex.koblas.bench

import com.eignex.koblas.vendor.hostPlatform
import com.eignex.koblas.vendor.OperatingSystem
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class PlatformTest {
    @Test
    fun `unix report directories support literal backslashes`() {
        if (hostPlatform().operatingSystem == OperatingSystem.Windows) return
        val path = "build/bench-file-${Random.nextInt().toUInt()}/a\\b/report.csv"
        val report = "case,ns\nfixture,12\n"

        writeTextFile(path, report)
        val actual = readTextFile(path)

        assertEquals(report, actual)
    }

    @Test
    fun `report files support spaces and unicode in parent directory names`() {
        val path = "build/bench-file-${Random.nextInt().toUInt()}/r\u00e4kna resultat/report.csv"
        val report = "case,ns\nfixture,12\n"

        writeTextFile(path, report)
        val actual = readTextFile(path)

        assertEquals(report, actual)
    }
}
