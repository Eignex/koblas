package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertEquals

class DirectoryPathsTest {
    @Test
    fun `windows paths preserve their drive and network roots`() {
        assertEquals(listOf("C:/reports", "C:/reports/round"), parentDirectories("C:\\reports\\round\\out.csv", windows = true))
        assertEquals(listOf("//server/share/reports"), parentDirectories("\\\\server\\share\\reports\\out.csv", windows = true))
        assertEquals(emptyList(), parentDirectories("C:/out.csv", windows = true))
        assertEquals(emptyList(), parentDirectories("//server/share/out.csv", windows = true))
    }

    @Test
    fun `unix backslashes remain literal directory characters`() {
        assertEquals(listOf("reports", "reports/a\\b"), parentDirectories("reports/a\\b/out.csv"))
        assertEquals(listOf("/tmp", "/tmp/reports"), parentDirectories("//tmp/reports/out.csv"))
    }

    @Test
    fun `unix and relative parents retain their existing addressing`() {
        assertEquals(listOf("/tmp", "/tmp/reports"), parentDirectories("/tmp/reports/out.csv"))
        assertEquals(listOf("reports", "reports/round"), parentDirectories("reports/round/out.csv"))
        assertEquals(emptyList(), parentDirectories("out.csv"))
        assertEquals(emptyList(), parentDirectories("/out.csv"))
    }
}
