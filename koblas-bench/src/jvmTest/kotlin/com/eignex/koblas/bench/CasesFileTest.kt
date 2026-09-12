package com.eignex.koblas.bench

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals

class CasesFileTest {
    @Test
    fun `shared workload is canonical and bounded`() {
        val cases = Cases.parse(Files.readString(Path.of("cases.txt")))

        assertEquals(215, cases.size)
        assertEquals(cases.size, cases.map { it.id }.toSet().size)
    }

    @Test
    fun `sparse slice comparisons are an exact workload subset`() {
        val cases = Cases.parse(Files.readString(Path.of("cases.txt"))).map { it.id }.toSet()
        val slices = Cases.parse(Files.readString(Path.of("sparse-slices-cases.txt")))

        assertEquals(26, slices.size)
        assertEquals(emptyList(), slices.map { it.id }.filterNot(cases::contains))
    }

    @Test
    fun `reference smoke cases are an exact workload subset`() {
        val cases = Cases.parse(Files.readString(Path.of("cases.txt"))).map { it.id }.toSet()
        val smoke = Cases.parse(Files.readString(Path.of("smoke-cases.txt")))

        assertEquals(11, smoke.size)
        assertEquals(emptyList(), smoke.map { it.id }.filterNot(cases::contains))
    }
}
