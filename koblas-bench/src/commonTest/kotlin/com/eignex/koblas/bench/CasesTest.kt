package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class CasesTest {
    @Test
    fun `suite membership preserves workload identity`() {
        for (id in listOf("dot+4096+uniform", "trsm+15x7+triangular+side=R+uplo=U+transA=T+diag=U")) {
            val original = Cases.parse(id).single()
            for (membership in listOf("default", "sweep", "default,sweep", "sweep,default")) {
                val tagged = Cases.parse("$id+suite=$membership").single()
                assertEquals(original.id, tagged.id)
                assertEquals(original.options, tagged.options)
                assertEquals(original.dimensions, tagged.dimensions)
                assertFailsWith<IllegalArgumentException> { Cases.parse("$id\n$id+suite=$membership") }
            }
        }
    }

    @Test
    fun `selection intersects suite and operation`() {
        val cases = Cases.parse("""
            dot+4096+uniform+suite=default,sweep
            sum+4096+uniform
            dot+7+uniform+suite=sweep
            sum+7+uniform+suite=sweep
        """.trimIndent())

        assertEquals(listOf("dot+4096+uniform", "sum+4096+uniform"), Cases.select(cases).map { it.id })
        assertEquals(listOf("dot+4096+uniform"), Cases.select(cases, operation = "dot").map { it.id })
        assertEquals(listOf("dot+4096+uniform", "dot+7+uniform"), Cases.select(cases, "sweep", "dot").map { it.id })
        for ((suite, operation) in listOf("sweep" to "all", "unknown" to "dot", "sweep" to "axpy", "default" to "unknown")) {
            assertFailsWith<IllegalArgumentException> { Cases.select(cases, suite, operation) }
        }
    }

    @Test
    fun `parser rejects invalid suite membership`() {
        for (suffix in listOf("", "all", "default,default", "sweep,", ",sweep", "default,other", "default+suite=sweep")) {
            assertFailsWith<IllegalArgumentException> { Cases.parse("dot+7+uniform+suite=$suffix") }
        }
    }

    @Test
    fun `parser accepts canonical cases`() {
        val cases = Cases.parse("""
            # comment
            dot+32+uniform
            gemm+32x21x48+uniform+transA=T
            spdot+4096+sparse-uniform+density=0.01
        """.trimIndent())

        assertEquals(listOf("dot", "gemm", "spdot"), cases.map { it.operation })
        assertEquals("gemm+32x21x48+uniform+transA=T", cases[1].id)
    }

    @Test
    fun `parser rejects malformed and duplicate cases`() {
        assertFailsWith<IllegalArgumentException> { Cases.parse("unknown+4+uniform") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("dot++4+uniform") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("gemm+4x4+uniform") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("dot+1000001+uniform") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("dot+4+uniform+transA=T") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("spdot+4096+sparse-uniform+density=0.01+transA=T") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("spdot+4096+uniform+density=0.01") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("spdot+4096+sparse-uniform") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("gemm+4x4x4+triangular") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("gemm-tile+9x4x32+uniform") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("dot+4096+uniform+packed=4x4") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("dot+4+uniform\ndot+4+uniform") }
    }

    @Test
    fun `vector timings reject unrelated boundaries`() {
        for (operation in listOf("scal+64+uniform", "spgather+64+sparse-uniform+density=0.25")) {
            assertEquals("arithmetic", Cases.parse("$operation+timing=arithmetic").single().options["timing"])
            for (timing in listOf("pack-plus-compute", "prepacked-compute", "reset-and-arithmetic", "unknown")) {
                assertFailsWith<IllegalArgumentException> { Cases.parse("$operation+timing=$timing") }
            }
        }
    }

    @Test
    fun `option order does not change case identity`() {
        val canonical = "trsm+15x7+triangular+side=R+uplo=U+transA=T+diag=U"
        val reordered = "trsm+15x7+triangular+diag=U+transA=T+uplo=U+side=R"

        val expected = Cases.parse(canonical).single()
        val actual = Cases.parse(reordered).single()

        assertEquals(expected, actual)
        assertFailsWith<IllegalArgumentException> { Cases.parse("$canonical\n$reordered") }
    }


    @Test
    fun `fixtures have stable golden digests`() {
        assertEquals("173cc3a80546954d", digest(Fixtures.vector(16, 1)))
        val sparse = Fixtures.sparse(17, 5, 0.2, 2)
        assertEquals("e6ac2de9cae9ebe8", digest(sparse.values))
        assertEquals(listOf(0, 3, 6, 9, 12, 15), sparse.copyColumnPointers().toList())
    }
}
