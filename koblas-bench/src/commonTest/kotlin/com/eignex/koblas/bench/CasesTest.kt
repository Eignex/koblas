package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class CasesTest {
    @Test
    fun `parser accepts canonical cases`() {
        val cases = Cases.parse("""
            # comment
            dot+32+uniform
            gemm+32x21x48+uniform+transA=T
            spgemv+257x129+sparse-uniform+density=0.01+mode=prepared
        """.trimIndent())

        assertEquals(listOf("dot", "gemm", "spgemv"), cases.map { it.operation })
        assertEquals("gemm+32x21x48+uniform+transA=T", cases[1].id)
    }

    @Test
    fun `parser rejects malformed and duplicate cases`() {
        assertFailsWith<IllegalArgumentException> { Cases.parse("unknown+4+uniform") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("gemm+4x4+uniform") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("gemm+4x4x4+uniform+transB=T+transA=T") }
        assertFailsWith<IllegalArgumentException> { Cases.parse("dot+4+uniform\ndot+4+uniform") }
    }

    @Test
    fun `fixtures have stable golden digests`() {
        assertEquals("173cc3a80546954d", digest(Fixtures.vector(16, 1)))
        val sparse = Fixtures.sparse(17, 5, 0.2, 2)
        assertEquals("e6ac2de9cae9ebe8", digest(sparse.values))
        assertEquals(listOf(0, 3, 6, 9, 12, 15), sparse.copyColumnPointers().toList())
    }
}
