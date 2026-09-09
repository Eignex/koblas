package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparseBenchmarkArmTest {
    @Test
    fun `built in arm reports exact sparse implementation identity`() {
        val arm = SparseBenchmarkArm.resolve(BUILTIN_BACKEND)

        assertEquals(BUILTIN_BACKEND, arm.name)
        assertTrue(arm.identity.startsWith("built-in/built-in/"), arm.identity)
        assertEquals("single calling thread", arm.threading)
    }

    @Test
    fun `unknown sparse arm is rejected`() {
        assertFailsWith<IllegalStateException> { SparseBenchmarkArm.resolve("available") }
    }

    @Test
    fun `resources close in reverse order and only once`() {
        val closed = mutableListOf<String>()
        val resources = BenchmarkResources()
        resources.own(AutoCloseable { closed += "first" })
        resources.own(AutoCloseable { closed += "second" })

        resources.close()
        resources.close()

        assertEquals(listOf("second", "first"), closed)
    }

    @Test
    fun `failed acquisition closes resources already owned`() {
        val closed = mutableListOf<String>()
        val resources = BenchmarkResources()
        resources.own(AutoCloseable { closed += "first" })

        assertFailsWith<IllegalArgumentException> {
            resources.acquire<AutoCloseable> { throw IllegalArgumentException("failed preparation") }
        }

        assertEquals(listOf("first"), closed)
        assertFailsWith<IllegalStateException> { resources.own(AutoCloseable {}) }
    }

    @Test
    fun `owned prepared arm is unusable after resource cleanup`() {
        val arm = SparseBenchmarkArm.resolve(BUILTIN_BACKEND)
        val resources = BenchmarkResources()
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
        val prepared = resources.acquire { arm.prepare(matrix) }
        val result = DoubleArray(2)

        prepared.gemv(1.0, doubleArrayOf(4.0, 5.0), 0.0, result)
        resources.close()

        assertFailsWith<IllegalStateException> {
            prepared.gemv(1.0, doubleArrayOf(1.0, 1.0), 0.0, result)
        }
    }
}
