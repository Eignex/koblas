package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

class JvmRunnerTest {
    @Test
    fun `measured forks preserve admission for dense and sparse engine workloads`() {
        val cases = Cases.parse("""
            gemm+15x7x31+uniform
            dot+7+uniform
            spdot+16+sparse-uniform+density=0.25
        """.trimIndent())
        val path = Files.createTempFile("koblas-admission-", ".txt")
        try {
            Files.writeString(path, cases.joinToString("\n") { it.id })
            val exact = if (KoblasEngine.simd != null) "jvm-simd" else "jvm-scalar"
            val engines = mapOf("jvm-default" to koblas, exact to (KoblasEngine.simd ?: KoblasEngine.scalar))
            for ((mode, engine) in engines) {
                for (case in cases) {
                    val expected = benchmarkArm(case, engine, vendor = false).work!!
                    val actual = JvmBenchmarkBridge.create(mode, case.id, path.toString())
                    assertEquals(expected.comparisonKind, actual.comparisonKind, case.id)
                    assertEquals(expected.kernel, actual.kernel, case.id)
                    assertEquals(expected.run(), actual.run(), case.id)
                }
            }
        } finally {
            Files.deleteIfExists(path)
        }
    }

}
