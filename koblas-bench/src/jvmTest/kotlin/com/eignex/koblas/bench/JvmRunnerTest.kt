package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class JvmRunnerTest {
    @Test
    fun `measured forks preserve admission for sparse and default policy workloads`() {
        val cases = Cases.parse("""
            gemm+15x7x31+uniform
            dot+7+uniform
            spdot+16+sparse-uniform+density=0.25
            spdot-view+16+sparse-uniform+density=0.25
            copy-overlap+7+uniform
        """.trimIndent())
        val path = Files.createTempFile("koblas-admission-", ".txt")
        try {
            Files.writeString(path, cases.joinToString("\n") { it.id })
            val exact = if (KoblasEngine.simd != null) "jvm-simd" else "jvm-scalar"
            val engines = mapOf("jvm-default" to koblas, exact to (KoblasEngine.simd ?: KoblasEngine.scalar))
            for ((mode, engine) in engines) {
                for (case in cases) {
                    val admission = assertNotNull(benchmarkArm(case, engine, vendor = false))
                    val expected = admission.work
                    if (expected == null) {
                        val failure = assertFailsWith<IllegalStateException> {
                            JvmBenchmarkBridge.create(mode, case.id, path.toString())
                        }
                        assertTrue(assertNotNull(admission.reason) in failure.message.orEmpty())
                    } else {
                        val actual = JvmBenchmarkBridge.create(mode, case.id, path.toString())
                        try {
                            assertEquals(expected.comparisonKind, actual.comparisonKind, case.id)
                            assertEquals(expected.kernel, actual.kernel, case.id)
                            assertEquals(expected.run(), actual.run(), case.id)
                        } finally {
                            expected.close()
                            actual.close()
                        }
                    }
                }
            }
        } finally {
            Files.deleteIfExists(path)
        }
    }

}
