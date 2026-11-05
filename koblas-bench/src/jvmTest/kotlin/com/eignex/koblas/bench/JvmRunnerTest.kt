package com.eignex.koblas.bench

import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.koblas
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

class JvmRunnerTest {
    @Test
    fun `the generic product is built by the measured fork on the default and an exact arm`() {
        // The bridge reads its case file, but its admission rule is independent of the benchmark workload size.
        val case = Cases.parse("gemm+15x7x31+uniform").single()
        val exact = if (BuiltinEngines.simd != null) "jvm-simd" else "jvm-scalar"
        val engines = mapOf("jvm-default" to koblas, exact to (BuiltinEngines.simd ?: BuiltinEngines.scalar))
        val path = Files.createTempFile("koblas-generic-case-", ".txt")
        try {
            Files.writeString(path, case.id)
            for ((mode, engine) in engines) {
                val work = JvmBenchmarkBridge.create(mode, case.id, path.toString())
                try {
                    assertEquals(ENGINE_COMPARISON, work.comparisonKind, mode)
                    assertEquals("${engine.name}/${case.operation}", work.kernel, mode)
                } finally {
                    work.close()
                }
            }
        } finally {
            Files.deleteIfExists(path)
        }
    }
}
