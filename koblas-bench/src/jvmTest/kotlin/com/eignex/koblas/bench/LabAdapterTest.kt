package com.eignex.koblas.bench

import com.eignex.lab.bench.BenchRequest
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LabAdapterTest {
    @Test fun `native capture declares the binary and compatible platform`() {
        val case = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "native", "smoke" to "true"))).single()
        assertEquals(setOf("platform:linux", "arch:x86_64"), case.resources.requiredTags)
        assertEquals(case.parameters.getValue("native-executable"), case.inputs.single().path)
    }

    @Test fun `invalid capture targets are refused before execution`() {
        assertFailsWith<IllegalArgumentException> { KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "typo"))) }
    }

    @Test fun `failed prepared captures retain partial CSV and logs`() {
        val root = Files.createTempDirectory("koblas-partial")
        try {
            val java = root.resolve("java")
            Files.writeString(java, "#!/usr/bin/env bash\nfor arg in \"\$@\"; do case \"\$arg\" in --output=*) printf 'case,ns\\npartial,12\\n' > \"\${arg#--output=}\";; esac; done\necho failure-log\nexit 7\n")
            check(java.toFile().setExecutable(true))
            val output = root.resolve("report")
            val script = File("capture-report.sh").absoluteFile
            val process = ProcessBuilder("bash", script.path, "--targets", "jvm-scalar", "--libraries", "openblas", "--prepared", "--retain-incomplete", "--smoke", "--output", output.toString())
                .redirectErrorStream(true).apply { environment()["KOBLAS_BENCH_JAVA"] = java.toString(); environment()["KOBLAS_BENCH_CLASSPATH"] = "unused" }.start()
            val log = process.inputStream.bufferedReader().readText()
            assertEquals(7, process.waitFor(), log)
            assertTrue(Files.readString(output.resolve("jvm-scalar.csv")).contains("partial,12"))
            assertEquals("status=incomplete", Files.readAllLines(output.resolve("metadata.txt")).first())
            assertTrue(Files.readString(output.resolve("jvm-scalar.log")).contains("failure-log"))
        } finally { root.toFile().deleteRecursively() }
    }
}
