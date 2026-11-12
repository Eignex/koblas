package com.eignex.koblas.bench

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.*

class CalibrationCaptureTest {
    @Test fun `paired captures retain ratios hashes and provenance`() = fixture { root ->
        rounds(root)

        val capture = readCalibration(listOf(root), "native", "onemkl", "onemkl")

        assertEquals(listOf(0.5, 0.5, 0.5), capture.timings.map { it.ratio })
        assertEquals(setOf("dot+32+uniform"), capture.cases)
        assertEquals(10, capture.hashes.size)
        assertTrue(capture.provenance.isNotEmpty())
    }

    @Test fun `capture rejects smoke dirty mismatched sources and missing pairs`() {
        for (failure in listOf("smoke", "dirty", "source", "case", "profile", "cpu")) fixture { root ->
            rounds(root)
            val metadata = root.resolve("pass-2/metadata.txt")
            when (failure) {
                "smoke" -> rewrite(metadata, "requested_samples=3", "requested_samples=1")
                "dirty" -> rewrite(metadata, "dirty=false", "dirty=true")
                "source" -> rewrite(metadata, "source_commit=${"a".repeat(40)}", "source_commit=${"b".repeat(40)}")
                "case" -> rewrite(root.resolve("pass-2/onemkl.csv"), "dot+32", "dot+64")
                "cpu" -> rewrite(metadata, "profile_cpu_model=A", "profile_cpu_model=B")
            }

            assertFailsWith<IllegalArgumentException>(failure) {
                readCalibration(listOf(root), "native", "onemkl", engineName = if (failure == "profile") "different" else null)
            }
        }
    }

    @Test fun `csv preserves quoted fields and rejects repeated cases`() {
        val csv = "case,provider\n\"gemm,quoted\",\"host\"\n"

        assertEquals("host", readTimingCsv(csv).getValue("gemm,quoted")["provider"])
        assertFailsWith<IllegalArgumentException> { readTimingCsv(csv + "\"gemm,quoted\",host\n") }
    }

    @Test fun `scalar baseline does not require or compare the SIMD arm`() = fixture { root ->
        rounds(root, jvm = true)
        val owned = readCalibration(listOf(root), "jvm-owned", "onemkl-jvm", "onemkl")
        for (round in 1..3) Files.delete(root.resolve("pass-$round/jvm-simd.csv"))

        val scalar = readCalibration(listOf(root), "jvm-scalar", "onemkl-jvm", "onemkl")

        assertEquals(listOf(2.5, 2.5, 2.5), owned.timings.map { it.ratio })
        assertEquals(listOf(0.5, 0.5, 0.5), scalar.timings.map { it.ratio })
    }

    @Test fun `a scalar baseline rejects vendor capabilities`() = fixture { root ->
        rounds(root, jvm = true)
        rewrite(root.resolve("pass-2/metadata.txt"), "caps[portable:Dot]", "caps[openblas:Dot]")

        assertFailsWith<IllegalArgumentException> {
            readCalibration(listOf(root), "jvm-scalar", "onemkl-jvm", "onemkl")
        }
    }

    private fun rewrite(path: Path, before: String, after: String) = Files.writeString(path, Files.readString(path).replace(before, after))
    private fun fixture(block: (Path) -> Unit) {
        val root = Files.createTempDirectory("koblas-tuning")
        try { block(root) } finally { root.toFile().deleteRecursively() }
    }
    private fun rounds(root: Path, jvm: Boolean = false) {
        val times = if (jvm) mapOf("jvm-scalar" to 10, "jvm-simd" to 2, "onemkl-jvm" to 5) else mapOf("native" to 10, "onemkl" to 5)
        Files.writeString(root.resolve("source-hashes.txt"), "${"a".repeat(64)} koblas/src/commonMain/kernel.kt\n")
        for (round in 1..3) {
            val pass = Files.createDirectory(root.resolve("pass-$round"))
            val metadata = """
                status=complete
                [capture]
                source_commit=${"a".repeat(40)}
                dirty=false
                requested_samples=3
                target_ns=200000000
                warmups=20
                requested_jvm_forks=2
                jvm_gc=serial
                jvm_heap=512m
                cpu_policy=unbound
                koblas_threads=1
                platform=Linux fixture kernel
            """.trimIndent() + times.keys.joinToString("") { target -> """

                [$target]
                runtime=Native
                harness=native
                warmup_target_ns=50000000
                implementation=$target/fixture${if (target.startsWith("onemkl")) "/oneMKL/libmkl.so/test/threads=1" else ""}/caps[${if (target == "jvm-scalar") "portable" else if (target == "onemkl-jvm") "onemkl" else target}:Dot]
                profile_runtime=${if (jvm) "jvm" else "native"}
                profile_os=Linux
                profile_arch=X86_64
                profile_cpu_model=A
                profile_cpu_class=Avx2
                profile_sve_vector_bits=*
            """.trimIndent().let { "\n$it\n" } }
            Files.writeString(pass.resolve("metadata.txt"), metadata)
            for ((name, time) in times) Files.writeString(pass.resolve("$name.csv"),
                "case,status,samples,median_ns,provider,routing_key,timing_mode\n" +
                    "dot+32+uniform,ok,3,${time},${if (name == "onemkl-jvm") "onemkl" else name},Dot:32:1:1:false:1:true,arithmetic\n")
        }
    }
}
