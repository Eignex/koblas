package com.eignex.koblas.bench

import com.eignex.koblas.engine.*
import kotlin.test.*
import java.nio.file.Files
import java.nio.file.Path

class EngineTuningTest {
    @Test fun `generated grids are unique disjoint and parseable`() {
        val training = CalibrationCases.training(true)
        val holdout = CalibrationCases.holdout(true)

        assertEquals(training.size, training.toSet().size)
        assertEquals(holdout.size, holdout.toSet().size)
        assertTrue(training.toSet().intersect(holdout.toSet()).isEmpty())
        Cases.parse((training + holdout).joinToString("\n"))
    }

    @Test fun `losing larger lengths delay a vector threshold`() {
        val data = listOf(32 to 0.5, 64 to 1.2, 128 to 0.7, 256 to 0.8).flatMap { (n, cost) ->
            List(3) { timing(n, cost) }
        }

        val rule = fitProfile("onemkl", data).single()

        assertEquals(128..Int.MAX_VALUE, rule.rows)
    }

    @Test fun `one cpu model cannot hide another model losing`() {
        val data = List(30) { timing(128, 0.5) } + List(3) { timing(128, 1.1).copy(cpu = "B") }

        assertTrue(fitProfile("onemkl", data).isEmpty())
    }

    @Test fun `a winning variant cannot hide a losing variant of the same shape`() {
        val data = List(3) { timing(128, 0.5) } +
            List(3) { timing(128, 1.1).copy(case = "dot+128+uniform+stride=1") }

        assertTrue(fitProfile("onemkl", data).isEmpty())
    }

    @Test fun `matrix regions exclude skinny losses and separate borrowed layouts`() {
        fun matrix(n: Int, m: Int, k: Int, ratio: Double, borrowed: Boolean = false) =
            timing(n, ratio).copy(shape = RoutingShape(EngineImplementation.Gemm, n, m, k, borrowed, 3, true))
        val data = listOf(matrix(8, 8, 8, 1.5), matrix(4, 512, 512, 1.5), matrix(32, 32, 32, 0.5),
            matrix(64, 64, 64, 0.5), matrix(32, 32, 32, 1.5, true))

        val rule = fitProfile("onemkl", data).single()

        assertEquals(32..Int.MAX_VALUE, rule.rows)
        assertEquals(32768L, rule.work)
        assertEquals(false, rule.borrowed)
    }

    @Test fun `composition orders vendors using overlapping paired regions`() {
        fun capture(cost: Double) = CalibrationCapture(emptyList(), setOf("dot"), List(3) { timing(128, cost) }, emptyMap(), emptyList())
        val captures = mapOf("onemkl" to capture(0.8), "openblas" to capture(0.5))

        val rules = composeProfiles(captures)

        assertEquals(listOf("openblas", "onemkl"), rules.map { it.source })
    }

    @Test fun `emitted configuration is kotlin with typed capabilities`() {
        val code = renderProfile("fixture", listOf("native", "Linux", "X86_64", "*", "Avx2", "*"),
            listOf(EngineRule("onemkl", setOf(EngineImplementation.Dot), rows = 128..Int.MAX_VALUE)))

        assertTrue("EngineRuntime.Native" in code)
        assertTrue("CpuClass.Avx2" in code)
        assertTrue("setOf(Dot)" in code)
        assertFalse("tsv" in code)
    }

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
                readCalibration(listOf(root), "native", "onemkl", profileName = if (failure == "profile") "different" else null)
            }
        }
    }

    @Test fun `csv preserves quoted fields and rejects repeated cases`() {
        val csv = "case,provider\n\"gemm,quoted\",\"host\"\n"

        assertEquals("host", readTimingCsv(csv).getValue("gemm,quoted")["provider"])
        assertFailsWith<IllegalArgumentException> { readTimingCsv(csv + "\"gemm,quoted\",host\n") }
    }

    private fun timing(n: Int, ratio: Double) = PairedTiming("dot+$n+uniform", "A", "onemkl",
        RoutingShape(EngineImplementation.Dot, n, 1, 1, false, 1, true), ratio)
    private fun rewrite(path: Path, before: String, after: String) = Files.writeString(path, Files.readString(path).replace(before, after))
    private fun fixture(block: (Path) -> Unit) {
        val root = Files.createTempDirectory("koblas-tuning")
        try { block(root) } finally { root.toFile().deleteRecursively() }
    }
    private fun rounds(root: Path) {
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
            """.trimIndent() + listOf("native", "onemkl").joinToString("") { target -> """

                [$target]
                runtime=Native
                harness=native
                warmup_target_ns=50000000
                implementation=$target/fixture${if (target == "onemkl") "/oneMKL/libmkl.so/test/threads=1" else ""}/caps[$target:Dot]
                profile_runtime=native
                profile_os=Linux
                profile_arch=X86_64
                profile_cpu_model=A
                profile_cpu_class=Avx2
                profile_sve_vector_bits=*
            """.trimIndent().let { "\n$it\n" } }
            Files.writeString(pass.resolve("metadata.txt"), metadata)
            for ((name, time) in mapOf("native" to 10, "onemkl" to 5)) Files.writeString(pass.resolve("$name.csv"),
                "case,status,samples,median_ns,provider,routing_key,timing_mode\n" +
                    "dot+32+uniform,ok,3,$time,$name,Dot:32:1:1:false:1:true,arithmetic\n")
        }
    }
}
