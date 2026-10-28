package com.eignex.koblas.bench

import com.eignex.lab.bench.*
import com.eignex.lab.bench.BenchCase as LabCase
import kotlinx.serialization.encodeToString
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
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

    @Test fun `capture controls are frozen with compatible resources`() {
        val parameters = mapOf("targets" to "jvm-simd,jvm-scalar", "rounds" to "3", "jvm-gc" to "zgc",
            "jvm-heap" to "1g", "memory-mib" to "4096", "timeout-seconds" to "120", "cpu" to "auto",
            "cases" to "koblas-bench/cases-reductions.txt", "samples" to "5", "warmups" to "0",
            "target-ms" to "200", "forks" to "2", "smoke" to "false")

        val case = KoblasLabSuite.plan(BenchRequest(parameters)).single()

        assertEquals(parameters - setOf("memory-mib", "timeout-seconds"), case.parameters)
        assertEquals(BenchResources(1, 4096, 120, setOf("platform:linux")), case.resources)
        assertEquals(listOf(BenchInput("koblas-bench/cases-reductions.txt", role = "cases")), case.inputs)
        assertEquals(case, BenchProtocol.json.decodeFromString<LabCase>(BenchProtocol.json.encodeToString(case)))
    }

    @Test fun `invalid capture controls are refused before execution`() {
        val invalid = listOf(
            mapOf("rounds" to "0"), mapOf("rounds" to "2147483648"), mapOf("rounds" to "2", "pass" to "3"),
            mapOf("samples" to "-1"), mapOf("warmups" to "-1"), mapOf("forks" to "0"),
            mapOf("jvm-gc" to "parallel"), mapOf("jvm-heap" to "1G"), mapOf("jvm-heap" to "1g"),
            mapOf("jvm-heap" to "999999999999999999999999g"), mapOf("memory-mib" to "1023"),
            mapOf("timeout-seconds" to "0"), mapOf("cpu" to "1,2"), mapOf("cpu" to "-1"),
            mapOf("vendors-only" to "true"), mapOf("vendors-only" to "yes"), mapOf("smoke" to "yes"),
            mapOf("suite" to "other"), mapOf("suite" to "sweep"), mapOf("operation" to "dot\n"),
            mapOf("targets" to "jvm-scalar,jvm-scalar"), mapOf("libraries" to "unknown"),
            mapOf("targets" to "openblas-jvm", "libraries" to "onemkl"),
            mapOf("targets" to "aocl-jvm", "libraries" to "all"),
            mapOf("targets" to "jvm-scalar", "vendors-only" to "true"),
            mapOf("targets" to "native", "jvm-heap" to "128m"), mapOf("targets" to "native", "jvm-gc" to "g1"),
            mapOf("targets" to "native", "native-executable" to "koblas-bench/build/unknown/bench"),
            mapOf("targets" to "native", "native-executable" to "koblas-bench/build/../linuxX64/bench"),
            mapOf("targets" to "native", "native-executable" to "/tmp/linuxX64/bench"),
            mapOf("targets" to "native", "native-executable" to "koblas-bench/build/bin/macosArm64/bench", "cpu" to "auto"),
            mapOf("targets" to "armpl,onemkl"), mapOf("targets" to "accelerate-jvm", "cpu" to "auto"),
            mapOf("output" to "elsewhere"), mapOf("prepared" to "true"), mapOf("retain-incomplete" to "false"),
        ) + listOf("/tmp/cases", "../cases", "koblas-bench/../cases", "a\\b", "a:b", "a//b", "./cases")
            .map { mapOf("cases" to it) }

        for (parameters in invalid) {
            assertFailsWith<IllegalArgumentException>(parameters.toString()) { KoblasLabSuite.plan(BenchRequest(parameters)) }
        }
    }

    @Test fun `vendor captures declare their libraries and hardware`() {
        for ((target, tags) in mapOf(
            "openblas-jvm" to emptySet(), "onemkl-jvm" to setOf("platform:linux", "arch:x86_64"),
            "armpl-jvm" to setOf("platform:linux", "arch:arm64"), "accelerate-jvm" to setOf("platform:mac"),
        )) {
            val case = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to target, "vendors-only" to "true"))).single()
            assertEquals(target.removeSuffix("-jvm"), case.parameters["libraries"])
            assertEquals(tags, case.resources.requiredTags)
        }
        val case = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "openblas-jvm", "libraries" to "all"))).single()
        assertEquals("all", case.parameters["libraries"])
    }

    @Test fun `custom cases resolve after relocation with literal arguments`() = withCapture { root, repository ->
        val relative = "koblas-bench/custom cases ${'$'}(touch escaped) `literal`.txt"
        Files.writeString(repository.resolve(relative), "dot+8+random\n")
        val planned = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "jvm-scalar", "cases" to relative,
            "rounds" to "2", "jvm-gc" to "serial", "jvm-heap" to "256m", "smoke" to "false", "vendors-only" to "false"))).single()
        val frozen = BenchProtocol.json.decodeFromString<LabCase>(BenchProtocol.json.encodeToString(planned))
        val relocated = Files.move(repository, root.resolve("relocated repository"))

        val result = KoblasLabSuite.run(frozen, BenchContext(root.resolve("out"), worktree = relocated))

        assertEquals(BenchStatus.SUCCESS, result.status)
        val arguments = Files.readAllLines(root.resolve("out/arguments.txt"))
        assertEquals(relocated.resolve(relative).toString(), arguments[arguments.indexOf("--cases") + 1])
        assertEquals("2", arguments[arguments.indexOf("--rounds") + 1])
        assertEquals("serial", arguments[arguments.indexOf("--jvm-gc") + 1])
        assertEquals("256m", arguments[arguments.indexOf("--jvm-heap") + 1])
        assertTrue("--smoke" !in arguments && "--vendors-only" !in arguments)
        assertEquals(1, arguments.count { it == "--output" })
        assertEquals(root.resolve("out/report").toString(), arguments[arguments.indexOf("--output") + 1])
        assertEquals("dot+8+random\n", Files.readString(root.resolve("out/report/cases.txt")))
        assertTrue(!Files.exists(relocated.resolve("escaped")))
    }

    @Test fun `repeated captures count target CSV rows once and retain every artifact`() = withCapture { root, repository ->
        val case = KoblasLabSuite.plan(BenchRequest(mapOf("rounds" to "2"))).single()

        val result = KoblasLabSuite.run(case, BenchContext(root.resolve("out"), worktree = repository))

        assertEquals(BenchStatus.SUCCESS, result.status)
        assertEquals(mapOf("captured-rows" to 4.0, "captured-csv-files" to 4.0, "completed-rounds" to 2.0),
            result.metrics.associate { it.name to it.value })
        assertTrue(result.artifacts.map { it.path }.containsAll(listOf("capture.log", "report/metadata.txt",
            "report/pass-1/jvm-scalar.csv", "report/pass-2/jvm-simd.csv", "report/pass-2/jvm-simd.log",
            "report/pass-1/sidecars/jvm-scalar.csv", "report/jvm-scalar.csv")))
        assertEquals(result.artifacts.size, result.artifacts.map { it.path }.distinct().size)
    }

    @Test fun `partial repeated captures retain completed and failed round output`() = withCapture("partial") { root, repository ->
        val case = KoblasLabSuite.plan(BenchRequest(mapOf("rounds" to "3"))).single()

        val result = KoblasLabSuite.run(case, BenchContext(root.resolve("out"), worktree = repository))

        assertEquals(BenchStatus.FAILED, result.status)
        assertEquals(mapOf("captured-rows" to 3.0, "captured-csv-files" to 3.0, "completed-rounds" to 1.0),
            result.metrics.associate { it.name to it.value })
        assertTrue(result.message.orEmpty().contains("exit 7, 1/3 rounds complete"))
        assertTrue(result.artifacts.map { it.path }.containsAll(listOf("capture.log", "report/metadata.txt",
            "report/pass-1/jvm-simd.csv", "report/pass-2/jvm-scalar.csv", "report/pass-2/jvm-scalar.log")))
        assertTrue(Files.readString(root.resolve("out/capture.log")).contains("failure-log"))
    }

    @Test fun `single captures count direct target CSV rows`() = withCapture { root, repository ->
        val case = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "openblas-jvm", "vendors-only" to "true"))).single()

        val result = KoblasLabSuite.run(case, BenchContext(root.resolve("out"), worktree = repository))

        assertEquals(BenchStatus.SUCCESS, result.status)
        assertEquals(mapOf("captured-rows" to 1.0, "captured-csv-files" to 1.0, "completed-rounds" to 1.0),
            result.metrics.associate { it.name to it.value })
        assertTrue("--vendors-only" in Files.readAllLines(root.resolve("out/arguments.txt")))
    }

    @Test fun `complete empty captures still fail`() = withCapture("empty") { root, repository ->
        val case = KoblasLabSuite.plan(BenchRequest()).single()

        val result = KoblasLabSuite.run(case, BenchContext(root.resolve("out"), worktree = repository))

        assertEquals(BenchStatus.FAILED, result.status)
        assertEquals(0.0, result.metrics.single { it.name == "captured-rows" }.value)
    }

    @Test fun `execution rejects undeclared inputs and output overrides`() = withCapture { root, repository ->
        val case = KoblasLabSuite.plan(BenchRequest(mapOf("cases" to "koblas-bench/cases.txt"))).single()
        val context = BenchContext(root.resolve("out"), worktree = repository)

        assertFailsWith<IllegalArgumentException> { KoblasLabSuite.run(case.copy(inputs = emptyList()), context) }
        assertFailsWith<IllegalArgumentException> {
            KoblasLabSuite.run(case.copy(parameters = case.parameters + ("output" to "elsewhere")), context)
        }
        assertTrue(!Files.exists(root.resolve("out/report")))
    }

    private fun withCapture(scenario: String = "complete", test: (Path, Path) -> Unit) {
        val root = Files.createTempDirectory("koblas-adapter")
        try {
            val repository = Files.createDirectories(root.resolve("repository/koblas-bench")).parent
            Files.writeString(repository.resolve("koblas-bench/capture-report.sh"), """
                #!/usr/bin/env bash
                set -eu
                scenario=$scenario
                rounds=1
                cases=
                original=("${'$'}@")
                while ((${ '$'}#)); do
                  case "${'$'}1" in
                    --prepared|--retain-incomplete|--smoke|--vendors-only) shift ;;
                    --output) output=${'$'}2; shift 2 ;;
                    --targets) targets=${'$'}2; shift 2 ;;
                    --rounds) rounds=${'$'}2; shift 2 ;;
                    --cases) cases=${'$'}2; shift 2 ;;
                    *) shift 2 ;;
                  esac
                done
                printf '%s\n' "${'$'}{original[@]}" > "${'$'}{output%/*}/arguments.txt"
                mkdir "${'$'}output"
                printf 'status=incomplete\n' > "${'$'}output/metadata.txt"
                if [[ -n ${'$'}cases ]]; then cp "${'$'}cases" "${'$'}output/cases.txt"; fi
                IFS=, read -r -a selected <<<"${'$'}targets"
                for ((round=1; round<=rounds; round++)); do
                  report=${'$'}output
                  if ((rounds > 1)); then report=${'$'}output/pass-${'$'}round; mkdir "${'$'}report"; fi
                  printf 'status=incomplete\n' > "${'$'}report/metadata.txt"
                  for target in "${'$'}{selected[@]}"; do
                    printf 'case,ns\n' > "${'$'}report/${'$'}target.csv"
                    if [[ ${'$'}scenario != empty ]]; then printf 'fixture,12\n' >> "${'$'}report/${'$'}target.csv"; fi
                    echo target-log > "${'$'}report/${'$'}target.log"
                    if [[ ${'$'}scenario == partial && ${'$'}round == 2 ]]; then echo failure-log; exit 7; fi
                  done
                  mkdir "${'$'}report/sidecars"
                  printf 'case,ns\nsidecar,12\n' > "${'$'}report/sidecars/jvm-scalar.csv"
                  printf 'status=complete\n' > "${'$'}report/metadata.txt"
                done
                if ((rounds > 1)); then printf 'case,ns\naggregate,12\n' > "${'$'}output/jvm-scalar.csv"; fi
                printf 'status=complete\n' > "${'$'}output/metadata.txt"
            """.trimIndent() + "\n")
            test(root, repository)
        } finally { root.toFile().deleteRecursively() }
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

    @Test fun `prepared rounds preserve target ordering and JVM settings`() {
        val root = Files.createTempDirectory("koblas-prepared-rounds")
        try {
            val repository = root.resolve("repository")
            val bench = Files.createDirectories(repository.resolve("koblas-bench"))
            listOf("cases.txt", "select-cases.awk").forEach { name ->
                Files.copy(File(name).toPath(), bench.resolve(name))
            }
            val java = root.resolve("java")
            Files.writeString(java, """
                #!/usr/bin/env bash
                set -eu
                for arg in "${'$'}@"; do
                  case "${'$'}arg" in --output=*) output=${'$'}{arg#--output=};; esac
                done
                printf 'case,ns\nfixture,12\n' > "${'$'}output"
                printf '%s\n' "${'$'}@" > "${'$'}{output%.csv}.args"
                echo 'resolved implementation=fixture runtime=fixture harness=fixture'
            """.trimIndent() + "\n")
            check(java.toFile().setExecutable(true))
            val output = root.resolve("report")
            val process = ProcessBuilder("bash", File("capture-report.sh").absolutePath,
                "--targets", "jvm-simd,jvm-scalar", "--libraries", "openblas", "--prepared",
                "--retain-incomplete", "--smoke", "--rounds", "2", "--jvm-gc", "serial",
                "--jvm-heap", "256m", "--output", output.toString()).redirectErrorStream(true).apply {
                environment()["KOBLAS_CAPTURE_ROOT"] = repository.toString()
                environment()["KOBLAS_BENCH_JAVA"] = java.toString()
                environment()["KOBLAS_BENCH_CLASSPATH"] = "fixture-classpath"
            }.start()
            val log = process.inputStream.bufferedReader().readText()
            assertEquals(0, process.waitFor(), log)
            val metadata = Files.readString(output.resolve("metadata.txt"))
            assertTrue(metadata.startsWith("status=complete\n"), metadata)
            assertTrue(metadata.contains("round_1_targets=jvm-simd,jvm-scalar"), metadata)
            assertTrue(metadata.contains("round_2_targets=jvm-scalar,jvm-simd"), metadata)
            for (round in 1..2) for (target in listOf("jvm-scalar", "jvm-simd")) {
                val arguments = Files.readAllLines(output.resolve("pass-$round/$target.args"))
                assertTrue("-XX:+UseSerialGC" in arguments)
                assertTrue("-Xms256m" in arguments && "-Xmx256m" in arguments)
                assertTrue("fixture-classpath" in arguments)
                assertEquals(target == "jvm-simd", "--add-modules=jdk.incubator.vector" in arguments)
                assertTrue(Files.readString(output.resolve("pass-$round/$target.csv")).contains("fixture,12"))
            }
        } finally { root.toFile().deleteRecursively() }
    }
}
