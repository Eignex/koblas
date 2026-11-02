package com.eignex.koblas.bench

import com.eignex.lab.bench.*
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LabAdapterTest {
    @Test fun `generated calibration inputs permit paired native vendors`() {
        val file = Path.of("build/calibration/plan-fixture.txt")
        Files.createDirectories(file.parent)
        Files.writeString(file, "dot+64+uniform\ncopy+17+uniform+stride=2\n")
        try {
            val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "onemkl",
                "cases" to "koblas-bench/build/calibration/plan-fixture.txt", "targets" to "native,onemkl,openblas")))

            assertEquals(2, cases.size)
            assertTrue(cases.all { it.resources.memoryMiB == 2048L })
            assertTrue(cases.all { it.inputs.any { input -> input.role == "executable" && "/linuxX64/" in input.path } })
            assertTrue(cases.all { case -> case.inputs.any { it.path.endsWith("calibration/plan-fixture.txt") } })
        } finally { Files.delete(file) }
    }

    @Test fun `explicit profiles are declared inputs of every comparison row`() {
        val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "openblas",
            "targets" to "jvm-simd,jvm-default", "profile" to "profiles/target.tsv")))

        assertTrue(cases.all { case -> case.inputs.any { it.role == "profile" && it.path == "profiles/target.tsv" } })
    }

    @Test fun `openblas is placed on the requested architecture`() {
        for (arch in listOf("x86_64", "arm64")) {
            val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "openblas", "arch" to arch)))

            assertEquals(setOf(setOf("platform:linux", "arch:$arch")), cases.map { it.resources.requiredTags }.toSet())
            assertTrue(cases.all { it.id.startsWith("comparison/openblas-$arch/") })
        }
    }

    @Test fun `onemkl and accelerate keep their fixed hosts`() {
        val onemkl = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "onemkl")))
        val accelerate = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "accelerate")))

        assertEquals(setOf(setOf("platform:linux", "arch:x86_64")), onemkl.map { it.resources.requiredTags }.toSet())
        assertEquals(setOf(setOf("platform:mac", "arch:arm64")), accelerate.map { it.resources.requiredTags }.toSet())
    }

    @Test fun `vendor placement rejects an incompatible architecture`() {
        assertFailsWith<IllegalArgumentException> {
            KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "arch" to "arm64")))
        }
    }

    @Test fun `every selected workload is its own complete paired case`() {
        val grid = "koblas-bench/lab/cases-parity-level2.txt"
        val selected = Cases.select(Cases.parse(repositoryFile(grid)))

        val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "cases" to grid)))

        val rows = cases.flatMap { case -> case.parameters.getValue("row").lines().map { case.id to it } }
        assertEquals(selected.size, rows.size)
        assertEquals(cases.size, cases.map { it.id }.distinct().size)
        for ((id, row) in rows) {
            val parsed = Cases.parse(row).single()
            assertEquals("comparison/onemkl/${parsed.operation}/${parsed.id}", id, row)
        }
        assertTrue(cases.all { it.parameters.getValue("row").lines().size == 1 })
        assertTrue(cases.all { it.resources.memoryMiB == 2048L })
        assertTrue(cases.all { case -> case.inputs.any { it.role == "executable" } })
        assertTrue(cases.all { case -> case.inputs.any { it.role == "vendor" } })
    }

    @Test fun `an operation selects only its own cells`() {
        val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "operation" to "dot")))

        assertTrue(cases.isNotEmpty() && cases.all { it.id.startsWith("comparison/onemkl/dot/") })
    }

    @Test fun `a request cannot supply its own rows`() {
        assertFailsWith<IllegalArgumentException> {
            KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "rows" to "dot+4+uniform")))
        }
    }

    @Test fun `jvm options take only hotspot flags`() {
        val accepted = mapOf("vendor" to "onemkl", "operation" to "dot", "jvm-options" to "-XX:MaxVectorSize=32 -XX:-UseFMA")

        KoblasLabSuite.plan(BenchRequest(accepted))

        for (rejected in listOf("-cp /tmp", "-XX:MaxVectorSize=32;rm", "-Dkoblas=1", "")) {
            assertFailsWith<IllegalArgumentException>(rejected) {
                KoblasLabSuite.plan(BenchRequest(accepted + ("jvm-options" to rejected)))
            }
        }
    }

    @Test fun `JVM only rows need no Native executable`() {
        val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "openblas", "arch" to "arm64",
            "targets" to "jvm-scalar,jvm-simd,openblas-jvm")))

        assertTrue(cases.all { case -> case.inputs.none { it.role == "executable" } })
    }

    @Test fun `the native engine is an accepted comparison target`() {
        val request = BenchRequest(mapOf("vendor" to "openblas", "targets" to "jvm-simd,native,openblas,openblas-jvm"))

        KoblasLabSuite.plan(request)
    }

    @Test fun `owned captures need no vendor and windows keeps compatible library architectures`() {
        val owned = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "jvm-scalar,jvm-simd")))
        assertTrue(owned.all { it.inputs.none { input -> input.role in setOf("vendor", "executable") } })
        for ((vendor, arch) in mapOf("onemkl" to "x86_64", "aocl" to "x86_64", "armpl" to "arm64")) {
            val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to vendor, "arch" to arch,
                "platform" to "windows", "targets" to "$vendor-jvm")))
            assertTrue(cases.all { it.resources.requiredTags == setOf("platform:windows", "arch:$arch") })
        }
        val native = KoblasLabSuite.plan(BenchRequest(mapOf("platform" to "windows", "targets" to "native")))
        assertTrue(native.all { case -> case.inputs.any { it.path.endsWith("mingwX64/releaseExecutable/koblas-bench.exe") } })
    }

    @Test fun `prepared capture retains partial outputs and receives its frozen profile`() {
        for ((complete, exit) in listOf(true to 0, false to 0, true to 7)) {
            val root = Files.createTempDirectory("koblas-lab-adapter")
            try {
                val repository = Files.createDirectories(root.resolve("repository/koblas-bench")).parent
                Files.createDirectories(repository.resolve("profiles"))
                Files.writeString(repository.resolve("profiles/target.tsv"),
                    "koblas-profile-v3\tfixture\t*\t*\t*\t*\t*\t*\tfixture\ttest\n")
                Files.writeString(repository.resolve("koblas-bench/capture-report.sh"), """
                    #!/usr/bin/env bash
                    set -eu
                    original=("${'$'}@")
                    while ((${ '$'}#)); do
                      case "${'$'}1" in
                        --prepared|--retain-incomplete) shift ;;
                        --output) output=${'$'}2; shift 2 ;;
                        --cases) cases=${'$'}2; shift 2 ;;
                        *) shift 2 ;;
                      esac
                    done
                    mkdir "${'$'}output"
                    printf '%s\n' "${'$'}{original[@]}" > "${'$'}output/arguments.txt"
                    printf '%s\n' "${'$'}KOBLAS_BENCH_ENGINE_PROFILE" > "${'$'}output/profile.txt"
                    cp "${'$'}cases" "${'$'}output/cases.txt"
                    printf 'case,ns\nfixture,12\n' > "${'$'}output/jvm-scalar.csv"
                    printf 'status=${if (complete) "complete" else "incomplete"}\n' > "${'$'}output/metadata.txt"
                    echo retained-log
                    exit $exit
                """.trimIndent() + "\n")
                val case = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "jvm-scalar", "operation" to "dot",
                    "profile" to "profiles/target.tsv"))).first()

                val result = KoblasLabSuite.run(case, BenchContext(root.resolve("out"), worktree = repository))

                assertEquals(if (complete && exit == 0) BenchStatus.SUCCESS else BenchStatus.FAILED, result.status)
                assertTrue(result.artifacts.any { it.path == "report/jvm-scalar.csv" })
                assertTrue(Files.readString(root.resolve("out/capture.log")).contains("retained-log"))
                assertEquals(repository.resolve("profiles/target.tsv").toString(), Files.readString(root.resolve("out/report/profile.txt")).trim())
                assertEquals(case.parameters.getValue("row"), Files.readString(root.resolve("out/report/cases.txt")).trim())
                val arguments = Files.readAllLines(root.resolve("out/report/arguments.txt"))
                assertTrue("--prepared" in arguments && "--retain-incomplete" in arguments)
                assertFailsWith<IllegalArgumentException> {
                    KoblasLabSuite.run(case.copy(inputs = emptyList()), BenchContext(root.resolve("forged"), worktree = repository))
                }
            } finally { root.toFile().deleteRecursively() }
        }
    }

    private fun repositoryFile(path: String): String =
        generateSequence(Path.of("").toAbsolutePath()) { it.parent }
            .map { it.resolve(path) }.first { java.nio.file.Files.isRegularFile(it) }.let(java.nio.file.Files::readString)
}
