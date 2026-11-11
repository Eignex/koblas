package com.eignex.koblas.bench

import com.eignex.lab.bench.*
import java.nio.file.Files
import java.nio.file.Path
import kotlinx.serialization.json.*
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
            assertEquals(2, cases.map { it.id }.distinct().size)
            assertEquals(setOf("dot+64+uniform", "copy+17+uniform+stride=2"), cases.map { it.parameters.getValue("row") }.toSet())
            assertTrue(cases.all { it.resources.memoryMiB == 2048L })
            assertTrue(cases.all { it.inputs.any { input -> input.role == "executable" && "/linuxX64/" in input.path } })
            assertTrue(cases.all { case -> case.inputs.any { it.path.endsWith("calibration/plan-fixture.txt") } })
        } finally { Files.delete(file) }
    }

    @Test fun `vendors select their compatible hosts`() {
        for ((vendor, arch, platform) in listOf(Triple("openblas", "x86_64", "linux"),
            Triple("openblas", "arm64", "linux"), Triple("onemkl", "x86_64", "linux"),
            Triple("accelerate", "arm64", "mac"))) {
            val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to vendor, "arch" to arch)))

            assertEquals(setOf(setOf("platform:$platform", "arch:$arch")), cases.map { it.resources.requiredTags }.toSet())
        }
    }

    @Test fun `vendor placement rejects incompatible hosts`() {
        for (parameters in listOf(mapOf("vendor" to "onemkl", "arch" to "arm64"),
            mapOf("vendor" to "openblas", "platform" to "mac"))) {
            assertFailsWith<IllegalArgumentException> { KoblasLabSuite.plan(BenchRequest(parameters)) }
        }
    }

    @Test fun `an operation selects only its own cells`() {
        val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to "onemkl", "operation" to "dot")))

        assertTrue(cases.isNotEmpty() && cases.all { it.id.startsWith("comparison/onemkl/dot/") })
    }

    @Test fun `requests reject reserved rows and nonportable input paths`() {
        for (parameters in listOf(mapOf("row" to "dot+4+uniform")) + listOf("cases").flatMap { key ->
            listOf("/tmp/cases", "../cases", "a/../cases", "a\\cases", "C:cases").map { mapOf(key to it) }
        }) {
            assertFailsWith<IllegalArgumentException> { KoblasLabSuite.plan(BenchRequest(parameters)) }
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

    @Test fun `owned captures need no vendor and windows keeps compatible library architectures`() {
        val owned = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "jvm-scalar,jvm-simd,jvm-default-no-simd")))
        assertTrue(owned.all { it.inputs.none { input -> input.role in setOf("vendor", "executable") } })
        for ((vendor, arch) in mapOf("onemkl" to "x86_64", "aocl" to "x86_64", "armpl" to "arm64")) {
            val cases = KoblasLabSuite.plan(BenchRequest(mapOf("vendor" to vendor, "arch" to arch,
                "platform" to "windows", "targets" to "$vendor-jvm")))
            assertTrue(cases.all { it.resources.requiredTags == setOf("platform:windows", "arch:$arch") })
        }
        val native = KoblasLabSuite.plan(BenchRequest(mapOf("platform" to "windows", "targets" to "native")))
        assertTrue(native.all { case -> case.inputs.any { it.path.endsWith("mingwX64/releaseExecutable/koblas-bench.exe") } })
    }

    @Test fun `prepared capture retains partial outputs`() {
        for ((complete, exit) in listOf(true to 0, false to 0, true to 7)) {
            val root = Files.createTempDirectory("koblas-lab-adapter")
            try {
                val repository = Files.createDirectories(root.resolve("repository/koblas-bench")).parent
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
                    cp "${'$'}cases" "${'$'}output/cases.txt"
                    printf 'case,ns\nfixture,12\n' > "${'$'}output/jvm-scalar.csv"
                    printf 'status=${if (complete) "complete" else "incomplete"}\n' > "${'$'}output/metadata.txt"
                    echo retained-log
                    exit $exit
                """.trimIndent() + "\n")
                val case = KoblasLabSuite.plan(BenchRequest(mapOf("targets" to "jvm-scalar", "operation" to "dot"))).first()

                val result = KoblasLabSuite.run(case, BenchContext(root.resolve("out"), worktree = repository))

                assertEquals(if (complete && exit == 0) BenchStatus.SUCCESS else BenchStatus.FAILED, result.status)
                assertTrue(result.artifacts.any { it.path == "report/jvm-scalar.csv" })
                assertTrue(Files.readString(root.resolve("out/capture.log")).contains("retained-log"))
                assertEquals(case.parameters.getValue("row"), Files.readString(root.resolve("out/report/cases.txt")).trim())
                val arguments = Files.readAllLines(root.resolve("out/report/arguments.txt"))
                assertTrue("--prepared" in arguments && "--retain-incomplete" in arguments)
            } finally { root.toFile().deleteRecursively() }
        }
    }

    @Test fun `status capture exports each engine metric with its measured attribution`() {
        val root = Files.createTempDirectory("koblas-engine-status")
        try {
            val repository = Files.createDirectories(root.resolve("repository/koblas-bench")).parent
            Files.writeString(repository.resolve("koblas-bench/capture-report.sh"), """
                #!/usr/bin/env bash
                set -eu
                while ((${ '$'}#)); do
                  case "${'$'}1" in
                    --prepared|--retain-incomplete) shift ;;
                    --output) output=${'$'}2; shift 2 ;;
                    --cases) cases=${'$'}2; shift 2 ;;
                    *) shift 2 ;;
                  esac
                done
                mkdir "${'$'}output"
                row=${'$'}(cat "${'$'}cases")
                for target in jvm-default jvm-scalar jvm-simd; do
                  printf '%s\n' '$CSV_HEADER' > "${'$'}output/${'$'}target.csv"
                  printf '%s,ok,direct,arithmetic,scalar/dot,scalar,key,3,1,12.5,12.5,12.5\n' "${'$'}row" >> "${'$'}output/${'$'}target.csv"
                done
                printf 'status=complete\n' > "${'$'}output/metadata.txt"
            """.trimIndent() + "\n")
            val case = KoblasLabSuite.plan(BenchRequest(mapOf("status-sweep" to "true", "samples" to "3",
                "warmups" to "3", "target-ms" to "100", "forks" to "1", "rounds" to "1",
                "targets" to "jvm-default,jvm-scalar,jvm-simd"))).first()

            val result = KoblasLabSuite.run(case, BenchContext(root.resolve("out"), worktree = repository))

            assertEquals(BenchStatus.SUCCESS, result.status)
            assertEquals(listOf("median_ns", "median_ns/jvm-scalar", "median_ns/jvm-simd"), result.metrics.map { it.name })
            assertTrue(result.metrics.all { it.value == 12.5 && it.unit == "ns/op" })
            val attribution = result.details!!.getValue("metricAttribution").jsonObject.getValue("median_ns/jvm-simd").jsonObject
            assertEquals("jvm-simd", attribution.getValue("engine").jsonPrimitive.content)
            assertEquals("scalar", attribution.getValue("provider").jsonPrimitive.content)
        } finally { root.toFile().deleteRecursively() }
    }

}
