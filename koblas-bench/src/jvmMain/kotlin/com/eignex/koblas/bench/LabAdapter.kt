package com.eignex.koblas.bench

import com.eignex.lab.bench.*
import com.eignex.lab.bench.BenchCase as LabCase
import java.nio.file.Files
import java.nio.file.Path

fun main(args: Array<String>) = BenchCli.execute(KoblasLabSuite, args)

internal object KoblasLabSuite : BenchSuite {
    private val options = setOf("targets", "libraries", "operation", "suite", "samples", "warmups", "target-ms", "forks", "pass", "smoke", "native-executable", "memory-mib", "timeout-seconds")
    private val vendors = setOf("openblas", "onemkl", "aocl", "armpl", "accelerate")

    override fun describe() = BenchDescriptor("koblas-bench", "Prepared JVM and Native captures with original CSV reports and backend attribution")

    override fun plan(request: BenchRequest): List<LabCase> {
        val p = request.parameters
        require(p.keys.all { it in options }) { "Unknown Koblas bench parameter" }
        val targets = (p["targets"] ?: "jvm-scalar,jvm-simd").split(',').distinct()
        val native = targets.any { it == "native" || it == "native-default" || it in vendors }
        require(targets.all { it in setOf("jvm-scalar", "jvm-simd", "jvm-default", "native", "native-default") || it in vendors || it.removeSuffix("-jvm") in vendors }) { "Unknown capture target" }
        listOf("samples", "target-ms", "forks", "pass").forEach { key -> p[key]?.let { require(it.toInt() > 0) } }
        p["warmups"]?.let { require(it.toInt() >= 0) }
        p["smoke"]?.toBooleanStrict()
        val executable = p["native-executable"] ?: "koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe"
        if (native) require(executable.startsWith("koblas-bench/build/") && executable.split('/').none { it in setOf("..", ".") })
        val nativeTags = if (!native) emptySet() else when {
            "/macosArm64/" in executable -> setOf("platform:mac", "arch:arm64")
            "/linuxArm64/" in executable -> setOf("platform:linux", "arch:arm64")
            else -> setOf("platform:linux", "arch:x86_64")
        }
        require((p["memory-mib"] ?: "2048").toLong() >= 1024)
        val selectedVendors = targets.map { it.removeSuffix("-jvm") }.filter { it in vendors }.distinct()
        val parameters = (if ("libraries" !in p && selectedVendors.isNotEmpty()) p + ("libraries" to selectedVendors.joinToString(",")) else p) - setOf("memory-mib", "timeout-seconds") + mapOf("targets" to targets.joinToString(",")) +
            if (native) mapOf("native-executable" to executable) else emptyMap()
        return listOf(LabCase("capture/${targets.joinToString("+")}", parameters,
            BenchResources(1, (p["memory-mib"] ?: "2048").toLong(), (p["timeout-seconds"] ?: "3600").toLong(), nativeTags),
            inputs = if (native) listOf(BenchInput(executable, role = "executable")) else emptyList()))
    }

    override fun run(case: LabCase, context: BenchContext): BenchResult {
        val output = context.reportPath()
        val root = requireNotNull(context.worktree)
        val argv = mutableListOf("bash", root.resolve("koblas-bench/capture-report.sh").toString(),
            "--prepared", "--retain-incomplete", "--output", output.toString())
        case.parameters.forEach { (key, value) ->
            if (key == "smoke") { if (value.toBooleanStrict()) argv += "--smoke" }
            else { argv += "--$key"; argv += if (key == "native-executable") root.resolve(value).toString() else value }
        }
        val exit = context.process(argv, "capture.log", environment = mapOf(
            "TMPDIR" to context.outputDirectory.toString(),
            "KOBLAS_BENCH_JAVA" to Path.of(System.getProperty("java.home"), "bin", "java").toString(),
            "KOBLAS_BENCH_CLASSPATH" to System.getProperty("java.class.path"),
            "KOBLAS_BENCH_HEAP_MIB" to ((case.resources.memoryMiB - 768) / 2).toString()))
        val artifacts = context.artifacts() + context.artifact("capture.log", "text/plain")
        val complete = Files.isRegularFile(output.resolve("metadata.txt")) &&
            Files.readAllLines(output.resolve("metadata.txt")).firstOrNull() == "status=complete"
        val rows = if (Files.exists(output)) Files.list(output).use { files ->
            files.filter { it.toString().endsWith(".csv") }.mapToLong { Files.readAllLines(it).drop(1).count(String::isNotBlank).toLong() }.sum()
        } else 0L
        val success = exit == 0 && complete && rows > 0
        return BenchResult(case.id, if (success) BenchStatus.SUCCESS else BenchStatus.FAILED,
            listOf(BenchMetric("captured-rows", rows.toDouble(), "rows")), artifacts,
            message = if (success) null else "Capture incomplete (exit $exit); inspect capture.log and partial reports")
    }
}
