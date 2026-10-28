package com.eignex.koblas.bench

import com.eignex.lab.bench.*
import com.eignex.lab.bench.BenchCase as LabCase
import java.nio.file.Files
import java.nio.file.Path

fun main(args: Array<String>) = BenchCli.execute(KoblasLabSuite, args)

internal object KoblasLabSuite : BenchSuite {
    private val options = setOf("targets", "libraries", "operation", "suite", "cases", "samples", "warmups", "target-ms", "forks", "pass", "rounds", "cpu", "jvm-gc", "jvm-heap", "vendors-only", "smoke", "native-executable", "memory-mib", "timeout-seconds")
    private val vendors = setOf("openblas", "onemkl", "aocl", "armpl", "accelerate")
    private val jvmTargets = setOf("jvm-scalar", "jvm-simd", "jvm-default")
    private val nativePlatforms = mapOf(
        "linuxX64" to setOf("platform:linux", "arch:x86_64"),
        "linuxArm64" to setOf("platform:linux", "arch:arm64"),
        "macosArm64" to setOf("platform:mac", "arch:arm64"),
    )

    override fun describe() = BenchDescriptor("koblas-bench", "Prepared JVM and Native captures with original CSV reports and backend attribution")

    override fun plan(request: BenchRequest): List<LabCase> {
        val p = request.parameters
        require(p.keys.all { it in options }) { "Unknown Koblas bench parameter" }
        require(p.values.none { value -> value.isBlank() || value.any(Char::isISOControl) }) { "Capture parameters must be nonblank literal arguments" }
        val vendorsOnly = p["vendors-only"]?.toBooleanStrict() ?: false
        require(!vendorsOnly || "targets" in p) { "vendors-only requires explicit vendor targets" }
        val targets = (p["targets"] ?: "jvm-scalar,jvm-simd").split(',')
        require(targets.distinct().size == targets.size) { "Duplicate capture target" }
        require(targets.all { it in jvmTargets || it in setOf("native", "native-default") || it.removeSuffix("-jvm") in vendors }) { "Unknown capture target" }
        require(!vendorsOnly || targets.all { it.removeSuffix("-jvm") in vendors }) { "vendors-only requires vendor targets" }
        val native = targets.any { it == "native" || it == "native-default" || it in vendors }
        val jvm = targets.any { it in jvmTargets || it.endsWith("-jvm") }
        listOf("samples", "target-ms", "forks", "pass", "rounds").forEach { key -> p[key]?.let { positiveInt(key, it) } }
        p["warmups"]?.let { require(it.matches(Regex("[0-9]+")) && it.toIntOrNull() != null) { "warmups must be a nonnegative integer" } }
        p["smoke"]?.toBooleanStrict()
        require((p["suite"] ?: "default") in setOf("default", "sweep")) { "suite must be default or sweep" }
        require(p["suite"] != "sweep" || (p["operation"] ?: "all") != "all") { "suite sweep requires a specific operation" }
        p["operation"]?.let { require(it.matches(Regex("[a-z][a-z0-9-]*"))) { "Invalid operation" } }
        require((p["rounds"] ?: "1").toInt() == 1 || "pass" !in p) { "rounds assigns pass numbers itself" }
        p["cpu"]?.let { require(it == "auto" || it.matches(Regex("[0-9]+")) && it.toIntOrNull() != null) { "cpu must be a logical CPU number or auto" } }
        p["jvm-gc"]?.let { require(jvm && it in setOf("g1", "serial", "zgc")) { "jvm-gc requires a JVM target and g1, serial or zgc" } }
        val memory = (p["memory-mib"] ?: "2048").toLong()
        require(memory >= 1024) { "memory-mib must be at least 1024" }
        val timeout = (p["timeout-seconds"] ?: "3600").toLong()
        require(timeout > 0) { "timeout-seconds must be positive" }
        p["jvm-heap"]?.let { heap ->
            require(jvm && heap.matches(Regex("[1-9][0-9]*[mg]"))) { "jvm-heap requires a JVM target and a positive size ending in m or g" }
            val size = heap.dropLast(1).toLongOrNull()
            val factor = if (heap.last() == 'g') 1024 else 1
            // The prepared launcher and its measured JMH fork can both hold this heap at once.
            require(size != null && size <= heapBudgetMiB(memory) / factor) { "jvm-heap exceeds the capture budget of ${heapBudgetMiB(memory)} MiB per JVM" }
        }
        val selectedVendors = targets.map { it.removeSuffix("-jvm") }.filter { it in vendors }.distinct()
        p["libraries"]?.let { libraries ->
            require(libraries == "all" || libraries.split(',').let { names -> names.all { it in vendors } && names.distinct().size == names.size }) { "Unknown or duplicate library" }
            require(libraries != "all" || "aocl" !in selectedVendors) { "AOCL requires explicit libraries" }
            require(libraries == "all" || selectedVendors.all { it in libraries.split(',') }) { "Vendor targets must be included in libraries" }
        }
        val inputs = mutableListOf<BenchInput>()
        p["cases"]?.let { inputs += BenchInput(it, role = "cases") }
        val tags = mutableSetOf<String>()
        val executable = p["native-executable"] ?: "koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe"
        require(native || "native-executable" !in p) { "native-executable requires a Native target" }
        if (native) {
            inputs += BenchInput(executable, role = "executable")
            require(executable.startsWith("koblas-bench/build/")) { "Native executable must be prepared under koblas-bench/build" }
            tags += requireNotNull(nativePlatforms.entries.singleOrNull { "/${it.key}/" in executable }?.value) { "Unknown Native executable platform" }
        }
        if ("cpu" in p) tags += "platform:linux"
        if ("accelerate" in selectedVendors) tags += "platform:mac"
        if ("armpl" in selectedVendors) tags += setOf("platform:linux", "arch:arm64")
        if (selectedVendors.any { it == "onemkl" || it == "aocl" }) tags += setOf("platform:linux", "arch:x86_64")
        require(tags.count { it.startsWith("platform:") } <= 1 && tags.count { it.startsWith("arch:") } <= 1) { "Incompatible capture platforms" }
        val parameters = p - setOf("memory-mib", "timeout-seconds") + mapOf("targets" to targets.joinToString(",")) +
            (if ("libraries" !in p && selectedVendors.isNotEmpty()) mapOf("libraries" to selectedVendors.joinToString(",")) else emptyMap()) +
            (if (native) mapOf("native-executable" to executable) else emptyMap())
        return listOf(LabCase("capture/${targets.joinToString("+")}", parameters,
            BenchResources(1, memory, timeout, tags), inputs = inputs))
    }

    private fun positiveInt(key: String, value: String) {
        require(value.matches(Regex("[1-9][0-9]*")) && value.toIntOrNull() != null) { "$key must be a positive integer" }
    }

    private fun heapBudgetMiB(memoryMiB: Long) = (memoryMiB - 768) / 2

    override fun run(case: LabCase, context: BenchContext): BenchResult {
        plan(BenchRequest(case.parameters + mapOf(
            "memory-mib" to case.resources.memoryMiB.toString(),
            "timeout-seconds" to case.resources.timeoutSeconds.toString())))
        val output = context.reportPath()
        val root = requireNotNull(context.worktree)
        val argv = mutableListOf("bash", root.resolve("koblas-bench/capture-report.sh").toString(),
            "--prepared", "--retain-incomplete", "--output", output.toString())
        case.parameters.forEach { (key, value) ->
            when (key) {
                "smoke", "vendors-only" -> if (value.toBooleanStrict()) argv += "--$key"
                "cases", "native-executable" -> {
                    require(case.inputs.any { it.path == value && it.root == BenchInputRoot.SOURCE }) { "Undeclared capture input: $key" }
                    argv += "--$key"
                    argv += root.resolve(value).toString()
                }
                else -> { argv += "--$key"; argv += value }
            }
        }
        val exit = context.process(argv, "capture.log", environment = mapOf(
            "TMPDIR" to context.outputDirectory.toString(),
            "KOBLAS_BENCH_JAVA" to Path.of(System.getProperty("java.home"), "bin", "java").toString(),
            "KOBLAS_BENCH_CLASSPATH" to System.getProperty("java.class.path"),
            "KOBLAS_BENCH_HEAP_MIB" to heapBudgetMiB(case.resources.memoryMiB).toString()))
        val artifacts = context.artifacts() + context.artifact("capture.log", "text/plain")
        val rounds = (case.parameters["rounds"] ?: "1").toInt()
        val targets = case.parameters.getValue("targets").split(',')
        val reports = if (rounds == 1) listOf(output) else (1..rounds).map { output.resolve("pass-$it") }
        val csvs = reports.flatMap { report -> targets.map { report.resolve("$it.csv") } }.filter { Files.isRegularFile(it) }
        val rows = csvs.sumOf { csv -> Files.lines(csv).use { lines -> lines.skip(1).filter { it.isNotBlank() }.count() } }
        val completedRounds = reports.count { complete(it) }
        val success = exit == 0 && complete(output) && completedRounds == rounds && rows > 0
        return BenchResult(case.id, if (success) BenchStatus.SUCCESS else BenchStatus.FAILED,
            listOf(BenchMetric("captured-rows", rows.toDouble(), "rows"),
                BenchMetric("captured-csv-files", csvs.size.toDouble(), "files"),
                BenchMetric("completed-rounds", completedRounds.toDouble(), "rounds")), artifacts,
            message = if (success) null else "Capture incomplete (exit $exit, $completedRounds/$rounds rounds complete); inspect capture.log and partial reports")
    }

    private fun complete(report: Path): Boolean = Files.isRegularFile(report.resolve("metadata.txt")) &&
        Files.newBufferedReader(report.resolve("metadata.txt")).use { it.readLine() == "status=complete" }
}
