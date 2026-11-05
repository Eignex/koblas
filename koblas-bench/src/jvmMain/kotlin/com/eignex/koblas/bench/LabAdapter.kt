package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.Vendor
import com.eignex.lab.bench.*
import com.eignex.lab.bench.BenchCase as LabCase
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import com.eignex.koblas.engine.hostPlatform

/** Prepared paired captures through the Lab Bench protocol. */
public object LabAdapter {
    /** Lab Bench protocol entry point. */
    @JvmStatic
    public fun main(args: Array<String>): Unit = BenchCli.execute(KoblasLabSuite, args)
}

/** One workload and all its selected arms share one machine and worker attempt. */
internal object KoblasLabSuite : BenchSuite {
    private const val CASES = "koblas-bench/lab/cases-vendor-comparison.txt"
    private const val ROW = "row"
    private val vendors = Vendor.entries.associateBy { it.vendorName.lowercase() }
    private val parameters = setOf("vendor", "arch", "platform", "operation", "cases", "samples", "warmups",
        "target-ms", "forks", "rounds", "targets", "profile-gc", "jvm-options", "smoke", "status-sweep")
    private val owned = setOf("jvm-scalar", "jvm-simd", "jvm-default", "native", "native-default")

    override fun describe() = BenchDescriptor("koblas-bench", "Prepared paired JVM and Native workloads")

    override fun plan(request: BenchRequest): List<LabCase> {
        val p = request.parameters
        validate(p)
        val vendor = p["vendor"]?.let(vendors::getValue)
        val platform = p["platform"] ?: if (vendor == Vendor.Accelerate) "mac" else "linux"
        val arch = p["arch"] ?: if (vendor in setOf(Vendor.Accelerate, Vendor.ArmPl)) "arm64" else "x86_64"
        require(platform in setOf("linux", "mac", "windows") && arch in setOf("x86_64", "arm64"))
        require(if (vendor == Vendor.Accelerate) platform == "mac" else vendor == null || platform != "mac")
        require(vendor !in setOf(Vendor.OneMkl, Vendor.Aocl) || arch == "x86_64")
        require(vendor != Vendor.ArmPl || arch == "arm64")
        val targets = p["targets"] ?: if (p["status-sweep"] == "true") "jvm-default" else if (vendor == null) "jvm-scalar,jvm-simd"
            else "jvm-scalar,jvm-simd,${vendor.vendorName.lowercase()}-jvm,${vendor.vendorName.lowercase()}"
        val selected = targets.split(',')
        require(selected.distinct().size == selected.size && selected.all {
            it in owned || it.removeSuffix("-jvm") == p["vendor"] ||
                (vendor == Vendor.OneMkl && it.removeSuffix("-jvm") == "openblas")
        }) { "Invalid capture targets" }
        val sweep = p["status-sweep"] == "true"
        val cases = p["cases"] ?: if (sweep) "koblas-bench/lab/cases-status-sweep-v1.txt" else CASES
        val inputs = mutableListOf(BenchInput(cases, role = "cases"))
        if (selected.any { !it.startsWith("jvm-") && !it.endsWith("-jvm") }) {
            val target = when {
                platform == "linux" -> if (arch == "arm64") "linuxArm64" else "linuxX64"
                platform == "mac" && arch == "arm64" -> "macosArm64"
                platform == "windows" && arch == "x86_64" -> "mingwX64"
                else -> error("No prepared Native target for $platform/$arch")
            }
            val binary = if (platform == "windows") "koblas-bench.exe" else "koblas-bench.kexe"
            inputs += BenchInput("koblas-bench/build/bin/$target/releaseExecutable/$binary", role = "executable")
        }
        if (vendor == Vendor.OneMkl && platform == "linux") {
            inputs += BenchInput("koblas-bench/build/vendor-reference", role = "vendor")
        }
        val rows = Files.readAllLines(repositoryRoot().resolve(cases)).map(String::trim)
            .filter { it.isNotEmpty() && !it.startsWith('#') }
            .map { line -> Cases.parse(line).single() to line }
            .filter { (case, _) -> "default" in case.suites && (p["operation"] == null || p["operation"] == "all" || case.operation == p["operation"]) }
        require(rows.isNotEmpty()) { "No workloads selected" }
        val host = if (vendor == Vendor.OpenBlas) "openblas-$arch" else p["vendor"] ?: "owned-$platform-$arch"
        return rows.map { (case, row) ->
            LabCase("comparison/$host/${case.operation}/${case.id}", p + mapOf(ROW to row, "targets" to targets),
                BenchResources(memoryMiB = 2048, timeoutSeconds = if (sweep) 60 else 7200,
                    requiredTags = setOf("platform:$platform", "arch:$arch")), inputs = inputs,
                attributes = if (sweep) mapOf("regressionPolicy" to "metric-v1", "regressionMetric" to "median_ns",
                    "regressionThreshold" to "0.15", "suite" to "koblas-status-v1") else emptyMap())
        }
    }

    override fun run(case: LabCase, context: BenchContext): BenchResult {
        validate(case.parameters - ROW)
        val root = requireNotNull(context.worktree)
        val output = context.reportPath()
        val cases = context.outputDirectory.resolve("cases.txt")
        Files.writeString(cases, case.parameters.getValue(ROW) + "\n")
        val argv = mutableListOf(captureShell(), root.resolve("koblas-bench/capture-report.sh").toString().replace('\\', '/'),
            "--prepared", "--retain-incomplete", "--output", output.toString(), "--cases", cases.toString(),
            "--jvm-gc", "serial", "--jvm-heap", "512m")
        for ((key, default) in mapOf("targets" to case.parameters.getValue("targets"), "samples" to "5",
            "warmups" to "20", "target-ms" to "200", "forks" to "2", "rounds" to "3")) {
            argv += listOf("--$key", case.parameters[key] ?: default)
        }
        if (case.parameters["smoke"] == "true") argv += "--smoke"
        val vendor = case.parameters["vendor"]
        if (vendor != null) argv += listOf("--libraries", if (vendor == "onemkl") "onemkl,openblas" else vendor)
        val executable = case.inputs.singleOrNull { it.role == "executable" }
        require(executable != null || case.parameters.getValue("targets").split(',').all { it.startsWith("jvm-") || it.endsWith("-jvm") })
        executable?.let { input ->
            require(input.root == BenchInputRoot.SOURCE)
            argv += listOf("--native-executable", root.resolve(input.path).toString())
        }
        val environment = mutableMapOf(
            "TMPDIR" to context.outputDirectory.toString(),
            "KOBLAS_BENCH_JAVA" to Path.of(System.getProperty("java.home"), "bin", "java").toString(),
            "KOBLAS_BENCH_CLASSPATH" to System.getProperty("java.class.path"),
            "KOBLAS_BENCH_PROFILE_GC" to (case.parameters["profile-gc"] ?: "false"),
        )
        case.parameters["jvm-options"]?.let { environment["JAVA_TOOL_OPTIONS"] = it }
        case.inputs.singleOrNull { it.role == "vendor" }?.let { input ->
            require(input.root == BenchInputRoot.SOURCE)
            val directory = root.resolve(input.path)
            Files.copy(directory.resolve("vendor-package.txt"), context.outputDirectory.resolve("vendor-package.txt"))
            environment["MKLROOT"] = directory.toString()
            environment["MKL_THREADING_LAYER"] = "SEQUENTIAL"
            environment["LD_LIBRARY_PATH"] = listOfNotNull(directory.resolve("lib").toString(),
                System.getenv("LD_LIBRARY_PATH")?.takeIf { it.isNotEmpty() }).joinToString(File.pathSeparator)
        }
        val exit = context.process(argv, "capture.log", environment = environment)
        val metadata = output.resolve("metadata.txt")
        val complete = Files.isRegularFile(metadata) && Files.newBufferedReader(metadata).use { it.readLine() == "status=complete" }
        val controls = buildJsonObject {
            put("policy", "koblas-jmh-status-v1")
            put("java", runtimeIdentity())
            put("vm", System.getProperty("java.vm.name"))
            put("os", System.getProperty("os.name"))
            put("arch", System.getProperty("os.arch"))
            put("cpu", hostPlatform().cpuModel ?: "unknown")
            put("gc", "serial"); put("heap", "512m")
        }
        val sweep = case.parameters["status-sweep"] == "true"
        val timing = if (sweep && exit == 0 && complete) runCatching {
            val rows = readTimingCsv(Files.readString(output.resolve("jvm-default.csv")))
            require(rows.size == 1) { "Sweep needs exactly one timed workload" }
            val row = rows.values.single()
            require(row.getValue("case") == Cases.parse(case.parameters.getValue(ROW)).single().id)
            require(row.getValue("status") == "ok") { "Default engine declined a sweep workload" }
            require(row.getValue("samples").toInt() == case.parameters.getValue("samples").toInt())
            require(row.getValue("forks") == "1")
            row.getValue("median_ns").toDouble().also { require(it.isFinite() && it > 0) }
        } else null
        val success = exit == 0 && complete && (!sweep || timing?.isSuccess == true)
        return BenchResult(case.id, if (success) BenchStatus.SUCCESS else BenchStatus.FAILED,
            artifacts = context.artifacts() + context.artifact("capture.log", "text/plain") +
                if (case.inputs.any { it.role == "vendor" }) listOf(context.artifact("vendor-package.txt", "text/plain")) else emptyList(),
            metrics = timing?.getOrNull()?.let { listOf(BenchMetric("median_ns", it, "ns/op", MetricDirection.LOWER)) }.orEmpty(),
            details = if (sweep) buildJsonObject { put("benchmarkControls", controls) } else null,
            message = if (success) null else "Capture exited $exit, was incomplete, or failed timing validation: ${timing?.exceptionOrNull()?.message}")
    }

    private fun validate(p: Map<String, String>) {
        require(p.keys.all { it in parameters } && p.values.none { value -> value.isBlank() || value.any(Char::isISOControl) })
        p["vendor"]?.let { require(it in vendors) { "Unknown vendor" } }
        for (key in listOf("samples", "target-ms", "forks", "rounds")) {
            p[key]?.let { require(it.toIntOrNull()?.let { value -> value > 0 } == true) { "$key must be positive" } }
        }
        p["warmups"]?.let { require(it.toIntOrNull()?.let { value -> value >= 0 } == true) }
        for (key in listOf("smoke", "profile-gc", "status-sweep")) p[key]?.toBooleanStrict()
        if (p["status-sweep"] == "true") {
            require(p["vendor"] == null && p["cases"] == null && p["operation"] == null && p["jvm-options"] == null)
            require(p["targets"] == null || p["targets"] == "jvm-default")
            require(p["samples"] == "3" && p["warmups"] == "3" && p["target-ms"] == "100" &&
                p["forks"] == "1" && p["rounds"] == "1" && p["smoke"] != "true" && p["profile-gc"] != "true") {
                "Status sweep v1 requires 3 warmups, 3 samples, 100ms, one fork and one round"
            }
        }
        p["jvm-options"]?.let { options ->
            require(options.split(' ').all { it.matches(Regex("-XX:[+-]?[A-Za-z]+(=[0-9]+)?")) }) {
                "jvm-options takes space-separated -XX options"
            }
        }
    }

    private fun repositoryRoot(): Path = System.getenv("LAB_WORKTREE")?.let(Path::of)
        ?: generateSequence(Path.of("").toAbsolutePath()) { it.parent }
            .first { Files.isDirectory(it.resolve("koblas-bench/lab")) }
}
