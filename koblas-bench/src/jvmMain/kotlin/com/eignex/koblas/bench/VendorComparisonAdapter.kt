package com.eignex.koblas.bench

import com.eignex.lab.bench.*
import com.eignex.lab.bench.BenchCase as LabCase
import java.nio.file.Files
import java.nio.file.Path

/** Captures matched JVM and vendor arms from prepared executables. */
public object VendorComparisonAdapter {
    /** Lab Bench protocol entry point. */
    @JvmStatic
    public fun main(args: Array<String>): Unit = BenchCli.execute(VendorComparisonSuite, args)
}

/**
 * One lab case per workload, with every arm on the same host in one attempt. Fine rows expose progress,
 * balance the worker slots and bound the work repeated after worker loss.
 */
internal object VendorComparisonSuite : BenchSuite {
    private const val CASES = "koblas-bench/lab/cases-vendor-comparison.txt"
    private val PARAMETERS = setOf(
        "vendor", "arch", "operation", "cases", "samples", "warmups", "target-ms", "forks", "rounds", "targets", "profile-gc",
        "jvm-options", "profile",
    )

    /** The rows of one planned case, one case-file line each; set by [plan], never by a request. */
    private const val ROWS = "rows"

    override fun describe() = BenchDescriptor("koblas-vendor-comparison", "Matched JVM and native vendor calls")

    override fun plan(request: BenchRequest): List<LabCase> {
        val p = request.parameters
        require(ROWS !in p) { "rows are planned from the case file" }
        validate(p)
        val cases = p["cases"] ?: CASES
        val rows = selectedRows(repositoryRoot().resolve(cases), p["operation"] ?: "all")
        val vendor = p.getValue("vendor")
        val arch = p["arch"] ?: if (vendor == "accelerate") "arm64" else "x86_64"
        val tags = if (vendor == "accelerate") setOf("platform:mac", "arch:arm64") else setOf("platform:linux", "arch:$arch")
        val host = if (vendor == "openblas") "$vendor-$arch" else vendor
        val targets = (p["targets"] ?: "jvm-scalar,jvm-simd,$vendor-jvm,$vendor").split(',')
        val native = targets.any { !it.startsWith("jvm-") && !it.endsWith("-jvm") }
        val platform = if (vendor == "accelerate") "macosArm64" else if (arch == "arm64") "linuxArm64" else "linuxX64"
        val resources = BenchResources(memoryMiB = 2048, timeoutSeconds = 7200, requiredTags = tags)
        val inputs = listOf(BenchInput(cases, role = "cases")) +
            (if (native) listOf(BenchInput("koblas-bench/build/bin/$platform/releaseExecutable/koblas-bench.kexe", role = "executable")) else emptyList()) +
            (if (vendor == "onemkl") listOf(BenchInput("koblas-bench/build/vendor-reference", role = "vendor")) else emptyList()) +
            p["profile"]?.let { listOf(BenchInput(it, role = "profile")) }.orEmpty()
        return rows.map { (case, row) ->
            LabCase("comparison/$host/${case.operation}/${case.id}", p + (ROWS to row),
                resources, inputs = inputs)
        }
    }

    override fun run(case: LabCase, context: BenchContext): BenchResult {
        validate(case.parameters - ROWS)
        val rows = requireNotNull(case.parameters[ROWS]) { "a planned case carries its rows" }
        val root = requireNotNull(context.worktree)
        val output = context.reportPath()
        val cases = context.outputDirectory.resolve("cases.txt")
        Files.writeString(cases, rows + "\n")
        val argv = mutableListOf(captureShell(), root.resolve("koblas-bench/lab/vendor-capture.sh").toString(),
            "--output", output.toString(), "--cases", cases.toString())
        case.inputs.singleOrNull { it.role == "executable" }?.let { input ->
            require(input.root == BenchInputRoot.SOURCE) { "Executable must be a prepared source input" }
            argv += listOf("--native-executable", root.resolve(input.path).toString())
        }
        case.parameters.forEach { (key, value) ->
            if (key != ROWS && key != "cases" && key != "operation" && key != "jvm-options" && key != "profile") {
                argv += "--$key"
                argv += value
            }
        }
        // Every JVM the capture starts reads these, the measured JMH fork included; native arms ignore them.
        val profile = case.parameters["profile"]?.let { path ->
            require(case.inputs.any { it.path == path && it.root == BenchInputRoot.SOURCE && it.role == "profile" }) { "Undeclared profile input" }
            root.resolve(path).also { com.eignex.koblas.EngineProfile.parse(Files.readString(it)) }
        }
        val environment = (profile?.let { mapOf("KOBLAS_BENCH_ENGINE_PROFILE" to it.toString()) } ?: emptyMap()) + mapOf("TMPDIR" to context.outputDirectory.toString()) +
            (case.parameters["jvm-options"]?.let { mapOf("JAVA_TOOL_OPTIONS" to it) } ?: emptyMap())
        val exit = context.process(argv, "comparison.log", environment = environment)
        val report = output.resolve("report")
        val complete = Files.isRegularFile(report.resolve("metadata.txt")) &&
            Files.readAllLines(report.resolve("metadata.txt")).firstOrNull() == "status=complete"
        return BenchResult(case.id, if (exit == 0 && complete) BenchStatus.SUCCESS else BenchStatus.FAILED,
            artifacts = context.artifacts() + context.artifact("comparison.log", "text/plain"),
            message = if (exit == 0 && complete) null else "Vendor comparison exited $exit or retained an incomplete capture")
    }

    private fun validate(p: Map<String, String>) {
        require(p.keys.all { it in PARAMETERS }) { "Unknown vendor comparison parameter" }
        val vendor = p.getValue("vendor")
        require(vendor in setOf("onemkl", "accelerate", "openblas")) { "vendor must be onemkl, accelerate or openblas" }
        // OpenBLAS ships on every Linux worker, so it is the one vendor that runs on either architecture.
        val arch = p["arch"] ?: if (vendor == "accelerate") "arm64" else "x86_64"
        require(arch in setOf("x86_64", "arm64") && (vendor == "openblas" || p["arch"] == null)) {
            "arch is x86_64 or arm64, and only openblas takes it"
        }
        p["targets"]?.let { targets ->
            val selected = targets.split(',')
            require(selected.distinct().size == selected.size && selected.all {
                it in setOf("jvm-scalar", "jvm-simd", "jvm-default", "native", "native-default", vendor, "$vendor-jvm") ||
                    vendor == "onemkl" && it in setOf("openblas", "openblas-jvm")
            }) { "Invalid comparison targets" }
        }
        p["profile-gc"]?.let { require(it in setOf("true", "false")) { "profile-gc must be true or false" } }
        p["jvm-options"]?.let { options ->
            require(options.split(' ').all { it.matches(Regex("-XX:[+-]?[A-Za-z]+(=[0-9]+)?")) }) {
                "jvm-options takes space-separated -XX options"
            }
        }
        p["operation"]?.let { require(it.matches(Regex("[a-z][a-z0-9-]*"))) { "Invalid operation" } }
        for (key in listOf("samples", "warmups", "target-ms", "forks", "rounds")) {
            p[key]?.let { require(it.matches(Regex("[1-9][0-9]*")) && it.toIntOrNull() != null) { "$key must be positive" } }
        }
        val cases = p["cases"] ?: CASES
        require(cases.matches(Regex("koblas-bench/(lab|build/calibration)/[a-z0-9-]+\\.txt"))) { "Invalid comparison case path" }
    }

    /** Each default-suite line of [file] that [operation] selects, parsed and as written. */
    private fun selectedRows(file: Path, operation: String): List<Pair<BenchCase, String>> {
        val rows = Files.readAllLines(file).map(String::trim).filter { it.isNotEmpty() && !it.startsWith('#') }
            .map { line -> Cases.parse(line).single() to line }
            .filter { (case, _) -> "default" in case.suites && (operation == "all" || case.operation == operation) }
        require(rows.isNotEmpty()) { "operation '$operation' selected no cases" }
        return rows
    }

    /** The prepared checkout during a lab plan, else the repository holding the working directory. */
    private fun repositoryRoot(): Path = System.getenv("LAB_WORKTREE")?.let(Path::of)
        ?: generateSequence(Path.of("").toAbsolutePath()) { it.parent }
            .first { Files.isDirectory(it.resolve("koblas-bench/lab")) }
}
