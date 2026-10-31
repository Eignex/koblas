package com.eignex.koblas.bench

import com.eignex.lab.bench.*
import com.eignex.lab.bench.BenchCase as LabCase
import java.nio.file.Files
import java.nio.file.Path

/** Builds the native reference on the execution host before capturing matched JVM and vendor arms. */
public object VendorComparisonAdapter {
    /** Lab Bench protocol entry point. */
    @JvmStatic
    public fun main(args: Array<String>): Unit = BenchCli.execute(VendorComparisonSuite, args)
}

/**
 * One lab case per routine and size band of the selected rows, each running every arm of its rows on one host in
 * one attempt. A cell is what a parity verdict is scored over, so a lost worker replaces exactly one verdict's rows
 * and the job's parallel slots spread the cells, without a row's arms ever landing on different machines.
 *
 * Native code must be linked after placement when preparation ran on another operating system.
 */
internal object VendorComparisonSuite : BenchSuite {
    private const val CASES = "koblas-bench/lab/cases-vendor-comparison.txt"
    private val PARAMETERS = setOf(
        "vendor", "arch", "operation", "cases", "samples", "warmups", "target-ms", "forks", "rounds", "targets", "profile-gc",
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
        val resources = BenchResources(2, 4608, 7200, tags)
        val inputs = listOf(BenchInput(cases, role = "cases"))
        return rows.groupBy { (case, _) -> case.operation to sizeBand(case) }.map { (cell, members) ->
            val (operation, band) = cell
            LabCase("comparison/$host/$operation/$band", p + (ROWS to members.joinToString("\n") { it.second }),
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
        case.parameters.forEach { (key, value) ->
            if (key != ROWS && key != "cases" && key != "operation") {
                argv += "--$key"
                argv += value
            }
        }
        val exit = context.process(argv, "comparison.log",
            environment = mapOf("TMPDIR" to context.outputDirectory.toString()))
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
                it in setOf("jvm-scalar", "jvm-simd", "jvm-default", "native", vendor, "$vendor-jvm")
            }) { "Invalid comparison targets" }
        }
        p["profile-gc"]?.let { require(it in setOf("true", "false")) { "profile-gc must be true or false" } }
        p["operation"]?.let { require(it.matches(Regex("[a-z][a-z0-9-]*"))) { "Invalid operation" } }
        for (key in listOf("samples", "warmups", "target-ms", "forks", "rounds")) {
            p[key]?.let { require(it.matches(Regex("[1-9][0-9]*")) && it.toIntOrNull() != null) { "$key must be positive" } }
        }
        val cases = p["cases"] ?: CASES
        require(cases.matches(Regex("koblas-bench/lab/[a-z0-9-]+\\.txt"))) { "Invalid comparison case path" }
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

private val LEVEL_2 = setOf("gemv", "ger", "symv", "syr", "syr2", "trsv", "trmv")

/**
 * The parity size band of [case]: small below 4096 vector elements or 16,384 matrix elements, large at the
 * 65,536-element cap, and mid between. A symmetric or triangular order counts its full square, as the grids do.
 */
internal fun sizeBand(case: BenchCase): String {
    val elements = when {
        case.dimensions.size >= 2 -> case.dimension(0) * case.dimension(1)
        case.operation in LEVEL_2 -> case.dimension(0) * case.dimension(0)
        else -> case.dimension(0)
    }
    val mid = if (case.dimensions.size >= 2 || case.operation in LEVEL_2) 16_384 else 4_096
    return when {
        elements >= 65_000 -> "large"
        elements >= mid -> "mid"
        else -> "small"
    }
}
