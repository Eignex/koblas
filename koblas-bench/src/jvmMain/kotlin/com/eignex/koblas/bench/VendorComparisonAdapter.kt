package com.eignex.koblas.bench

import com.eignex.lab.bench.*
import com.eignex.lab.bench.BenchCase as LabCase
import java.nio.file.Files

/** Builds the native reference on the execution host before capturing matched JVM and vendor arms. */
public object VendorComparisonAdapter {
    /** Lab Bench protocol entry point. */
    @JvmStatic
    public fun main(args: Array<String>): Unit = BenchCli.execute(VendorComparisonSuite, args)
}

/** Native code must be linked after placement when preparation ran on another operating system. */
internal object VendorComparisonSuite : BenchSuite {
    private const val CASES = "koblas-bench/lab/cases-vendor-comparison.txt"
    private val PARAMETERS = setOf(
        "vendor", "arch", "operation", "cases", "samples", "warmups", "target-ms", "forks", "rounds", "targets", "profile-gc",
    )

    override fun describe() = BenchDescriptor("koblas-vendor-comparison", "Matched JVM and native vendor calls")

    override fun plan(request: BenchRequest): List<LabCase> {
        val p = request.parameters
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
        val tags = if (vendor == "accelerate") setOf("platform:mac", "arch:arm64") else setOf("platform:linux", "arch:$arch")
        val host = if (vendor == "openblas") "$vendor-$arch" else vendor
        return listOf(LabCase("comparison/$host/${p["operation"] ?: "all"}", p,
            BenchResources(2, 4608, 7200, tags), inputs = listOf(BenchInput(cases, role = "cases"))))
    }

    override fun run(case: LabCase, context: BenchContext): BenchResult {
        plan(BenchRequest(case.parameters))
        val root = requireNotNull(context.worktree)
        val output = context.reportPath()
        val argv = mutableListOf(captureShell(), root.resolve("koblas-bench/lab/vendor-capture.sh").toString(),
            "--output", output.toString())
        case.parameters.forEach { (key, value) ->
            argv += "--$key"
            argv += if (key == "cases") root.resolve(value).toString() else value
        }
        val exit = context.process(argv, "comparison.log")
        val report = output.resolve("report")
        val complete = Files.isRegularFile(report.resolve("metadata.txt")) &&
            Files.readAllLines(report.resolve("metadata.txt")).firstOrNull() == "status=complete"
        return BenchResult(case.id, if (exit == 0 && complete) BenchStatus.SUCCESS else BenchStatus.FAILED,
            artifacts = context.artifacts() + context.artifact("comparison.log", "text/plain"),
            message = if (exit == 0 && complete) null else "Vendor comparison exited $exit or retained an incomplete capture")
    }
}
