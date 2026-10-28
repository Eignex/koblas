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

    override fun describe() = BenchDescriptor("koblas-vendor-comparison", "Matched JVM and native vendor calls")

    override fun plan(request: BenchRequest): List<LabCase> {
        val p = request.parameters
        require(p.keys.all { it in setOf("vendor", "operation") }) { "Unknown vendor comparison parameter" }
        val vendor = p.getValue("vendor")
        require(vendor in setOf("onemkl", "accelerate")) { "vendor must be onemkl or accelerate" }
        p["operation"]?.let { require(it.matches(Regex("[a-z][a-z0-9-]*"))) { "Invalid operation" } }
        val tags = if (vendor == "accelerate") setOf("platform:mac", "arch:arm64") else setOf("platform:linux", "arch:x86_64")
        return listOf(LabCase("comparison/$vendor/${p["operation"] ?: "all"}", p,
            BenchResources(2, 4608, 7200, tags), inputs = listOf(BenchInput(CASES, role = "cases"))))
    }

    override fun run(case: LabCase, context: BenchContext): BenchResult {
        plan(BenchRequest(case.parameters))
        val root = requireNotNull(context.worktree)
        val output = context.reportPath()
        val argv = listOf("bash", root.resolve("koblas-bench/lab/vendor-capture.sh").toString(),
            "--output", output.toString(), "--vendor", case.parameters.getValue("vendor"),
            "--operation", case.parameters["operation"] ?: "all")
        val exit = context.process(argv, "comparison.log")
        val report = output.resolve("report")
        val complete = Files.isRegularFile(report.resolve("metadata.txt")) &&
            Files.readAllLines(report.resolve("metadata.txt")).firstOrNull() == "status=complete"
        return BenchResult(case.id, if (exit == 0 && complete) BenchStatus.SUCCESS else BenchStatus.FAILED,
            artifacts = context.artifacts() + context.artifact("comparison.log", "text/plain"),
            message = if (exit == 0 && complete) null else "Vendor comparison exited $exit or retained an incomplete capture")
    }
}
