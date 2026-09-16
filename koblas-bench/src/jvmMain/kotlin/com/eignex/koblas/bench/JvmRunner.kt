package com.eignex.koblas.bench

import java.io.File
import java.util.concurrent.TimeUnit
import org.openjdk.jmh.results.RunResult
import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import org.openjdk.jmh.runner.options.TimeValue

public class JvmCaseWork internal constructor(private val delegate: CaseWork) {
    public fun run(): Double = delegate.run()
    public fun close(): Unit = delegate.close()
}

public object JvmBenchmarkBridge {
    @JvmStatic
    public fun create(mode: String, caseId: String, casesPath: String): JvmCaseWork {
        val case = Cases.parse(readTextFile(casesPath)).single { it.id == caseId }
        if (vendorForRuntime(mode, JVM_VENDOR_PREFIX) != null) {
            val arm = vendorArm(case, openVendorForMode(mode).first)
            // The fork already accepted this case, so a rejection here means the arm changed between the
            // scan and the measurement rather than that the case was never admissible.
            return JvmCaseWork(arm.work ?: error("vendor arm declined $caseId in the measured fork: ${arm.reason}"))
        }
        val engine = resolveEngine(mode).first
        val sparse = sparseArm(case, engine)
        // Same contract as the vendor arm above: the scan admitted this case, so a decline now is a change of
        // route between the scan and the measurement, not an inadmissible case.
        if (sparse != null) {
            return JvmCaseWork(sparse.work ?: error("sparse arm declined $caseId in the measured fork: ${sparse.reason}"))
        }
        val work = denseWork(case, engine) ?: error("unsupported case passed to JMH: $caseId")
        return JvmCaseWork(work)
    }
}

public fun main(args: Array<String>) {
    val settings = parseArguments(args)
    val vendorMode = vendorForRuntime(settings.mode, JVM_VENDOR_PREFIX) != null
    require(
        settings.mode in setOf("jvm-simd", "jvm-scalar") || vendorMode,
    ) { "JMH supports only JVM benchmark modes" }
    val allCases = Cases.parse(readTextFile(settings.casesPath))
    val selected = Cases.select(allCases, settings.suite, settings.operation)
    val vendor = if (vendorMode) openVendorForMode(settings.mode) else null
    val resolved = if (vendorMode) null else resolveEngine(settings.mode)
    val engine = resolved?.first
    val implementation = vendor?.second ?: checkNotNull(resolved).second
    val supported = linkedMapOf<String, Pair<String, String>>()
    val kernels = linkedMapOf<String, String>()
    val rowsByCase = linkedMapOf<String, MutableList<Measurement>>()
    for (case in selected) {
        // Routes are resolved here, before any timing, so describing a call costs nothing inside the
        // measured loop and an inadmissible arm is declined rather than timed.
        val arm = vendor?.let { vendorArm(case, it.first) }
            ?: engine?.let { sparseArm(case, it) ?: denseWork(case, it)?.let { work -> ArmChoice(work, null) } }
        val work = arm?.work
        if (work == null) {
            rowsByCase.getOrPut(case.id, ::arrayListOf) += measurement(
                case, settings, 0, null, "unsupported", "unsupported",
                case.option("timing", "arithmetic"), arm?.reason,
            )
        } else {
            supported[case.id] = work.comparisonKind to work.timingMode
            work.kernel?.let { kernels[case.id] = it }
            work.close()
        }
    }
    if (supported.isNotEmpty()) {
        val results = Runner(jmhOptions(settings, supported.keys)).run()
        appendJmhRows(rowsByCase, results, allCases.associateBy { it.id }, supported, settings, kernels)
    }
    writeRows(settings, selected, rowsByCase)
    println("wrote ${selected.size} case summaries from ${rowsByCase.values.sumOf { rows -> rows.count { it.nanos != null } }} measurements to ${settings.outputPath}")
    println("resolved implementation=$implementation runtime=${runtimeIdentity()} harness=jmh-average-time/JMH-1.37")
}

private fun writeRows(settings: Settings, selected: List<BenchCase>, rowsByCase: Map<String, List<Measurement>>) {
    val rows = arrayListOf<Measurement>()
    for (case in selected) rows += requireNotNull(rowsByCase[case.id]) { "JMH returned no result for ${case.id}" }
    writeTextFile(settings.outputPath, reportCsv(rows))
}

private fun jmhOptions(settings: Settings, caseIds: Collection<String>) = OptionsBuilder()
    .include("^${Regex.escape(KoblasJmhBenchmark::class.java.name)}\\.run$")
    .param("benchmarkMode", settings.mode)
    .param("caseId", *caseIds.toTypedArray())
    .warmupIterations(settings.warmups)
    .warmupTime(TimeValue(settings.targetNanos, TimeUnit.NANOSECONDS))
    .measurementIterations(settings.samples)
    .measurementTime(TimeValue(settings.targetNanos, TimeUnit.NANOSECONDS))
    .forks(settings.forks)
    .threads(1)
    .shouldFailOnError(true)
    .jvmArgsAppend("-Dkoblas.bench.cases=${File(settings.casesPath).absolutePath}")
    .build()

private fun appendJmhRows(
    rowsByCase: MutableMap<String, MutableList<Measurement>>,
    results: Collection<RunResult>,
    cases: Map<String, BenchCase>,
    supported: Map<String, Pair<String, String>>,
    settings: Settings,
    kernels: Map<String, String>,
) {
    for (result in results.sortedBy { it.params.getParam("caseId") }) {
        val caseId = result.params.getParam("caseId")
        val case = requireNotNull(cases[caseId])
        val (comparison, timing) = requireNotNull(supported[caseId])
        var sample = 0
        for (fork in result.benchmarkResults) for (iteration in fork.iterationResults) {
            val operations = iteration.metadata.measuredOps
            val nanosPerOperation = iteration.primaryResult.score
            require(operations > 0 && nanosPerOperation.isFinite() && nanosPerOperation > 0) {
                "invalid JMH result for $caseId"
            }
            rowsByCase.getOrPut(caseId, ::arrayListOf) += measurement(
                case, settings, ++sample, nanosPerOperation, "ok", comparison, timing, kernels[caseId],
            )
        }
        require(sample == settings.samples * settings.forks) { "JMH returned $sample samples for $caseId" }
    }
    for (caseId in supported.keys) require(caseId in rowsByCase) { "JMH returned no result for $caseId" }
}
