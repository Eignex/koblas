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
        val engine = resolveEngine(mode).first
        val work = denseWork(case, engine) ?: sparseWork(case, engine)
            ?: error("unsupported case passed to JMH: $caseId")
        return JvmCaseWork(work)
    }
}

public fun main(args: Array<String>) {
    val settings = parseArguments(args)
    require(settings.mode in setOf("jvm-c", "jvm-simd", "jvm-scalar") || settings.mode.startsWith("jvm-c-raw-")) { "JMH supports only JVM benchmark modes" }
    val allCases = Cases.parse(readTextFile(settings.casesPath))
    val selected = if (settings.operation == "all") allCases else allCases.filter { it.operation == settings.operation }
    require(selected.isNotEmpty()) { "operation '${settings.operation}' selected no cases" }
    val (engine, implementation) = resolveEngine(settings.mode)
    val supported = linkedMapOf<String, Pair<String, String>>()
    val rowsByCase = linkedMapOf<String, MutableList<Measurement>>()
    for (case in selected) {
        val work = denseWork(case, engine) ?: sparseWork(case, engine)
        if (work == null) {
            rowsByCase.getOrPut(case.id, ::arrayListOf) += measurement(
                case, settings, 0, null, "unsupported", "unsupported",
                case.option("mode", "arithmetic"),
            )
        } else {
            supported[case.id] = work.comparisonKind to work.timingMode
            work.close()
        }
    }
    if (supported.isNotEmpty()) {
        val results = Runner(jmhOptions(settings, supported.keys)).run()
        appendJmhRows(rowsByCase, results, allCases.associateBy { it.id }, supported, settings)
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
                case, settings, ++sample, nanosPerOperation, "ok", comparison, timing,
            )
        }
        require(sample == settings.samples * settings.forks) { "JMH returned $sample samples for $caseId" }
    }
    for (caseId in supported.keys) require(caseId in rowsByCase) { "JMH returned no result for $caseId" }
}
