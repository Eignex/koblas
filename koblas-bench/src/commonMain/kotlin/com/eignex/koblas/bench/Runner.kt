package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import kotlin.math.max
import kotlin.time.TimeSource

internal expect fun readTextFile(path: String): String
internal expect fun writeTextFile(path: String, text: String)
internal expect fun resolveEngine(mode: String): Pair<KoblasEngine, String>
internal expect fun runtimeIdentity(): String
internal expect fun environment(name: String): String?
private val clockOrigin = TimeSource.Monotonic.markNow()
internal fun nanoTime(): Long = clockOrigin.elapsedNow().inWholeNanoseconds

internal data class Settings(
    val mode: String,
    val operation: String,
    val casesPath: String,
    val outputPath: String,
    val warmups: Int,
    val samples: Int,
    val targetNanos: Long,
    val forks: Int,
    val pass: String,
    val sourceCommit: String,
    val dirty: String,
)

public fun main(args: Array<String>) {
    val settings = parseArguments(args)
    require(settings.mode == "native") { "JVM benchmarks must run through the JMH entry point" }
    val allCases = Cases.parse(readTextFile(settings.casesPath))
    val selected = if (settings.operation == "all") allCases else allCases.filter { it.operation == settings.operation }
    require(selected.isNotEmpty()) { "operation '${settings.operation}' selected no cases" }
    val (engine, implementation) = resolveEngine(settings.mode)
    val rows = ArrayList<Measurement>()
    var sink = 0.0
    for (case in selected) {
        val work = denseWork(case, engine) ?: sparseWork(case, engine)
        if (work == null) {
            rows += measurement(case, implementation, settings, 0, 0, 0L, "", "unsupported", "unsupported", case.option("mode", "arithmetic"))
            continue
        }
        try {
            val calibration = warmAndCalibrate(work, settings.targetNanos, settings.warmups)
            sink += calibration.sink
            val operations = calibration.operations
            for (sample in 1..settings.samples) {
                val start = nanoTime()
                repeat(operations) { sink += work.run() }
                val elapsed = max(1L, nanoTime() - start)
                rows += measurement(case, implementation, settings, sample, operations.toLong(), elapsed, formatDouble(elapsed.toDouble() / operations), "ok", work.comparisonKind, work.timingMode)
            }
        } catch (failure: Throwable) {
            rows += measurement(case, implementation, settings, 0, 0, 0L, "", "failed:${sanitize(failure.message ?: failure::class.simpleName ?: "error")}", work.comparisonKind, work.timingMode)
            throw failure
        } finally {
            work.close()
        }
    }
    if (sink == Double.POSITIVE_INFINITY) throw IllegalStateException("unreachable result sink")
    writeTextFile(settings.outputPath, reportCsv(rows))
    println("wrote ${selected.size} cases and ${rows.size} measurements to ${settings.outputPath}")
    println("resolved implementation=$implementation runtime=${runtimeIdentity()}")
}

private data class Calibration(val operations: Int, val sink: Double)
private data class Batch(val elapsed: Long, val sink: Double)

private fun warmAndCalibrate(work: CaseWork, targetNanos: Long, warmupBatches: Int): Calibration {
    var operations = 1
    var sink = 0.0
    val warmupNanos = max(1_000_000L, targetNanos / 4)
    repeat(warmupBatches) {
        val deadline = nanoTime() + warmupNanos
        do {
            val batch = runBatch(work, operations)
            sink += batch.sink
            operations = adjustedOperations(operations, batch.elapsed, warmupNanos)
        } while (nanoTime() < deadline)
    }
    var stableBatches = 0
    repeat(16) {
        val batch = runBatch(work, operations)
        sink += batch.sink
        val acceptable = batch.elapsed in (targetNanos / 2)..(targetNanos * 2)
        stableBatches = if (acceptable) stableBatches + 1 else 0
        if (stableBatches == 2 || operations == MAX_OPERATIONS && batch.elapsed >= targetNanos / 2) {
            return Calibration(operations, sink)
        }
        operations = adjustedOperations(operations, batch.elapsed, targetNanos)
    }
    return Calibration(operations, sink)
}

private fun runBatch(work: CaseWork, operations: Int): Batch {
    var sink = 0.0
    val start = nanoTime()
    repeat(operations) { sink += work.run() }
    return Batch(max(1L, nanoTime() - start), sink)
}

private fun adjustedOperations(operations: Int, elapsed: Long, targetNanos: Long): Int {
    val ratio = (targetNanos.toDouble() / elapsed).coerceIn(0.25, 4.0)
    val adjusted = (operations * ratio).toInt().coerceIn(1, MAX_OPERATIONS)
    return if (adjusted == operations && elapsed < targetNanos) (operations + 1).coerceAtMost(MAX_OPERATIONS) else adjusted
}

internal fun parseArguments(args: Array<String>): Settings {
    val values = linkedMapOf<String, String>()
    for (argument in args) {
        require(argument.startsWith("--") && '=' in argument) { "arguments must use --name=value: $argument" }
        val (name, value) = argument.removePrefix("--").split('=', limit = 2)
        require(name !in values) { "duplicate argument --$name" }
        values[name] = value
    }
    val allowed = setOf("mode", "operation", "cases", "output", "warmups", "samples", "target-ms", "forks", "pass", "source-commit", "dirty")
    require(values.keys.all { it in allowed }) { "unknown argument: ${values.keys.first { it !in allowed }}" }
    val mode = values["mode"] ?: error("--mode is required")
    require(mode in setOf("jvm-c", "jvm-simd", "jvm-scalar", "native")) {
        "mode must be jvm-c, jvm-simd, jvm-scalar, or native"
    }
    val warmups = values["warmups"]?.toIntOrNull() ?: 3
    val samples = values["samples"]?.toIntOrNull() ?: 5
    val targetMillis = values["target-ms"]?.toLongOrNull() ?: 1_000L
    val forks = values["forks"]?.toIntOrNull() ?: 1
    require(warmups >= 0 && samples > 0 && targetMillis in 1..60_000 && forks > 0) {
        "timing settings and forks must be positive, target-ms must not exceed 60000 (warmups may be zero)"
    }
    return Settings(
        mode, values["operation"] ?: "all", values["cases"] ?: "koblas-bench/cases.txt",
        values["output"] ?: "koblas-bench/build/benchmarks/$mode.csv", warmups, samples,
        targetMillis * 1_000_000L, forks, values["pass"] ?: "1",
        values["source-commit"] ?: environment("KOBLAS_SOURCE_COMMIT") ?: "unknown",
        values["dirty"] ?: environment("KOBLAS_SOURCE_DIRTY") ?: "unknown",
    )
}

internal fun measurement(
    case: BenchCase,
    implementation: String,
    settings: Settings,
    sample: Int,
    operations: Long,
    elapsed: Long,
    nanosPerOperation: String,
    status: String,
    comparisonKind: String,
    timingMode: String,
    runtime: String = runtimeIdentity(),
): Measurement = Measurement(
    run = listOf(
        implementation, WORKLOAD_VERSION, FIXTURE_VERSION, settings.pass, "ns", settings.sourceCommit,
        settings.dirty, runtime, "1", settings.warmups.toString(), settings.targetNanos.toString(),
        if (settings.mode.startsWith("jvm")) "jmh-average-time-v1" else "native-calibrated-v1",
        (if (settings.mode.startsWith("jvm")) settings.targetNanos else max(1_000_000L, settings.targetNanos / 4)).toString(),
        settings.forks.toString(),
    ),
    case = listOf(
        case.id, status, comparisonKind, timingMode, case.logicalId, case.configurationId, case.physicalWork,
        actualPackedKernel(case, settings.mode, status),
    ),
    sample = if (status != "ok") null else listOf(
        (if (settings.mode.startsWith("jvm")) (sample - 1) / settings.samples + 1 else 1).toString(),
        sample.toString(), operations.toString(), elapsed.toString(), nanosPerOperation,
    ),
)

internal data class Measurement(val run: List<String>, val case: List<String>, val sample: List<String>?)

internal fun reportCsv(measurements: List<Measurement>): String = buildString {
    appendLine(CSV_HEADER)
    val runs = linkedMapOf<List<String>, Int>()
    val cases = linkedMapOf<List<String>, Int>()
    for (measurement in measurements) {
        val runId = runs.getOrPut(measurement.run) {
            (runs.size + 1).also { appendLine(csvRecord(listOf("run", it.toString()) + measurement.run)) }
        }
        val definition = listOf(runId.toString()) + measurement.case
        val caseId = cases.getOrPut(definition) {
            (cases.size + 1).also { appendLine(csvRecord(listOf("case", it.toString()) + definition)) }
        }
        measurement.sample?.let { appendLine(csvRecord(listOf("sample", caseId.toString()) + it)) }
    }
}

private fun csvRecord(fields: List<String>): String = fields.joinToString(",", transform = ::csv)

private fun csv(value: String): String = if (value.any { it == ',' || it == '"' || it == '\n' }) {
    "\"${value.replace("\"", "\"\"")}\""
} else value

internal fun formatDouble(value: Double): String = value.toString()
internal fun sanitize(value: String): String = value.replace(',', ';').replace('\n', ' ').take(160)

internal const val CSV_HEADER = "schema,5\n" +
    "run,id,implementation,workload_version,fixture_version,pass,unit,source_commit,dirty,runtime,threads,warmups,target_ns,harness,warmup_target_ns,forks\n" +
    "case,id,run_id,case,status,comparison_kind,timing_mode,logical_id,configuration,physical_work,actual_kernel\n" +
    "sample,case_id,fork,sample,operations,elapsed_ns,ns_per_op"
private const val MAX_OPERATIONS = 1_000_000
