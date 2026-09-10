@file:OptIn(com.eignex.koblas.ExperimentalKoblasApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.KoblasContext
import kotlin.math.max
import kotlin.time.TimeSource

internal expect fun readTextFile(path: String): String
internal expect fun writeTextFile(path: String, text: String)
internal expect fun resolveEngine(mode: String): Pair<KoblasContext, String>
internal expect fun runtimeIdentity(): String
internal expect fun environment(name: String): String?
private val clockOrigin = TimeSource.Monotonic.markNow()
internal fun nanoTime(): Long = clockOrigin.elapsedNow().inWholeNanoseconds

private data class Settings(
    val mode: String,
    val operation: String,
    val casesPath: String,
    val outputPath: String,
    val warmups: Int,
    val samples: Int,
    val targetNanos: Long,
    val pass: String,
    val sourceCommit: String,
    val dirty: String,
)

public fun main(args: Array<String>) {
    val settings = parseArguments(args)
    val allCases = Cases.parse(readTextFile(settings.casesPath))
    val selected = if (settings.operation == "all") allCases else allCases.filter { it.operation == settings.operation }
    require(selected.isNotEmpty()) { "operation '${settings.operation}' selected no cases" }
    val (engine, implementation) = resolveEngine(settings.mode)
    val rows = ArrayList<String>()
    rows += CSV_HEADER
    var sink = 0.0
    for (case in selected) {
        val work = denseWork(case, engine) ?: sparseWork(case, engine)
        if (work == null) {
            rows += csvRow(case, implementation, settings, 0, 0, 0L, "", "unsupported", "unsupported", case.option("mode", "arithmetic"))
            continue
        }
        try {
            repeat(settings.warmups) { sink += work.run() }
            val operations = calibrate(work, settings.targetNanos)
            for (sample in 1..settings.samples) {
                val start = nanoTime()
                repeat(operations) { sink += work.run() }
                val elapsed = max(1L, nanoTime() - start)
                rows += csvRow(case, implementation, settings, sample, operations, elapsed, formatDouble(elapsed.toDouble() / operations), "ok", work.comparisonKind, work.timingMode)
            }
        } catch (failure: Throwable) {
            rows += csvRow(case, implementation, settings, 0, 0, 0L, "", "failed:${sanitize(failure.message ?: failure::class.simpleName ?: "error")}", work.comparisonKind, work.timingMode)
            throw failure
        } finally {
            work.close()
        }
    }
    if (sink == Double.POSITIVE_INFINITY) throw IllegalStateException("unreachable result sink")
    writeTextFile(settings.outputPath, rows.joinToString("\n", postfix = "\n"))
    println("wrote ${selected.size} cases and ${rows.size - 1} rows to ${settings.outputPath}")
    println("resolved implementation=$implementation runtime=${runtimeIdentity()}")
}

private fun calibrate(work: CaseWork, targetNanos: Long): Int {
    var operations = 1
    while (operations < 1_000_000) {
        val start = nanoTime()
        repeat(operations) { work.run() }
        val elapsed = nanoTime() - start
        if (elapsed >= targetNanos / 4) break
        operations = (operations * 2).coerceAtMost(1_000_000)
    }
    return operations
}

private fun parseArguments(args: Array<String>): Settings {
    val values = linkedMapOf<String, String>()
    for (argument in args) {
        require(argument.startsWith("--") && '=' in argument) { "arguments must use --name=value: $argument" }
        val (name, value) = argument.removePrefix("--").split('=', limit = 2)
        require(name !in values) { "duplicate argument --$name" }
        values[name] = value
    }
    val allowed = setOf("mode", "operation", "cases", "output", "warmups", "samples", "target-ms", "pass", "source-commit", "dirty")
    require(values.keys.all { it in allowed }) { "unknown argument: ${values.keys.first { it !in allowed }}" }
    val mode = values["mode"] ?: error("--mode is required")
    require(mode in setOf("jvm-c", "jvm-simd", "native")) { "mode must be jvm-c, jvm-simd, or native" }
    val warmups = values["warmups"]?.toIntOrNull() ?: 3
    val samples = values["samples"]?.toIntOrNull() ?: 5
    val targetMillis = values["target-ms"]?.toLongOrNull() ?: 100L
    require(warmups >= 0 && samples > 0 && targetMillis > 0) { "timing settings must be positive (warmups may be zero)" }
    return Settings(
        mode, values["operation"] ?: "all", values["cases"] ?: "koblas-bench/cases.txt",
        values["output"] ?: "koblas-bench/build/benchmarks/$mode.csv", warmups, samples,
        targetMillis * 1_000_000L, values["pass"] ?: "1",
        values["source-commit"] ?: environment("KOBLAS_SOURCE_COMMIT") ?: "unknown",
        values["dirty"] ?: environment("KOBLAS_SOURCE_DIRTY") ?: "unknown",
    )
}

private fun csvRow(
    case: BenchCase,
    implementation: String,
    settings: Settings,
    sample: Int,
    operations: Int,
    elapsed: Long,
    nanosPerOperation: String,
    status: String,
    comparisonKind: String,
    timingMode: String,
): String = listOf(
    "2", case.id, implementation, WORKLOAD_VERSION, FIXTURE_VERSION, settings.pass, sample.toString(),
    operations.toString(), elapsed.toString(), nanosPerOperation, "ns", status, comparisonKind, timingMode,
    settings.sourceCommit, settings.dirty, runtimeIdentity(), "1",
).joinToString(",", transform = ::csv)

private fun csv(value: String): String = if (value.any { it == ',' || it == '"' || it == '\n' }) {
    "\"${value.replace("\"", "\"\"")}\""
} else value

private fun formatDouble(value: Double): String = value.toString()
private fun sanitize(value: String): String = value.replace(',', ';').replace('\n', ' ').take(160)

private const val CSV_HEADER = "schema,case,implementation,workload_version,fixture_version,pass,sample,operations,elapsed_ns,ns_per_op,unit,status,comparison_kind,timing_mode,source_commit,dirty,runtime,threads"
