package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import kotlin.math.max
import kotlin.time.TimeSource

internal expect fun readTextFile(path: String): String
internal expect fun writeTextFile(path: String, text: String)
internal expect fun resolveEngine(mode: String): Pair<KoblasEngine, String>
internal expect fun runtimeIdentity(): String
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
    val suite: String = "default",
)

public fun main(args: Array<String>) {
    val settings = parseArguments(args)
    val vendorMode = vendorForRuntime(settings.mode, NATIVE_VENDOR_PREFIX) != null
    require(settings.mode == "native" || settings.mode.startsWith("native-raw-") || vendorMode) {
        "JVM benchmarks must run through the JMH entry point"
    }
    val allCases = Cases.parse(readTextFile(settings.casesPath))
    val selected = Cases.select(allCases, settings.suite, settings.operation)
    val vendor = if (vendorMode) openVendorForMode(settings.mode) else null
    val resolved = if (vendorMode) null else resolveEngine(settings.mode)
    val engine = resolved?.first
    val implementation = vendor?.second ?: checkNotNull(resolved).second
    val rows = ArrayList<Measurement>()
    var sink = 0.0
    for (case in selected) {
        val arm = vendor?.let { vendorArm(case, it.first) }
            ?: engine?.let { sparseArm(case, it) ?: denseWork(case, it)?.let { work -> ArmChoice(work, null) } }
        val work = arm?.work
        if (work == null) {
            rows += measurement(
                case, settings, 0, null, "unsupported",
                arm?.reason ?: "unsupported", "arithmetic",
            )
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
                rows += measurement(
                    case, settings, sample, elapsed.toDouble() / operations, "ok",
                    work.comparisonKind, work.timingMode, work.kernel,
                )
            }
        } finally {
            work.close()
        }
    }
    if (sink == Double.POSITIVE_INFINITY) throw IllegalStateException("unreachable result sink")
    writeTextFile(settings.outputPath, reportCsv(rows))
    println("wrote ${selected.size} case summaries from ${rows.count { it.nanos != null }} measurements to ${settings.outputPath}")
    println("resolved implementation=$implementation runtime=${runtimeIdentity()} harness=native-calibrated")
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
    val allowed = setOf("mode", "operation", "suite", "cases", "output", "warmups", "samples", "target-ms", "forks")
    require(values.keys.all { it in allowed }) { "unknown argument: ${values.keys.first { it !in allowed }}" }
    val mode = values["mode"] ?: error("--mode is required")
    require(
        mode in setOf("jvm-c", "jvm-simd", "jvm-scalar", "native") ||
            rawNativeVariant(mode) != null || vendorFromMode(mode) != null,
    ) {
        "mode must be jvm-c, jvm-simd, jvm-scalar, native, or a jvm-vendor-/native-vendor- arm"
    }
    val warmups = values["warmups"]?.toIntOrNull() ?: 3
    val samples = values["samples"]?.toIntOrNull() ?: 5
    val targetMillis = values["target-ms"]?.toLongOrNull() ?: 1_000L
    val forks = values["forks"]?.toIntOrNull() ?: 1
    require(warmups >= 0 && samples > 0 && targetMillis in 1..60_000 && forks > 0) {
        "timing settings and forks must be positive, target-ms must not exceed 60000 (warmups may be zero)"
    }
    val suite = values["suite"] ?: "default"
    val operation = values["operation"] ?: "all"
    Cases.validateSelection(suite, operation)
    return Settings(
        mode, operation, values["cases"] ?: "koblas-bench/cases.txt",
        values["output"] ?: "koblas-bench/build/benchmarks/$mode.csv", warmups, samples,
        targetMillis * 1_000_000L, forks, suite,
    )
}

internal fun measurement(
    case: BenchCase,
    settings: Settings,
    sample: Int,
    nanos: Double?,
    status: String,
    comparisonKind: String,
    timingMode: String,
    kernel: String? = null,
): Measurement = Measurement(
    listOf(
        case.id, status, comparisonKind, timingMode,
        kernel ?: actualPackedKernel(case, settings.mode, status),
    ),
    if (settings.mode.startsWith("jvm")) (sample - 1) / settings.samples + 1 else 1,
    nanos,
)

internal data class Measurement(val case: List<String>, val fork: Int, val nanos: Double?)

internal fun reportCsv(measurements: List<Measurement>): String = buildString {
    appendLine(CSV_HEADER)
    for ((case, rows) in measurements.groupBy { it.case }) {
        val samples = rows.filter { it.nanos != null }
        val values = samples.map { requireNotNull(it.nanos) }.sorted()
        require(values.all { it.isFinite() && it > 0 }) { "invalid measured timing" }
        val statistics = if (values.isEmpty()) listOf("", "", "") else {
            val middle = values.size / 2
            val median = if (values.size % 2 == 1) values[middle] else values[middle - 1] / 2 + values[middle] / 2
            listOf(median, values.first(), values.last()).map(Double::toString)
        }
        val counts = listOf(samples.size.toString(), samples.map { it.fork }.distinct().size.toString())
        appendLine(csvRecord(case + counts + statistics))
    }
}

private fun csvRecord(fields: List<String>): String = fields.joinToString(",", transform = ::csv)

private fun csv(value: String): String = if (value.any { it == ',' || it == '"' || it == '\n' }) {
    "\"${value.replace("\"", "\"\"")}\""
} else value

internal const val CSV_HEADER = "case,status,comparison_kind,timing_mode,actual_kernel,samples,forks,median_ns,min_ns,max_ns"
private const val MAX_OPERATIONS = 1_000_000
