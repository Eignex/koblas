package com.eignex.koblas.bench

/** A status arm keeps its requested engine separate from its measured provider and kernel. */
internal data class StatusTiming(val engine: String, val nanos: Double?, val provider: String, val kernel: String) {
    val metric: String get() = statusMetric(engine)
}

internal fun statusMetric(engine: String): String = if (engine == "jvm-default") "median_ns" else "median_ns/$engine"

/** Only known view adapters may decline exact engines; a missing or malformed timing fails the row. */
internal fun readStatusTiming(engine: String, case: BenchCase, samples: Int, csv: String): StatusTiming {
    val rows = readTimingCsv(csv)
    require(rows.size == 1 && rows.keys.single() == case.id) { "Sweep needs exactly its requested workload" }
    val row = rows.values.single()
    val unsupported = row.getValue("status") == "unsupported"
    if (unsupported) {
        require(case.operation in setOf("copy-overlap", "axpy-views", "spdot-view") &&
            engine !in setOf("jvm-default", "native-default")) { "$engine declined a sweep workload" }
        require(row.getValue("samples") == "0" && row.getValue("forks") == "0" && row.getValue("median_ns").isEmpty())
        return StatusTiming(engine, null, "", row.getValue("actual_kernel"))
    }
    require(row.getValue("status") == "ok") { "$engine failed a sweep workload" }
    require(row.getValue("samples").toInt() == samples && row.getValue("forks") == "1")
    val nanos = row.getValue("median_ns").toDouble()
    require(nanos.isFinite() && nanos > 0)
    val provider = row.getValue("provider"); val kernel = row.getValue("actual_kernel")
    require(provider.isNotBlank() && kernel.isNotBlank()) { "$engine timing needs kernel attribution" }
    return StatusTiming(engine, nanos, provider, kernel)
}
