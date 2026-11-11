package com.eignex.koblas.bench

import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.*
import java.nio.file.Files
import java.nio.file.Path
import kotlinx.serialization.json.*

/** Explicit development tool; reports measured cutoffs and keeps evidence outside library sources. */
public object EngineTuning {
    /** Runs explicit capture, fitting or holdout validation. */
    @JvmStatic
    public fun main(args: Array<String>) {
        val options = linkedMapOf<String, MutableList<String>>()
        val allowed = setOf("write-cases", "baseline", "runtime", "backend", "engine-name", "output", "results", "capture", "cases",
            "validate-results", "training", "validation-cases", "lab-arm", "cpu", "cpu-class", "sve-vector-bits", "native-executable")
        var i = 0
        while (i < args.size) {
            val key = args[i++].removePrefix("--")
            require(key in allowed) { "Unknown argument $key" }
            options.getOrPut(key) { mutableListOf() } += if (key == "capture") "true" else args[i++]
        }
        fun option(key: String): String? = options[key]?.single()
        option("write-cases")?.let { CalibrationCases.write(Path.of(it)); return }
        val output = Path.of(requireNotNull(option("output")))
        require(!Files.exists(output)) { "Output already exists: $output" }
        Files.createDirectories(output)
        val runtime = option("runtime") ?: "native"
        require(runtime in listOf("native", "jvm"))
        var baseline = calibrationBaseline(runtime, option("baseline"))
        fun read(paths: List<String>, candidate: String, backend: String? = null, name: String? = null) = readCalibration(
            paths.map(Path::of), if (backend == "jvm-simd") "jvm-scalar" else baseline, candidate, backend,
            option("lab-arm"), option("cpu-class"), option("sve-vector-bits"), name,
        )
        options["validate-results"]?.let { paths ->
            val training = Json.parseToJsonElement(Files.readString(Path.of(requireNotNull(option("training")), "evidence.json"))).jsonObject
            require(training.field("runtime") == runtime)
            val recordedBaseline = calibrationBaseline(runtime, training.field("baseline").takeIf(String::isNotEmpty))
            require(option("baseline") == null || baseline == recordedBaseline) { "Validation baseline differs from training" }
            baseline = recordedBaseline
            val candidate = if (baseline == "jvm-scalar" && training.field("backend") != "jvm-simd") "jvm-default-no-simd" else "$runtime-default"
            val checked = read(paths, candidate, name = requireNotNull(option("engine-name")) { "Specify the rebuilt engine name" })
            val expected = option("validation-cases")?.let { readCases(Path.of(it)) } ?: CalibrationCases.holdout(runtime == "native")
            val measured = checked.timings.map { canonicalCase(it.case) }.toSet()
            require(expected.isNotEmpty() && expected.all { canonicalCase(it) in measured }) { "Missing holdouts or controls" }
            require(training.getValue("cases").jsonArray.none { canonicalCase(it.jsonPrimitive.content) in measured }) { "Training overlap" }
            val reached = checked.timings.map { it.provider }.toSet()
            require(training.field("backend") in reached) { "The fitted provider was not exercised" }
            val regressions = checked.timings.groupBy { it.case to it.cpu }.filterValues { median(it.map { t -> t.ratio }) > 1.05 }
            Files.writeString(output.resolve("validation.json"), json(mapOf("engine_name" to option("engine-name"), "baseline" to baseline,
                "timings" to checked.timings.map(::timingEvidence), "hashes" to checked.hashes, "provenance" to checked.provenance,
                "regressions" to regressions.mapKeys { it.key.toString() }.mapValues { it.value.map(::timingEvidence) })).toString())
            require(regressions.isEmpty()) { "Selected engine or control regression; inspect validation.json" }
            return
        }
        val vendors = Vendor.entries.associateBy { it.vendorName.lowercase() }
        val source = requireNotNull(option("backend")) { "Specify one backend per fit" }
        require(source in vendors || source == "jvm-simd" && runtime == "jvm")
        require(source != "jvm-simd" || option("baseline") == null || baseline == "jvm-scalar") {
            "SIMD calibration requires the scalar baseline"
        }
        val cases = option("cases")?.let(Path::of) ?: output.resolve("training.txt").also {
            Files.writeString(it, CalibrationCases.training(runtime == "native").joinToString("\n", postfix = "\n"))
        }
        val targets = (if (source == "jvm-simd") listOf("jvm-scalar") else calibrationTargets(baseline)) +
            listOf(if (source == "jvm-simd" || runtime == "native") source else "$source-jvm")
        val directories = options["results"] ?: mutableListOf(output.resolve("capture").toString())
        require(options.containsKey("capture") != options.containsKey("results")) { "Choose --capture or --results" }
        if (options.containsKey("capture")) {
            val command = mutableListOf(captureShell(), "koblas-bench/capture-report.sh", "--prepared", "--targets", targets.distinct().joinToString(","),
                "--cases", cases.toString(), "--rounds", "3", "--samples", "5", "--warmups", "20", "--target-ms", "200",
                "--forks", "2", "--output", directories.single())
            if (source in vendors) command += listOf("--libraries", source)
            option("native-executable")?.let { command += listOf("--native-executable", it) }
            val process = ProcessBuilder(command).inheritIO().apply {
                environment()["KOBLAS_BENCH_JAVA"] = Path.of(System.getProperty("java.home"), "bin", "java").toString()
                environment()["KOBLAS_BENCH_CLASSPATH"] = System.getProperty("java.class.path")
            }
            check(process.start().waitFor() == 0) { "Capture failed; retained outputs are in ${directories.single()}" }
        }
        val capture = read(directories, if (source == "jvm-simd" || runtime == "native") source else "$source-jvm", source)
        val requested = readCases(cases).map(::canonicalCase).toSet()
        val captured = capture.cases.map(::canonicalCase).toSet()
        require(requested.all { it in captured }) { "Missing requested training cases" }
        val target = capture.target.toMutableList()
        require(target[0] == runtime)
        require(option("cpu") != null || target[3] != "*") { "Use --cpu * explicitly when the runtime CPU model is unknown" }
        val model = option("cpu") ?: target[3]
        require(model == "*" || model == target[3])
        if (model != "*") require(capture.timings.all { it.cpu == model })
        target[3] = model
        OperatingSystem.valueOf(target[1]); Architecture.valueOf(target[2]); CpuClass.valueOf(target[4])
        val cutoffs = fitCutoffs(capture.timings)
        Files.writeString(output.resolve("cutoffs.tsv"), renderCutoffs(cutoffs))
        Files.writeString(output.resolve("evidence.json"), json(mapOf("backend" to source, "runtime" to runtime, "baseline" to (if (source == "jvm-simd") "jvm-scalar" else baseline),
            "target" to target, "cutoffs_sha256" to digest(output.resolve("cutoffs.tsv")), "cases" to captured,
            "hashes" to capture.hashes, "provenance" to capture.provenance,
            "timings" to capture.timings.map(::timingEvidence))).toString())
        println("Wrote ${cutoffs.size} measured cutoffs to ${output.resolve("cutoffs.tsv")}; update EngineTuning and rebuild before holdouts.")
    }
}

internal fun canonicalCase(case: String): String = case.split('+').let { (it.take(2) + it.drop(2).sorted()).joinToString("+") }
private fun timingEvidence(t: PairedTiming): Map<String, Any?> = mapOf("case" to t.case, "cpu" to t.cpu, "provider" to t.provider,
    "routing" to t.shape.toString(), "ratio" to t.ratio)

/** A measurement report, never loaded by the engine. */
internal data class FittedCutoff(val operation: EngineImplementation, val rows: Int, val columns: Int = 0,
    val depth: Int = 0, val work: Long = 0, val contiguous: Boolean? = null, val borrowed: Boolean? = null)

internal fun fitCutoffs(timings: List<PairedTiming>): List<FittedCutoff> = buildList {
    val shapes = timings.filter { it.shape != null }.groupBy { it.shape!! }
    fun wins(shape: RoutingShape): Boolean = shapes.getValue(shape).groupBy { it.cpu to it.case }.values.all {
        median(it.map { t -> t.ratio }) <= 0.95
    }
    for ((_, keys) in shapes.keys.groupBy { listOf(it.operation, it.contiguous, it.borrowed, it.dimensions) }) {
        val first = keys.first()
        if (first.dimensions == 1) {
            val ordered = keys.sortedBy { it.rows }
            val minimum = ordered.indices.firstOrNull { i -> ordered.drop(i).all(::wins) } ?: continue
            add(FittedCutoff(first.operation, rows = ordered[minimum].rows, contiguous = first.contiguous))
        } else {
            // A single broad work/extent region replaces exact shape islands.
            val regions = keys.filter(::wins).mapNotNull { start ->
                val covered = keys.filter { it.extent >= start.extent && it.work >= start.work }
                if (covered.all(::wins)) start to covered.size else null
            }
            val start = regions.maxWithOrNull(compareBy<Pair<RoutingShape, Int>> { it.second }.thenBy { -it.first.work })?.first ?: continue
            add(FittedCutoff(first.operation, rows = start.extent, columns = start.extent,
                depth = (if (first.dimensions == 3) start.extent else 0), work = start.work, borrowed = first.borrowed))
        }
    }
}

internal fun renderCutoffs(cutoffs: List<FittedCutoff>): String = buildString {
    append("operation\trows\tcolumns\tdepth\twork\tcontiguous\tborrowed\n")
    for (cutoff in cutoffs) append(listOf(cutoff.operation, cutoff.rows, cutoff.columns, cutoff.depth, cutoff.work,
        cutoff.contiguous ?: "*", cutoff.borrowed ?: "*").joinToString("\t", postfix = "\n"))
}

/** The fallback measured by one fit, independent of which optional modules the fitting JVM loaded. */
internal fun calibrationBaseline(runtime: String, requested: String?): String {
    val baseline = requested ?: if (runtime == "native") "native" else "jvm-owned"
    require(if (runtime == "native") baseline == "native" else baseline in setOf("jvm-owned", "jvm-scalar", "jvm-simd")) {
        "Choose native for Native, or jvm-owned, jvm-scalar or jvm-simd for JVM"
    }
    return baseline
}

internal fun calibrationTargets(baseline: String): List<String> =
    if (baseline == "jvm-owned") listOf("jvm-scalar", "jvm-simd") else listOf(baseline)
