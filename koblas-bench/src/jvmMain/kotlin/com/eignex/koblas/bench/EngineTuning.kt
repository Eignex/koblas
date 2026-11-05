package com.eignex.koblas.bench

import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.engine.*
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.exp
import kotlin.math.ln
import kotlinx.serialization.json.*

/** Explicit development tool; emits Kotlin configuration and keeps evidence outside library sources. */
public object EngineTuning {
    /** Runs explicit capture, fitting or holdout validation. */
    @JvmStatic
    public fun main(args: Array<String>) {
        val options = linkedMapOf<String, MutableList<String>>()
        val allowed = setOf("write-cases", "runtime", "backend", "fallback", "output", "results", "capture", "cases",
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
        val baseline = if (runtime == "native") "native" else "jvm-owned"
        fun read(paths: List<String>, candidate: String, backend: String? = null, name: String? = null) = readCalibration(
            paths.map(Path::of), if (backend == "jvm-simd") "jvm-scalar" else baseline, candidate, backend,
            option("lab-arm"), option("cpu-class"), option("sve-vector-bits"), name,
        )
        options["validate-results"]?.let { paths ->
            val training = Json.parseToJsonElement(Files.readString(Path.of(requireNotNull(option("training")), "evidence.json"))).jsonObject
            require(training.field("runtime") == runtime)
            val checked = read(paths, "$runtime-default", name = training.field("profile_name"))
            val expected = option("validation-cases")?.let { readCases(Path.of(it)) } ?: CalibrationCases.holdout(runtime == "native")
            val measured = checked.timings.map { canonicalCase(it.case) }.toSet()
            require(expected.isNotEmpty() && expected.all { canonicalCase(it) in measured }) { "Missing holdouts or controls" }
            require(training.getValue("cases").jsonArray.none { canonicalCase(it.jsonPrimitive.content) in measured }) { "Training overlap" }
            val reached = checked.timings.map { it.provider }.toSet()
            require(training.getValue("fitted_sources").jsonArray.all { it.jsonPrimitive.content in reached }) { "A fitted provider was not exercised" }
            val regressions = checked.timings.groupBy { it.case to it.cpu }.filterValues { median(it.map { t -> t.ratio }) > 1.05 }
            Files.writeString(output.resolve("validation.json"), json(mapOf("profile_name" to training.field("profile_name"),
                "timings" to checked.timings.map(::timingEvidence), "hashes" to checked.hashes, "provenance" to checked.provenance,
                "regressions" to regressions.mapKeys { it.key.toString() }.mapValues { it.value.map(::timingEvidence) })).toString())
            require(regressions.isEmpty()) { "Combined profile or control regression; inspect validation.json" }
            return
        }
        val vendors = Vendor.entries.associateBy { it.vendorName.lowercase() }
        val sources = requireNotNull(options["backend"]).distinct().toMutableList()
        require(sources.all { it in vendors || it == "jvm-simd" && runtime == "jvm" })
        if (runtime == "jvm" && "jvm-simd" !in sources) sources += "jvm-simd"
        val fallbacks = options["fallback"].orEmpty().map { it.split('=') }.associate { parts ->
            require(parts.size == 2 && parts[0] in vendors && parts[1] in sources && parts[1] in vendors && parts[0] !in sources)
            parts[0] to parts[1]
        }
        val cases = option("cases")?.let(Path::of) ?: output.resolve("training.txt").also {
            Files.writeString(it, CalibrationCases.training(runtime == "native").joinToString("\n", postfix = "\n"))
        }
        val targets = (if (runtime == "native") listOf("native") else listOf("jvm-scalar", "jvm-simd")) +
            sources.map { if (it == "jvm-simd" || runtime == "native") it else "$it-jvm" }
        val directories = options["results"] ?: mutableListOf(output.resolve("capture").toString())
        require(options.containsKey("capture") != options.containsKey("results")) { "Choose --capture or --results" }
        if (options.containsKey("capture")) {
            val command = mutableListOf(captureShell(), "koblas-bench/capture-report.sh", "--prepared", "--targets", targets.distinct().joinToString(","),
                "--cases", cases.toString(), "--rounds", "3", "--samples", "5", "--warmups", "20", "--target-ms", "200",
                "--forks", "2", "--output", directories.single())
            sources.filter { it in vendors }.takeIf { it.isNotEmpty() }?.let { command += listOf("--libraries", it.joinToString(",")) }
            option("native-executable")?.let { command += listOf("--native-executable", it) }
            val process = ProcessBuilder(command).inheritIO().apply {
                environment()["KOBLAS_BENCH_JAVA"] = Path.of(System.getProperty("java.home"), "bin", "java").toString()
                environment()["KOBLAS_BENCH_CLASSPATH"] = System.getProperty("java.class.path")
            }
            check(process.start().waitFor() == 0) { "Capture failed; retained outputs are in ${directories.single()}" }
        }
        val captures = sources.associateWith { source -> read(directories,
            if (source == "jvm-simd" || runtime == "native") source else "$source-jvm", source) }
        val requested = readCases(cases).map(::canonicalCase).toSet()
        val captured = captures.values.flatMap { it.cases }.map(::canonicalCase).toSet()
        require(requested.all { it in captured }) { "Missing requested training cases" }
        val target = captures.values.first().target.toMutableList()
        require(target[0] == runtime && captures.values.all { it.target == target }) { "Composition targets differ" }
        require(option("cpu") != null || target[3] != "*") { "Use --cpu * explicitly when the runtime CPU model is unknown" }
        val model = option("cpu") ?: target[3]
        require(model == "*" || model == target[3])
        if (model != "*") require(captures.values.all { capture -> capture.timings.all { it.cpu == model } })
        target[3] = model
        OperatingSystem.valueOf(target[1]); Architecture.valueOf(target[2]); CpuClass.valueOf(target[4])
        val rules = composeProfiles(captures)
        for ((source, donor) in fallbacks) rules += rules.filter { it.source == donor }.map { rule ->
            EngineRule(source, rule.implementations, rule.rows, rule.columns, rule.depth, rule.work, rule.contiguous, rule.borrowed)
        }
        rules += captures["jvm-simd"]?.let { fitProfile("jvm-simd", it.timings) }.orEmpty()
        val fittedSources = rules.map { it.source }.filter { it !in fallbacks }.distinct()
        // Fixed owned arithmetic comes last, after all whole-call vendor preferences.
        if (runtime == "jvm") rules += EngineRule("jvm-simd", setOf(EngineImplementation.Panels, EngineImplementation.Products, EngineImplementation.Triangular))
        val name = "$runtime-${target[1].lowercase()}-${target[2].lowercase()}-${target[4].lowercase()}-${captures.values.first().hashes.values.first().take(12)}"
        val code = renderProfile(name, target, rules)
        Files.writeString(output.resolve("Profile.kt"), code)
        Files.writeString(output.resolve("evidence.json"), json(mapOf("profile_name" to name, "runtime" to runtime,
            "target" to target, "profile_sha256" to digest(output.resolve("Profile.kt")), "cases" to captured,
            "fitted_sources" to fittedSources, "estimates" to fallbacks,
            "sources" to captures.mapValues { (_, capture) -> mapOf("hashes" to capture.hashes, "provenance" to capture.provenance,
                "timings" to capture.timings.map(::timingEvidence)) })).toString())
        println("Wrote ${rules.size} broad routing preferences to ${output.resolve("Profile.kt")}; rebuild before capturing default-engine holdouts.")
    }
}

internal fun canonicalCase(case: String): String = case.split('+').let { (it.take(2) + it.drop(2).sorted()).joinToString("+") }
private fun timingEvidence(t: PairedTiming): Map<String, Any?> = mapOf("case" to t.case, "cpu" to t.cpu, "provider" to t.provider,
    "routing" to t.shape.toString(), "ratio" to t.ratio)

internal fun fitProfile(source: String, timings: List<PairedTiming>): List<EngineRule> = buildList {
    val shapes = timings.filter { it.shape != null }.groupBy { it.shape!! }
    fun wins(shape: RoutingShape): Boolean = shapes.getValue(shape).groupBy { it.cpu to it.case }.values.all {
        median(it.map { t -> t.ratio }) <= 0.95
    }
    for ((_, keys) in shapes.keys.groupBy { listOf(it.operation, it.contiguous, it.borrowed, it.dimensions) }) {
        val first = keys.first()
        if (first.dimensions == 1) {
            val ordered = keys.sortedBy { it.rows }
            val minimum = ordered.indices.firstOrNull { i -> ordered.drop(i).all(::wins) } ?: continue
            add(EngineRule(source, setOf(first.operation), rows = ordered[minimum].rows, contiguous = first.contiguous))
        } else {
            // A single broad work/extent region replaces exact shape islands.
            val regions = keys.filter(::wins).mapNotNull { start ->
                val covered = keys.filter { it.extent >= start.extent && it.work >= start.work }
                if (covered.all(::wins)) start to covered.size else null
            }
            val start = regions.maxWithOrNull(compareBy<Pair<RoutingShape, Int>> { it.second }.thenBy { -it.first.work })?.first ?: continue
            add(EngineRule(source, setOf(first.operation), rows = start.extent, columns = start.extent,
                depth = (if (first.dimensions == 3) start.extent else 0), work = start.work, borrowed = first.borrowed))
        }
    }
}

internal fun composeProfiles(captures: Map<String, CalibrationCapture>): MutableList<EngineRule> {
    val fitted = captures.mapValues { (source, capture) -> fitProfile(source, capture.timings) }
    val result = mutableListOf<EngineRule>()
    for (operation in EngineImplementation.entries) {
        fun covered(source: String, t: PairedTiming): Boolean = t.shape?.let { s -> fitted.getValue(source).any { rule ->
            operation in rule.implementations && rule.matches(s.rows, s.columns, s.depth, s.contiguous, s.borrowed)
        } } == true
        val sources = captures.keys.filter { it != "jvm-simd" && fitted.getValue(it).any { rule -> operation in rule.implementations } }
        val common = sources.map { source -> captures.getValue(source).timings.filter { covered(source, it) }.map { it.case to it.cpu }.toSet() }
            .reduceOrNull { left, right -> left.intersect(right) }.orEmpty()
        for (source in sources.sortedBy { source ->
            val ratios = captures.getValue(source).timings.filter { it.case to it.cpu in common }.map { ln(it.ratio) }
            if (ratios.isEmpty()) 0.0 else exp(ratios.average())
        }) result += fitted.getValue(source).filter { operation in it.implementations }
    }
    return result
}

internal fun renderProfile(name: String, target: List<String>, rules: List<EngineRule>): String = buildString {
    append("package com.eignex.koblas.internal.engine\n\nimport com.eignex.koblas.engine.EngineImplementation.*\n\n")
    append("internal val recalculatedTuning = EngineTuning(\n    ${JsonPrimitive(name)},\n    listOf(\n")
    for (rule in rules) {
        append("        EngineRule(${JsonPrimitive(rule.source)}, setOf(${rule.implementations.joinToString { it.name }}),\n")
        append("            rows = ${rule.rows}, columns = ${rule.columns}, depth = ${rule.depth},\n")
        append("            work = ${rule.work}, contiguous = ${rule.contiguous}, borrowed = ${rule.borrowed}),\n")
    }
    append("    ),\n    evidence = ${JsonPrimitive("Recalculated for ${target.joinToString("/")}; retained evidence.json accompanies this source")},\n)\n")
}
