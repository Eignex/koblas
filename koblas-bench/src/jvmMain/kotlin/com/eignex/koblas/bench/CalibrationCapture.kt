package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.EngineImplementation
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import kotlinx.serialization.json.*

internal fun digest(path: Path): String = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path))
    .joinToString("") { "%02x".format(it) }
internal fun json(value: Any?): JsonElement = when (value) {
    null -> JsonNull
    is JsonElement -> value
    is Map<*, *> -> JsonObject(value.entries.associate { it.key.toString() to json(it.value) })
    is Iterable<*> -> JsonArray(value.map(::json))
    is Number -> JsonPrimitive(value)
    is Boolean -> JsonPrimitive(value)
    else -> JsonPrimitive(value.toString())
}
internal fun JsonElement.field(key: String): String = jsonObject[key]?.jsonPrimitive?.content.orEmpty()
internal fun median(values: List<Double>): Double = values.sorted().let { (it[(it.size - 1) / 2] + it[it.size / 2]) / 2 }

internal data class RoutingShape(
    val operation: EngineImplementation, val rows: Int, val columns: Int, val depth: Int,
    val borrowed: Boolean, val dimensions: Int, val contiguous: Boolean,
) {
    val work: Long get() = rows.toLong() * columns * depth
    val extent: Int get() = if (dimensions == 3) minOf(rows, columns, depth) else minOf(rows, columns)
}
internal data class PairedTiming(
    val case: String, val cpu: String, val provider: String, val shape: RoutingShape?, val ratio: Double,
)
internal data class CalibrationCapture(
    val target: List<String>, val cases: Set<String>, val timings: List<PairedTiming>, val hashes: Map<String, String>, val provenance: List<Any>,
)

/** Reads complete paired attempts, retaining their provenance and the job's unsuccessful attempt costs. */
internal fun readCalibration(
    directories: List<Path>, baseline: String, candidate: String, backend: String? = null,
    arm: String? = null, cpuClass: String? = null, sveBits: String? = null, profileName: String? = null,
): CalibrationCapture {
    require(directories.isNotEmpty() && directories.map { it.toRealPath() }.distinct().size == directories.size)
    val targets = (if (baseline == "jvm-owned") listOf("jvm-scalar", "jvm-simd") else listOf(baseline)) + candidate
    val hashes = linkedMapOf<String, String>()
    val provenance = mutableListOf<Any>()
    val timings = mutableListOf<PairedTiming>()
    val cases = mutableSetOf<String>()
    val platforms = mutableSetOf<List<String>>()
    val manifests = mutableSetOf<List<String>>()
    val protocols = mutableSetOf<Map<String, String>>()
    val environments = mutableMapOf<String, List<Any>>()
    val jobs = mutableSetOf<String>()
    fun remember(path: Path): String = digest(path).also { hashes[path.toString()] = it }
    for (directory in directories) {
        val index = directory.resolve("lab-cases.jsonl")
        val groups = if (!Files.exists(index)) {
            require(arm == null) { "--lab-arm requires a fetched job" }
            listOf(Files.list(directory).use { paths -> paths.filter { it.fileName.toString().matches(Regex("pass-[0-9]+")) }.toList() })
        } else {
            remember(index)
            val snapshots = listOf("lab-job.json", "lab-execution.json").associateWith { name ->
                val path = directory.resolve(name); remember(path); Json.parseToJsonElement(Files.readString(path))
            }
            val job = snapshots.getValue("lab-job.json")
            require(job.field("status") == "DONE" && jobs.add(job.field("id"))) { "Require distinct completed lab jobs" }
            val rows = Files.readAllLines(index).filter(String::isNotBlank).map { Json.parseToJsonElement(it).jsonObject }
            require(rows.map { it.field("index").toInt() }.let { it.all { n -> n >= 0 } && it.distinct().size == it.size })
            val arms = rows.map { it.field("arm") }.toSet()
            val selected = arm ?: arms.single()
            require(selected in arms)
            provenance += mapOf("resources" to snapshots, "rows" to rows, "arm" to selected)
            rows.groupBy { listOf(it.getValue("problem").field("problem"), it.field("seed"), it.field("repeat")) }.values.map { pair ->
                require(pair.size == arms.size && pair.map { it.field("arm") }.toSet() == arms && pair.all {
                    it.field("status") == "DONE" && it.getValue("record").field("status") == "success"
                }) { "Every requested row must contain all successful paired arms" }
                val row = pair.single { it.field("arm") == selected }
                val root = directory.resolve("cases/${row.field("index")}/out").toRealPath()
                val artifacts = row.getValue("record").jsonObject.getValue("artifacts").jsonArray
                    .associateBy { it.field("path") }
                fun declared(name: String): Path {
                    val path = root.resolve(name).toRealPath()
                    require(name in artifacts && path.startsWith(root) && Files.isRegularFile(path)) { "Missing accepted artifact $name" }
                    val expected = artifacts.getValue(name).field("sha256")
                    require(expected.isEmpty() || remember(path) == expected) { "Accepted artifact hash differs: $name" }
                    return path
                }
                artifacts.keys.filter { it.matches(Regex("(?:.*/)?pass-[0-9]+/metadata.txt")) }.map { name ->
                    val path = declared(name)
                    for (target in targets) declared(name.removeSuffix("metadata.txt") + "$target.csv")
                    declared(root.relativize(path.parent.parent.resolve("source-hashes.txt")).toString())
                    path.parent
                }
            }
        }
        for (passes in groups) {
            require(passes.size >= 3 && passes.distinct().size == passes.size) { "At least three distinct paired rounds required" }
            val attemptTargets = mutableSetOf<List<String>>()
            val commits = mutableSetOf<String>()
            val models = mutableSetOf<String>()
            var expectedCases: Set<String>? = null
            val paired = mutableListOf<PairedTiming>()
            for (pass in passes.sorted()) {
                val metadataPath = pass.resolve("metadata.txt"); remember(metadataPath)
                val metadata = sections(Files.readString(metadataPath))
                val fields = metadata.getValue("capture")
                require(metadata.getValue("")["status"] == "complete" && fields.getValue("requested_samples").toInt() >= 3 &&
                    fields.getValue("target_ns").toLong() >= 10000000) { "Incomplete capture or smoke timings" }
                require(fields["dirty"] == "false" && fields.getValue("source_commit").matches(Regex("[0-9a-f]{40}")))
                commits += fields.getValue("source_commit")
                val identities = targets.map { metadata.getValue(it) }
                val vendorNames = setOf("onemkl", "openblas", "aocl", "armpl", "accelerate")
                val declared = identities.map { identity -> Regex("""/caps\[([^:]+):""")
                    .findAll(identity.getValue("implementation")).map { it.groupValues[1] }.toSet() }
                if (baseline == "jvm-owned") require(declared.dropLast(1).all { it.isNotEmpty() && it.all { source -> source in setOf("portable", "jvm-simd") } })
                identities.forEachIndexed { index, identity ->
                    val implementation = identity.getValue("implementation")
                    require(declared[index].none { it in vendorNames } || "/threads=" in implementation) { "Missing vendor version or thread evidence" }
                }
                val packagePath = pass.parent.parent.resolve("vendor-package.txt")
                if (Files.isRegularFile(packagePath)) remember(packagePath)
                if (Files.exists(index) && identities.any { "/oneMKL/" in it.getValue("implementation") })
                    require(Files.isRegularFile(packagePath)) { "Missing pinned oneMKL package evidence" }
                if (profileName != null) require(identities.last().getValue("implementation").startsWith("$candidate/$profileName/")) { "Different compiled profile" }
                val target = identities.first().let { id ->
                    listOf("runtime", "os", "arch", "cpu_model", "cpu_class", "sve_vector_bits").map { id.getValue("profile_$it") }
                }
                require(identities.all { id -> target == listOf("runtime", "os", "arch", "cpu_model", "cpu_class", "sve_vector_bits")
                    .map { id.getValue("profile_$it") } }) { "Paired targets differ" }
                attemptTargets += target
                if (cpuClass != null && target[4] != cpuClass || sveBits != null && target[5] != sveBits) continue
                val hardware = metadata["hardware"].orEmpty()
                val model = hardware["model name"] ?: hardware["cpu_model"] ?: target[3]
                require(model != "*" && target[4] != "*") { "Unknown CPU model or instruction class" }
                models += model
                if (target[4] == "Sve") require(target[5].toIntOrNull() in 128..2048 && target[5].toInt() % 128 == 0)
                if (target[0] == "jvm") require(fields.getValue("warmups").toInt() >= 20 &&
                    fields.getValue("requested_jvm_forks").toInt() >= 2 && fields.getValue("target_ns").toLong() >= 200000000)
                platforms += target
                val protocolKeys = listOf("warmups", "requested_samples", "target_ns", "requested_jvm_forks", "jvm_gc", "jvm_heap", "cpu_policy", "koblas_threads")
                require(protocolKeys.all { !fields[it].isNullOrEmpty() })
                protocols += fields.filterKeys { it in protocolKeys || it.startsWith("KOBLAS_") || it in listOf("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS") }
                val environment = listOf(fields.getValue("platform").split(' ').drop(2), identities.map { id ->
                    listOf(id.getValue("runtime"), id.getValue("harness"), id.getValue("warmup_target_ns"),
                        Regex("""/(Accelerate|oneMKL|OpenBLAS|AOCL|ArmPL)/([^\n]*?)/threads=([^/]+)""")
                            .findAll(id.getValue("implementation")).map { listOf(it.groupValues[1], it.groupValues[2].substringAfterLast('/'), it.groupValues[3]) }.toList())
                })
                require(environments.getOrPut(model) { environment } == environment) { "Different runtime or vendor settings within one CPU model" }
                val manifestPath = pass.parent.resolve("source-hashes.txt"); remember(manifestPath)
                val manifest = Files.readAllLines(manifestPath).filter { line ->
                    val path = line.substringAfter(' ')
                    path.startsWith("koblas/src/") || path.startsWith("koblas-bench/src/") || path.startsWith("koblas-bench/jmh/") || path.startsWith("gradle/") ||
                        path.endsWith(".gradle.kts") || path in listOf("gradle.properties", "koblas-bench/capture-report.sh", "koblas-bench/script-helpers.sh", "koblas-bench/lab/prepare-vendor.sh")
                }.sorted()
                require(manifest.isNotEmpty() && manifest.all { it.substringBefore(' ').matches(Regex("[0-9a-f]{64}")) })
                require(manifest.map { it.substringAfter(' ') }.distinct().size == manifest.size)
                manifests += manifest
                val tables = targets.map { targetName ->
                    val csv = pass.resolve("$targetName.csv"); remember(csv); readTimingCsv(Files.readString(csv))
                }
                require(tables.map { it.keys }.distinct().size == 1) { "Paired arms contain different cases" }
                require(expectedCases == null || expectedCases == tables.first().keys) { "Rounds contain different cases" }
                expectedCases = tables.first().keys
                cases += tables.first().keys
                for (case in tables.first().keys) {
                    val references = tables.dropLast(1).map { it.getValue(case) }
                    val right = tables.last().getValue(case)
                    if (right["status"] == "unsupported" && backend != null) continue
                    val rows = references + listOf(right)
                    require(rows.all { it["status"] == "ok" && it.getValue("samples").toInt() >= 3 &&
                        it.getValue("median_ns").toDouble().let { n -> n.isFinite() && n > 0 } }) { "Untimed case $case" }
                    require(rows.map { it.getValue("timing_mode") to it.getValue("routing_key") }.distinct().size == 1)
                    val provider = right.getValue("provider")
                    require(provider.isNotBlank())
                    if (backend != null && provider != backend) continue
                    val key = right.getValue("routing_key").split(':')
                    val shape = if (key.first().isEmpty()) null else {
                        require(key.size == 7)
                        RoutingShape(EngineImplementation.valueOf(key[0]), key[1].toInt(), key[2].toInt(), key[3].toInt(),
                            key[4].toBooleanStrict(), key[5].toInt(), key[6].toBooleanStrict())
                    }
                    paired += PairedTiming(case, model, provider, shape,
                        right.getValue("median_ns").toDouble() / references.minOf { it.getValue("median_ns").toDouble() })
                }
            }
            require(commits.size == 1 && attemptTargets.size == 1 && models.size <= 1) { "An attempt changed CPU or source revision" }
            require(paired.groupBy { it.case }.values.all { it.size == passes.size && it.map { p -> p.shape to p.provider }.distinct().size == 1 })
            timings += paired
        }
    }
    require(timings.isNotEmpty() && protocols.size == 1 && manifests.size == 1) { "Inputs have no timings or mix protocols or measured code" }
    require(platforms.map { listOf(it[0], it[1], it[2], it[4], it[5]) }.distinct().size == 1) { "Fit each target separately" }
    provenance += mapOf("target_models" to platforms, "measured_sources" to manifests.single(), "protocol" to protocols.single())
    return CalibrationCapture(platforms.first(), cases, timings, hashes, provenance)
}

private fun sections(text: String): Map<String, Map<String, String>> {
    val result = linkedMapOf("" to linkedMapOf<String, String>())
    var current = ""
    for (line in text.lineSequence()) {
        if (line.startsWith('[') && line.endsWith(']')) { current = line.drop(1).dropLast(1); result.getOrPut(current) { linkedMapOf() } }
        else if ('=' in line) result.getValue(current)[line.substringBefore('=')] = line.substringAfter('=')
    }
    return result
}

internal fun readTimingCsv(text: String): Map<String, Map<String, String>> {
    fun fields(line: String): List<String> {
        val result = mutableListOf<String>(); val field = StringBuilder(); var quoted = false; var index = 0
        while (index < line.length) {
            val ch = line[index++]
            when {
                ch == '"' && quoted && index < line.length && line[index] == '"' -> { field.append('"'); index++ }
                ch == '"' -> quoted = !quoted
                ch == ',' && !quoted -> { result += field.toString(); field.clear() }
                else -> field.append(ch)
            }
        }
        require(!quoted); return result + field.toString()
    }
    val lines = text.lineSequence().filter(String::isNotBlank).map(::fields).toList()
    val header = lines.first(); require(header.distinct().size == header.size)
    val rows = lines.drop(1).map { require(it.size == header.size); header.zip(it).toMap() }
    require(rows.map { it.getValue("case") }.distinct().size == rows.size)
    return rows.associateBy { it.getValue("case") }
}
