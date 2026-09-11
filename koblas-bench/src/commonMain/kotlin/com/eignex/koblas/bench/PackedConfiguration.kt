package com.eignex.koblas.bench

internal val PACKED_OPTIONS = listOf("work", "leftLayout", "rightLayout", "leftGroup", "rightGroup", "leftStride", "rightStride", "padding", "alignment", "block", "panel", "diagonal", "rhs", "batch", "variant", "timing")

/** Benchmark-only description of the existing interleaved panel formats; no engine defaults supply work. */
internal class PackedConfiguration(val case: BenchCase) {
    val rows = case.options.getValue("leftGroup").toInt()
    val columns = case.options.getValue("rightGroup").toInt()
    val depth = case.options.getValue("panel").toInt()
    val timing = case.options.getValue("timing")
    val m = case.dimension(0)
    val n = case.dimension(1)
    val rowTiles = (m + rows - 1) / rows
    val columnTiles = (n + columns - 1) / columns
    val leftSize = rowTiles * rows * depth
    val rightSize = columnTiles * columns * depth

    companion object {
        fun validate(operation: String, dimensions: List<Int>, options: Map<String, String>) {
            fun exact(key: String, expected: String) {
                require(options[key] == expected) { "$operation requires $key=$expected" }
            }
            val (r, c) = options.getValue("physical").split('x').map(String::toInt)
            val m = dimensions[0]; val n = dimensions[1]
            val layout = operation.startsWith("pack-") || operation.startsWith("write-") || operation.startsWith("clear-")
            val solve = operation == "packed-trsm" || operation == "gemm-trsm"
            val k = if (dimensions.size == 3) dimensions[2] else if (layout && "right" in operation) m else n
            val work = when (operation) {
                "gemm-tile", "gemm-block" -> "gemm-add-v1"
                "packed-trsm" -> "right-solve-v1"
                "gemm-trsm" -> "update-solve-v1"
                else -> "$operation-v1"
            }
            exact("work", work)
            exact("leftLayout", "depth-rows-v1"); exact("rightLayout", "depth-columns-v1")
            exact("leftGroup", "$r"); exact("rightGroup", "$c")
            exact("leftStride", "$r"); exact("rightStride", "$c")
            exact("padding", "zero"); exact("alignment", "8")
            exact("block", "${m}x${n}x$k"); exact("panel", "$k")
            exact("diagonal", if (solve) "$n" else "0"); exact("rhs", if (solve) "$m" else "0")
            exact("batch", "1"); exact("variant", "current-tile-v1")
            val timings = when {
                operation == "gemm-block" -> setOf("prepacked-compute", "pack-plus-compute")
                operation.startsWith("pack-") -> setOf("packing-only")
                layout -> setOf("layout-only")
                else -> setOf("raw-tile")
            }
            require(options["timing"] in timings) { "unsupported timing for $operation" }
            require(dimensions.all { it <= 4096 } && m.toLong() * n * k <= 16_777_216L) {
                "packed benchmark exceeds bounded allocation budget"
            }
            if (operation.contains("symmetric") || operation.contains("triangular")) {
                require(m == n) { "structured layout cases require a square source" }
            }
        }
    }
}

internal val BenchCase.logicalId: String get() = buildString {
    append(options["work"] ?: "$operation-v1")
    append('+'); append(dimensions.joinToString("x")); append('+'); append(fixture)
    for ((key, value) in options) if (key !in PACKED_OPTIONS && key != "physical") {
        append('+'); append(key); append('='); append(value)
    }
}

internal val BenchCase.configurationId: String get() = if ("physical" !in options) "policy-v1" else
    options.filterKeys { it == "physical" || it in PACKED_OPTIONS && it != "work" && it != "timing" }
        .entries.joinToString("+") { "${it.key}=${it.value}" }

/** Physical arithmetic visits and buffer extents, in doubles; see coverage.md for byte and lifetime formulas. */
internal val BenchCase.physicalWork: String get() {
    if ("physical" !in options) return "policy"
    val p = PackedConfiguration(this)
    val layout = operation.startsWith("pack-") || operation.startsWith("write-") || operation.startsWith("clear-")
    if (layout) {
        val size = if ("left" in operation) ((p.m + p.rows - 1) / p.rows) * p.rows * p.n
            else ((p.n + p.columns - 1) / p.columns) * p.columns * p.m
        return "panelDoubles=$size"
    }
    val tiles = if (operation == "gemm-block") p.rowTiles * p.columnTiles else 1
    val left = if (operation == "packed-trsm") 0 else p.leftSize
    val right = if (operation == "packed-trsm") 0 else p.rightSize
    val triangle = if (operation in setOf("packed-trsm", "gemm-trsm")) p.columns * p.columns else 0
    val output = if (operation == "gemm-block") p.m * p.n else p.rows * p.columns
    val scratch = if (operation == "gemm-block") p.rows * p.columns else 0
    return "tiles=$tiles;leftDoubles=$left;rightDoubles=$right;outputDoubles=$output;resetDoubles=$output;scratchDoubles=$scratch;triangleDoubles=$triangle"
}

internal expect fun benchmarkPackedKernels(engine: com.eignex.koblas.KoblasEngine): com.eignex.koblas.dense.PackedKernels

internal fun actualPackedKernel(case: BenchCase, mode: String, status: String): String {
    if (status != "ok") return "unavailable"
    if ("physical" !in case.options) return "$mode/policy-v1"
    val operation = case.operation
    val component = when {
        operation.startsWith("pack-") || operation.startsWith("write-") || operation.startsWith("clear-") -> "portable-layout-v1"
        operation == "packed-trsm" && mode.startsWith("jvm") -> "portable-solve-v1"
        mode == "jvm-scalar" -> "portable-tile-v1"
        mode == "jvm-simd" && operation == "gemm-trsm" -> "vector-update-portable-solve-v1"
        mode == "jvm-simd" -> "vector-tile-v1"
        else -> "c-tile-v1"
    }
    return "$mode/$component/${case.options.getValue("physical")}"
}
