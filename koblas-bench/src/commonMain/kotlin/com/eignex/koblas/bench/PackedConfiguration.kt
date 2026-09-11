package com.eignex.koblas.bench

/** Version one fixes depth-major interleaving, positive-zero padding, natural double alignment and one tile per call. */
internal class PackedConfiguration(val case: BenchCase) {
    val rows = when (case.options.getValue("packed")) {
        "4x4-v1" -> 4
        "8x4-v1" -> 8
        else -> error("unsupported packed recipe")
    }
    val columns = 4
    val m = case.dimension(0)
    val n = case.dimension(1)
    val depth = depth(case.operation, case.dimensions)
    val timing = when {
        case.operation == "gemm-block" -> case.options.getValue("timing")
        case.operation.startsWith("pack-") -> "packing-only"
        case.operation.startsWith("write-") || case.operation.startsWith("clear-") -> "layout-only"
        else -> "raw-tile"
    }
    val rowTiles = (m + rows - 1) / rows
    val columnTiles = (n + columns - 1) / columns
    val leftSize = rowTiles * rows * depth
    val rightSize = columnTiles * columns * depth

    companion object {
        private fun depth(operation: String, dimensions: List<Int>): Int = when {
            dimensions.size == 3 -> dimensions[2]
            "right" in operation -> dimensions[0]
            else -> dimensions[1]
        }

        fun validate(operation: String, dimensions: List<Int>, options: Map<String, String>) {
            if (operation == "gemm-block") {
                require(options["timing"] in setOf("prepacked-compute", "pack-plus-compute")) { "unsupported block timing" }
            }
            val m = dimensions[0]; val n = dimensions[1]; val k = depth(operation, dimensions)
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
    append(when (operation) {
        "gemm-tile", "gemm-block" -> "gemm-add-v1"
        "packed-trsm" -> "right-solve-v1"
        "gemm-trsm" -> "update-solve-v1"
        else -> "$operation-v1"
    })
    append('+'); append(dimensions.joinToString("x")); append('+'); append(fixture)
    for ((key, value) in options) if (key != "packed" && key != "timing") {
        append('+'); append(key); append('='); append(value)
    }
}

internal val BenchCase.configurationId: String get() = options["packed"]?.let { "packed=$it" } ?: "policy-v1"

/** Physical arithmetic visits and buffer extents, in doubles. */
internal val BenchCase.physicalWork: String get() {
    if ("packed" !in options) return "policy"
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
    if ("packed" !in case.options) return "$mode/policy-v1"
    val operation = case.operation
    val component = when {
        operation.startsWith("pack-") || operation.startsWith("write-") || operation.startsWith("clear-") -> "portable-layout-v1"
        operation == "packed-trsm" && mode.startsWith("jvm") -> "portable-solve-v1"
        mode == "jvm-scalar" -> "portable-tile-v1"
        mode == "jvm-simd" && operation == "gemm-trsm" -> "vector-update-portable-solve-v1"
        mode == "jvm-simd" -> "vector-tile-v1"
        else -> "c-tile-v1"
    }
    return "$mode/$component/${case.options.getValue("packed").removeSuffix("-v1")}"
}
