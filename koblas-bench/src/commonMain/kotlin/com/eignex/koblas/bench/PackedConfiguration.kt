package com.eignex.koblas.bench

/** The recipe fixes depth-major interleaving, positive-zero padding, natural double alignment and one tile per call. */
internal class PackedConfiguration(val case: BenchCase) {
    val rows = when (case.options.getValue("packed")) {
        "4x4" -> 4
        "8x4" -> 8
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

internal expect fun benchmarkPackedKernels(engine: com.eignex.koblas.KoblasEngine): com.eignex.koblas.dense.PackedKernels

internal fun actualPackedKernel(case: BenchCase, mode: String, status: String): String {
    if (status != "ok") return "unavailable"
    if ("packed" !in case.options) return "policy"
    val operation = case.operation
    val component = when {
        operation.startsWith("pack-") || operation.startsWith("write-") || operation.startsWith("clear-") -> "portable-layout"
        operation == "packed-trsm" && mode.startsWith("jvm") -> "portable-solve"
        mode == "jvm-scalar" -> "portable-tile"
        mode == "jvm-simd" && operation == "gemm-trsm" -> "vector-update-portable-solve"
        mode == "jvm-simd" -> "vector-tile"
        else -> "c-tile"
    }
    return component
}
