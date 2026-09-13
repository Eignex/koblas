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
    val operation = case.operation
    val raw = rawNativeVariant(mode)
    val suffix = raw?.name?.lowercase()
    if ("packed" !in case.options) {
        if (suffix == null) return "policy"
        val leaf = when (operation) {
            "scal" -> "scale"
            "axpy-arithmetic" -> "axpy_arithmetic"
            "dot-axpy" -> "dot_axpy"
            "rot" -> "rotm"
            "dot", "sum", "ssqd", "asum", "nrm2", "iamax", "axpy", "swap", "rotm", "dot4", "axpy4" -> operation
            else -> return "portable-orchestration/native-$suffix"
        }
        return "koblas_dense_${leaf}_$suffix"
    }
    if (operation.startsWith("pack-") || operation.startsWith("write-") || operation.startsWith("clear-"))
        return "portable-layout"
    if (mode == "jvm-scalar") return "portable-tile"
    if (mode == "jvm-simd") return when (operation) {
        "packed-trsm" -> "portable-solve"
        "gemm-trsm" -> "vector-update-portable-solve"
        else -> "vector-tile"
    }
    val variant = suffix ?: requireNotNull(resolveEngine(mode).first.nativeVariant).name.lowercase()
    val leaf = when (operation) {
        "packed-trsm" -> "trsm_tile"
        "gemm-trsm" -> "gemm_trsm_tile"
        else -> "gemm_tile"
    }
    return "koblas_dense_${leaf}_$variant"
}

/** Exact native modes never choose a different variant when unavailable. */
internal fun rawNativeVariant(mode: String): com.eignex.koblas.NativeVariant? {
    val prefix = when {
        mode.startsWith("jvm-c-raw-") -> "jvm-c-raw-"
        mode.startsWith("native-raw-") -> "native-raw-"
        else -> return null
    }
    return com.eignex.koblas.NativeVariant.entries.singleOrNull { it.name.lowercase() == mode.removePrefix(prefix) }
        ?: error("unknown native variant in $mode")
}
