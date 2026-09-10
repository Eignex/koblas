package com.eignex.koblas.bench

internal const val WORKLOAD_VERSION = "2"
internal const val FIXTURE_VERSION = "1"

internal data class BenchCase(
    val operation: String,
    val dimensions: List<Int>,
    val fixture: String,
    val options: Map<String, String>,
    val id: String,
) {
    fun option(name: String, default: String): String = options[name] ?: default
    fun flag(name: String, default: String = "N"): Boolean = option(name, default) == "T"
    fun dimension(index: Int): Int = dimensions[index]
}

internal object Cases {
    private val dimensionCounts = mapOf(
        "dot" to 1, "axpy" to 1, "scal" to 1, "nrm2" to 1, "asum" to 1, "sum" to 1,
        "compensated-sum" to 1, "iamax" to 1, "swap" to 1, "rot" to 1, "rotm" to 1, "rotmg" to 1,
        "ssqd" to 1, "dot4" to 1, "axpy4" to 1, "dot-axpy" to 1,
        "gemv" to 2, "symv" to 1, "ger" to 2, "syr" to 1, "syr2" to 1, "trsv" to 1,
        "trmv" to 1, "gemm" to 3, "symm" to 2, "gemmt" to 2, "syrk" to 2, "syr2k" to 2,
        "trsm" to 2, "trmm" to 2, "gemm-tile" to 3, "packed-trsm" to 2, "gemm-trsm" to 3,
        "pack-left" to 2, "pack-right" to 2, "pack-symmetric-left" to 2,
        "pack-symmetric-right" to 2, "pack-triangular-left" to 2, "pack-triangular-right" to 2,
        "write-left" to 2, "write-right" to 2, "clear-left-padding" to 2, "clear-right-padding" to 2,
        "spdot" to 1, "spdot-sparse" to 1, "spaxpy" to 1, "spnrm2" to 1, "spasum" to 1,
        "spscatter" to 1, "spgather" to 1, "spgather-zero" to 1, "spgemv" to 2, "spmm" to 3, "spgemm" to 3,
        "spsymv" to 1, "spsymm" to 2, "sptrsv" to 1, "sptrmv" to 1, "sptrsm" to 2, "sptrmm" to 2,
        "spsyrk-dense" to 2, "spsyrk-sparse" to 2, "spadd" to 2,
        "workspace-scatter" to 1, "workspace-scatter-checked" to 1, "workspace-gather" to 1,
        "workspace-gather-clear" to 1, "workspace-max" to 1, "workspace-filter" to 1,
    )
    private val fixtures = setOf("uniform", "triangular", "sparse-uniform", "sparse-triangular")
    private val optionOrder = listOf("density", "mode", "physical", "side", "uplo", "transA", "transB", "diag")
    private val commonOptions = setOf("density", "mode", "physical", "side", "uplo", "transA", "transB", "diag")

    fun parse(text: String): List<BenchCase> {
        val cases = text.lineSequence().mapIndexedNotNull { index, raw ->
            val line = raw.trim()
            if (line.isEmpty() || line.startsWith('#')) null else parseLine(line, index + 1)
        }.toList()
        require(cases.isNotEmpty()) { "case file contains no cases" }
        val duplicate = cases.groupBy { it.id }.entries.firstOrNull { it.value.size > 1 }
        require(duplicate == null) { "duplicate case: ${duplicate?.key}" }
        return cases
    }

    private fun parseLine(line: String, lineNumber: Int): BenchCase {
        fun invalid(message: String): Nothing = throw IllegalArgumentException("line $lineNumber: $message")
        val parts = line.split('+')
        if (parts.size < 3 || parts.any { it.isEmpty() }) invalid("expected operation+dimensions+fixture")
        val operation = parts[0]
        val count = dimensionCounts[operation] ?: invalid("unknown operation '$operation'")
        val dimensions = parts[1].split('x').map {
            it.toIntOrNull()?.takeIf { value -> value > 0 } ?: invalid("invalid positive dimension '$it'")
        }
        if (dimensions.size != count) invalid("$operation requires $count dimensions")
        val fixture = parts[2]
        if (fixture !in fixtures) invalid("unknown fixture '$fixture'")
        val options = linkedMapOf<String, String>()
        var previous = -1
        for (field in parts.drop(3)) {
            val pair = field.split('=', limit = 2)
            if (pair.size != 2 || pair[0] !in commonOptions || pair[1].isEmpty()) invalid("unknown option '$field'")
            if (pair[0] in options) invalid("duplicate option '${pair[0]}'")
            val order = optionOrder.indexOf(pair[0])
            if (order <= previous) invalid("options are not in canonical order")
            previous = order
            validateOption(pair[0], pair[1], invalid = ::invalid)
            options[pair[0]] = pair[1]
        }
        validateCompatibility(operation, fixture, options, invalid = ::invalid)
        return BenchCase(operation, dimensions, fixture, options, line)
    }

    private fun validateOption(name: String, value: String, invalid: (String) -> Nothing) {
        when (name) {
            "density" -> if (value.toDoubleOrNull()?.let { it > 0.0 && it <= 1.0 } != true) invalid("invalid density '$value'")
            "mode" -> if (value !in setOf("prepared", "oneshot")) invalid("invalid mode '$value'")
            "physical" -> if (value !in setOf("4x4", "8x4")) invalid("unsupported physical tile '$value'")
            "side" -> if (value !in setOf("L", "R")) invalid("invalid side '$value'")
            "uplo" -> if (value !in setOf("L", "U")) invalid("invalid uplo '$value'")
            "transA", "transB" -> if (value !in setOf("N", "T")) invalid("invalid transpose '$value'")
            "diag" -> if (value !in setOf("N", "U")) invalid("invalid diag '$value'")
        }
    }

    private fun validateCompatibility(
        operation: String,
        fixture: String,
        options: Map<String, String>,
        invalid: (String) -> Nothing,
    ) {
        val sparse = operation.startsWith("sp") || operation.startsWith("workspace-")
        if (sparse != fixture.startsWith("sparse-")) invalid("fixture '$fixture' is incompatible with $operation")
        if ("density" in options && !sparse) invalid("density applies only to sparse cases")
        if ("mode" in options && operation !in setOf("spgemv", "spmm", "spgemm", "spsymv", "spsymm", "sptrsv", "sptrmv", "sptrsm", "sptrmm")) invalid("mode is incompatible with $operation")
        if (operation in setOf("spgemv", "spmm", "spgemm", "spsymv", "spsymm", "sptrsv", "sptrmv", "sptrsm", "sptrmm") && "mode" !in options) invalid("$operation requires mode")
        if ("physical" in options && operation !in setOf("gemm-tile", "packed-trsm", "gemm-trsm", "pack-left", "pack-right", "pack-symmetric-left", "pack-symmetric-right", "pack-triangular-left", "pack-triangular-right", "write-left", "write-right", "clear-left-padding", "clear-right-padding")) invalid("physical is incompatible with $operation")
        val required = requiredOptions(operation, sparse)
        val missing = required.firstOrNull { it !in options }
        if (missing != null) invalid("$operation requires option '$missing'")
        val defaults = mapOf("side" to "L", "uplo" to "L", "transA" to "N", "transB" to "N", "diag" to "N")
        val redundant = options.entries.firstOrNull { (name, value) -> name !in required && defaults[name] == value }
        if (redundant != null) invalid("redundant default option '${redundant.key}=${redundant.value}'")
    }

    private fun requiredOptions(operation: String, sparse: Boolean): Set<String> {
        val required = linkedSetOf<String>()
        if (sparse) required += "density"
        if (operation in setOf("gemm-tile", "packed-trsm", "gemm-trsm", "pack-left", "pack-right", "pack-symmetric-left", "pack-symmetric-right", "pack-triangular-left", "pack-triangular-right", "write-left", "write-right", "clear-left-padding", "clear-right-padding")) required += "physical"
        when (operation) {
            "symv", "syr", "syr2" -> required += "uplo"
            "symm", "spsymm" -> required += setOf("side", "uplo")
            "gemmt" -> required += setOf("uplo", "transA", "transB")
            "syrk", "syr2k" -> required += setOf("uplo", "transA")
            "trsv", "trmv", "sptrsv", "sptrmv" -> required += setOf("uplo", "transA", "diag")
            "trsm", "trmm", "sptrsm", "sptrmm" -> required += setOf("side", "uplo", "transA", "diag")
            "packed-trsm", "gemm-trsm", "pack-triangular-left", "pack-triangular-right" -> required += setOf("uplo", "diag")
            "pack-symmetric-left", "pack-symmetric-right", "spsymv", "spsyrk-dense", "spsyrk-sparse" -> required += "uplo"
        }
        return required
    }
}
