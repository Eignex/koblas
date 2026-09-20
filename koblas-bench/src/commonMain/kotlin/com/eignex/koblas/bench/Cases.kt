package com.eignex.koblas.bench

internal data class BenchCase(
    val operation: String,
    val dimensions: List<Int>,
    val fixture: String,
    val options: Map<String, String>,
    val id: String,
    val suites: Set<String> = setOf("default"),
) {
    fun option(name: String, default: String): String = options[name] ?: default
    fun flag(name: String, default: String = "N"): Boolean = option(name, default) == "T"
    fun dimension(index: Int): Int = dimensions[index]
}

internal object Cases {
    private val dimensionCounts = mapOf(
        "dot" to 1, "axpy" to 1, "scal" to 1, "nrm2" to 1, "asum" to 1, "sum" to 1,
        "iamax" to 1, "swap" to 1, "rot" to 1,
        "panel-multidot" to 2, "panel-columnupdate" to 2, "panel-coupled" to 2, "panel-rankupdate" to 2,
        "gemv" to 2, "symv" to 1, "ger" to 2, "syr" to 1, "syr2" to 1, "trsv" to 1,
        "trmv" to 1, "gemm" to 3, "symm" to 2, "gemmt" to 2, "syrk" to 2, "syr2k" to 2,
        "product-block" to 3, "gemm-pack" to 3, "gemm-packed" to 3, "gemm-packed-left" to 3,
        "gemm-packed-right" to 3, "gemm-generic" to 3,
        "trsm" to 2, "trmm" to 2,
        "spdot" to 1, "spdot-raw" to 1, "spdot-sparse" to 1, "spaxpy" to 1, "spaxpy-raw" to 1,
        "spnrm2" to 1, "spnrm2-indexed" to 1, "spasum" to 1,
        "spscatter" to 1, "spscatter-raw" to 1, "spgather" to 1, "spgather-zero" to 1,
        "spaccumulate" to 1,
        "spgemv" to 2, "spmm" to 3, "spmm-right" to 3, "spgemm" to 3,
        "spsymv" to 1, "spsymm" to 2, "sptrsv" to 1, "sptrmv" to 1, "sptrsm" to 2, "sptrmm" to 2,
        "spsyrk-dense" to 2, "spsyrk-sparse" to 2, "spadd" to 2,
        "spmm-generic" to 3, "spmm-generic-right" to 3, "spgemm-generic" to 3,
    )
    private val fixtures = setOf("uniform", "triangular", "sparse-uniform", "sparse-triangular")
    private val optionOrder =
        listOf("density", "support", "mode", "reuse", "side", "uplo", "transA", "transB", "diag", "timing")

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

    fun select(cases: List<BenchCase>, suite: String = "default", operation: String = "all"): List<BenchCase> {
        validateSelection(suite, operation)
        val selected = cases.filter { suite in it.suites && (operation == "all" || it.operation == operation) }
        require(selected.isNotEmpty()) { "suite '$suite' and operation '$operation' selected no cases" }
        return selected
    }

    fun validateSelection(suite: String, operation: String) {
        require(suite in setOf("default", "sweep")) { "suite must be default or sweep" }
        require(suite != "sweep" || operation != "all") { "suite sweep requires a specific operation" }
    }

    private fun parseLine(line: String, lineNumber: Int): BenchCase {
        fun invalid(message: String): Nothing = throw IllegalArgumentException("line $lineNumber: $message")
        val parts = line.split('+')
        if (parts.size < 3 || parts.any { it.isEmpty() }) invalid("expected operation+dimensions+fixture")
        val operation = parts[0]
        val count = dimensionCounts[operation] ?: invalid("unknown operation '$operation'")
        val dimensions = parts[1].split('x').map {
            it.toIntOrNull()?.takeIf { value -> value in 1..MAX_DIMENSION } ?: invalid("invalid positive dimension '$it'")
        }
        if (dimensions.size != count) invalid("$operation requires $count dimensions")
        val fixture = parts[2]
        if (fixture !in fixtures) invalid("unknown fixture '$fixture'")
        val options = linkedMapOf<String, String>()
        var suites: Set<String>? = null
        for (field in parts.drop(3)) {
            val pair = field.split('=', limit = 2)
            if (pair.size == 2 && pair[0] == "suite") {
                if (suites != null) invalid("duplicate suite")
                val names = pair[1].split(',')
                if (names.any { it !in setOf("default", "sweep") } || names.distinct().size != names.size) {
                    invalid("suite must list default and/or sweep once")
                }
                suites = names.toSet()
                continue
            }
            if (pair.size != 2 || pair[0] !in optionOrder || pair[1].isEmpty()) invalid("unknown option '$field'")
            if (pair[0] in options) invalid("duplicate option '${pair[0]}'")
            validateOption(pair[0], pair[1], invalid = ::invalid)
            options[pair[0]] = pair[1]
        }
        validateCompatibility(operation, dimensions, fixture, options, invalid = ::invalid)
        val ordered = optionOrder.filter { it in options }.associateWith { options.getValue(it) }
        val id = (listOf(operation, dimensions.joinToString("x"), fixture) + ordered.map { "${it.key}=${it.value}" }).joinToString("+")
        return BenchCase(operation, dimensions, fixture, ordered, id, suites ?: setOf("default"))
    }

    private fun validateOption(name: String, value: String, invalid: (String) -> Nothing) {
        when (name) {
            "density" -> if (value.toDoubleOrNull()?.let { it > 0.0 && it <= 1.0 } != true) invalid("invalid density '$value'")
            "support" -> if (value !in SUPPORTS) invalid("invalid support '$value'")
            "reuse" -> if (value.toIntOrNull()?.let { it in 1..MAX_REUSE } != true) invalid("invalid reuse '$value'")
            "mode" -> if (value !in MODES) invalid("invalid mode '$value'")
            "side" -> if (value !in setOf("L", "R")) invalid("invalid side '$value'")
            "uplo" -> if (value !in setOf("L", "U")) invalid("invalid uplo '$value'")
            "transA", "transB" -> if (value !in setOf("N", "T")) invalid("invalid transpose '$value'")
            "diag" -> if (value !in setOf("N", "U")) invalid("invalid diag '$value'")
        }
    }

    private fun validateCompatibility(
        operation: String,
        dimensions: List<Int>,
        fixture: String,
        options: Map<String, String>,
        invalid: (String) -> Nothing,
    ) {
        val sparse = operation.startsWith("sp")
        val triangular = operation in TRIANGULAR_FIXTURE_OPERATIONS
        val expectedFixture = when {
            sparse && triangular -> "sparse-triangular"
            sparse -> "sparse-uniform"
            triangular -> "triangular"
            else -> "uniform"
        }
        if (fixture != expectedFixture) invalid("$operation requires fixture '$expectedFixture'")
        val allowed = allowedOptions(operation, sparse)
        val incompatible = options.keys.firstOrNull { it !in allowed }
        if (incompatible != null) invalid("option '$incompatible' is incompatible with $operation")
        if (operation in MODE_OPERATIONS && "mode" !in options) invalid("$operation requires mode")
        // A reuse count is what the amortized mode divides by, and it means nothing to any other mode.
        if ((options["mode"] == "amortized") != ("reuse" in options)) {
            invalid("mode=amortized and reuse go together")
        }
        if (operation !in PREPARABLE_OPERATIONS && options["mode"] != null && options["mode"] != "oneshot") {
            invalid("$operation supports only mode=oneshot")
        }
        val required = requiredOptions(operation, sparse)
        val missing = required.firstOrNull { it !in options }
        if (missing != null) invalid("$operation requires option '$missing'")
        val defaults = mapOf("side" to "L", "uplo" to "L", "transA" to "N", "transB" to "N", "diag" to "N")
        val redundant = options.entries.firstOrNull { (name, value) -> name !in required && defaults[name] == value }
        if (redundant != null) invalid("redundant default option '${redundant.key}=${redundant.value}'")
        if (operation in setOf("scal", "spgather") && "timing" in options && options["timing"] != "arithmetic") {
            invalid("$operation supports only timing=arithmetic as an explicit override")
        }
    }

    private fun allowedOptions(operation: String, sparse: Boolean): Set<String> = buildSet {
        if (sparse) add("density")
        // Only where a matrix operand is drawn: a sparse vector case has one column and no distribution.
        if (operation in SUPPORT_OPERATIONS) add("support")
        addAll(requiredOptions(operation, sparse))
        when (operation) {
            "gemv", "gemm" -> add("transA")
        }
        // The sparse operand's own orientation, which is what a prepared transposed product derives once.
        if (operation in TRANSPOSABLE_SPARSE_OPERATIONS) add("transA")
        if (operation == "gemm") add("transB")
        // Dense storage changes the axis available to a right-hand side panel. The benchmark currently
        // exposes this flag for left-sparse products; the production right-sparse API also supports it.
        if (operation in setOf("spmm", "spmm-generic")) add("transB")
        if (operation in setOf("scal", "spgather")) add("timing")
        if (operation in MODE_OPERATIONS) add("mode")
        if (operation in PREPARABLE_OPERATIONS) add("reuse")
    }

    private fun requiredOptions(operation: String, sparse: Boolean): Set<String> {
        val required = linkedSetOf<String>()
        if (sparse) required += "density"
        when (operation) {
            "symv", "syr", "syr2" -> required += "uplo"
            "symm", "spsymm" -> required += setOf("side", "uplo")
            "gemmt" -> required += setOf("uplo", "transA", "transB")
            "syrk", "syr2k" -> required += setOf("uplo", "transA")
            "trsv", "trmv", "sptrsv", "sptrmv" -> required += setOf("uplo", "transA", "diag")
            "trsm", "trmm", "sptrsm", "sptrmm" -> required += setOf("side", "uplo", "transA", "diag")
            "spsymv", "spsyrk-dense", "spsyrk-sparse" -> required += "uplo"
        }
        return required
    }

    /**
     * How a prepared operand is accounted for.
     *
     * `oneshot` times the whole call with no snapshot. `prepared` times steady-state reuse of one built
     * outside the timed region. `setup` times building it and nothing else, and `firstuse` times building it
     * and the first call against it, which is where a derived orientation is paid for. `amortized` times
     * building one and using it `reuse` times as one batch, which is what a break-even is read from: the
     * batch against that many one-shot calls. The five are different logical work and are never compared
     * with each other, and an amortized row is a batch rather than a per-use cost.
     */
    private val MODES = setOf("oneshot", "prepared", "setup", "firstuse", "amortized")
    private val MODE_OPERATIONS = setOf(
        "spgemv", "spmm", "spmm-right", "spgemm", "spsymv", "spsymm", "sptrsv", "sptrmv", "sptrsm", "sptrmm",
        "spmm-generic", "spmm-generic-right", "spgemm-generic",
    )
    private val PREPARABLE_OPERATIONS = setOf("spgemv", "spmm", "spgemm")
    // The allocating generic product has no transpose flag to carry, so spgemm-generic is not here: a case
    // may only declare an option the call it makes actually applies.
    private val TRANSPOSABLE_SPARSE_OPERATIONS = setOf(
        "spgemv", "spmm", "spmm-right", "spgemm", "spmm-generic", "spmm-generic-right",
    )
    private val TRIANGULAR_FIXTURE_OPERATIONS = setOf(
        "trsv", "trmv", "trsm", "trmm", "spsymv", "spsymm", "sptrsv", "sptrmv", "sptrsm", "sptrmm",
    )
    /** The support distributions a sparse fixture can be drawn from, which cost differently at one density. */
    private val SUPPORTS = setOf("uniform", "banded", "skewed", "empty", "mixed")

    /** The operations whose operand is a sparse matrix, which is where a distribution means anything. */
    private val SUPPORT_OPERATIONS = setOf(
        "spgemv", "spmm", "spmm-right", "spgemm", "spsymv", "spsymm", "sptrsv", "sptrmv", "sptrsm", "sptrmm",
        "spsyrk-dense", "spsyrk-sparse", "spadd", "spmm-generic", "spmm-generic-right", "spgemm-generic",
    )
    private const val MAX_DIMENSION = 1_000_000

    /** Uses of one snapshot an amortized row batches, bounded so a case cannot ask for an unbounded loop. */
    private const val MAX_REUSE = 4_096
}
