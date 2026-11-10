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
    private val fixtures = operationSpecs.values.mapTo(mutableSetOf()) { it.fixture }
    private val optionOrder =
        listOf("density", "support", "mode", "side", "uplo", "transA", "transB", "diag", "timing", "workspace", "layout", "stride")

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
        val spec = operationSpecs[operation] ?: invalid("unknown operation '$operation'")
        val dimensions = parts[1].split('x').map {
            it.toIntOrNull()?.takeIf { value -> value > 0 } ?: invalid("invalid positive dimension '$it'")
        }
        if (dimensions.size != spec.dimensions) invalid("$operation requires ${spec.dimensions} dimensions")
        if (dimensions[0] < spec.minimumOrder) {
            invalid("$operation requires an order of at least two")
        }
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
        validateCompatibility(operation, spec, fixture, options, invalid = ::invalid)
        val ordered = optionOrder.filter { it in options }.associateWith { options.getValue(it) }
        val id = (listOf(operation, dimensions.joinToString("x"), fixture) + ordered.map { "${it.key}=${it.value}" }).joinToString("+")
        return BenchCase(operation, dimensions, fixture, ordered, id, suites ?: setOf("default"))
    }

    private fun validateOption(name: String, value: String, invalid: (String) -> Nothing) {
        when (name) {
            "density" -> if (value.toDoubleOrNull()?.let { it > 0.0 && it <= 1.0 } != true) invalid("invalid density '$value'")
            "support" -> if (value !in SUPPORTS) invalid("invalid support '$value'")
            "mode" -> if (value != "oneshot") invalid("invalid mode '$value'")
            "side" -> if (value !in setOf("L", "R")) invalid("invalid side '$value'")
            "uplo" -> if (value !in setOf("L", "U")) invalid("invalid uplo '$value'")
            "transA", "transB" -> if (value !in setOf("N", "T")) invalid("invalid transpose '$value'")
            "diag" -> if (value !in setOf("N", "U")) invalid("invalid diag '$value'")
            "workspace" -> if (value != "retained") invalid("workspace must be retained")
            "layout" -> if (value != "borrowed") invalid("layout must be borrowed")
            "stride" -> if (value.toIntOrNull()?.let { it != 0 && it != Int.MIN_VALUE } != true) invalid("invalid stride '$value'")
        }
    }

    private fun validateCompatibility(
        operation: String,
        spec: OperationSpec,
        fixture: String,
        options: Map<String, String>,
        invalid: (String) -> Nothing,
    ) {
        if (fixture != spec.fixture) invalid("$operation requires fixture '${spec.fixture}'")
        val incompatible = options.keys.firstOrNull { it !in spec.allowed }
        if (incompatible != null) invalid("option '$incompatible' is incompatible with $operation")
        val missing = spec.required.firstOrNull { it !in options }
        if (missing != null) invalid("$operation requires option '$missing'")
        val defaults = mapOf("side" to "L", "uplo" to "L", "transA" to "N", "transB" to "N", "diag" to "N")
        val redundant = options.entries.firstOrNull { (name, value) -> name !in spec.required && defaults[name] == value }
        if (redundant != null) invalid("redundant default option '${redundant.key}=${redundant.value}'")
        if ("timing" in options && options["timing"] != "arithmetic") {
            invalid("$operation supports only timing=arithmetic as an explicit override")
        }
    }

    /** Sparse fixture distributions have different costs even at one density. */
    private val SUPPORTS = setOf("uniform", "banded", "skewed", "empty", "mixed")
}
