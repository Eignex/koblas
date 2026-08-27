package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.internal.backend.hostSparseDispatchThresholds

/** KLU's built-in fill-reducing orderings, excluding callback-based orderings that koblas does not expose. */
public enum class KluOrdering {
    /** Approximate minimum degree ordering. */
    AMD,

    /** Column approximate minimum degree ordering. */
    COLAMD,
}

/** Row scaling KLU applies when [com.eignex.koblas.sparse.F64SparseDecompositions.factor] requests equilibration. */
public enum class KluScaling {
    /** Do not scale rows. */
    NONE,

    /** Divide each row by its absolute-value sum. */
    SUM,

    /** Divide each row by its largest absolute value. */
    MAX,
}

/**
 * Numerical and execution policy shared by host and bundled KLU providers.
 *
 * @property factorizeMin smallest stored-entry count routed to native factorization.
 * @property equilibrate whether to scale rows before factorization.
 * @property pivotTolerance pivot tolerance for diagonal preference.
 * @property memoryGrowth factor-storage growth multiplier.
 * @property amdInitialMemoryFactor initial storage multiplier with an AMD fill estimate.
 * @property initialMemoryFactor initial storage multiplier without an AMD fill estimate.
 * @property maxBtfWork maximum block-triangular-form work.
 * @property useBtf whether to apply block-triangular-form preordering.
 * @property ordering built-in ordering applied to each block.
 * @property equilibratedScaling scaling used when [equilibrate] is true.
 * @property haltIfSingular whether KLU stops at the first singular pivot.
 * @property qrFactorizeMin smallest stored-entry count routed to SPQR.
 */
public data class KluOptions(
    val factorizeMin: Int? = null,
    val equilibrate: Boolean = false,
    val pivotTolerance: Double? = null,
    val memoryGrowth: Double? = null,
    val amdInitialMemoryFactor: Double? = null,
    val initialMemoryFactor: Double? = null,
    val maxBtfWork: Double? = null,
    val useBtf: Boolean? = null,
    val ordering: KluOrdering? = null,
    val equilibratedScaling: KluScaling = KluScaling.MAX,
    val haltIfSingular: Boolean? = null,
    val qrFactorizeMin: Int? = null,
) {
    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
        require(qrFactorizeMin == null || qrFactorizeMin >= 0) { "qrFactorizeMin must not be negative" }
        require(pivotTolerance == null || pivotTolerance in 0.0..1.0) {
            "pivotTolerance must be between zero and one"
        }
        require(memoryGrowth == null || memoryGrowth > 0.0) { "memoryGrowth must be positive" }
        require(amdInitialMemoryFactor == null || amdInitialMemoryFactor > 0.0) {
            "amdInitialMemoryFactor must be positive"
        }
        require(initialMemoryFactor == null || initialMemoryFactor > 0.0) {
            "initialMemoryFactor must be positive"
        }
        require(maxBtfWork == null || maxBtfWork.isFinite()) { "maxBtfWork must be finite" }
    }
}

/** Policy for one KLU backend instance. Null controls retain KLU's native defaults. */
public data class KluConfig(
    /** An absolute KLU library path, or the platform lookup chain when null. */
    val libraryPath: String? = null,
    /** Smallest stored-entry count routed to the native factorization; null keeps the platform default. */
    val factorizeMin: Int? = null,
    /** Whether to scale rows before factorizing and undo it in the solves. */
    val equilibrate: Boolean = false,
    /** Pivot tolerance for diagonal preference. */
    val pivotTolerance: Double? = null,
    /** Factor-storage growth multiplier when KLU reallocates. */
    val memoryGrowth: Double? = null,
    /** Initial factor-storage multiplier when AMD estimates fill. */
    val amdInitialMemoryFactor: Double? = null,
    /** Initial factor-storage multiplier without an AMD fill estimate. */
    val initialMemoryFactor: Double? = null,
    /** Maximum BTF work, or zero/negative for no limit. */
    val maxBtfWork: Double? = null,
    /** Whether to apply block triangular form preordering. */
    val useBtf: Boolean? = null,
    /** Built-in ordering applied to each block. */
    val ordering: KluOrdering? = null,
    /** Scaling for a factorization requested with `equilibrate = true`. */
    val equilibratedScaling: KluScaling = KluScaling.MAX,
    /** Whether KLU stops immediately after finding a singular pivot. */
    val haltIfSingular: Boolean? = null,
    /** Smallest stored-entry count routed to SPQR; null keeps the platform sparse-QR default. */
    val qrFactorizeMin: Int? = null,
) {
    /** Creates a deployment-discovered KLU configuration from shared [options]. */
    public constructor(options: KluOptions) : this(null, options)

    /** Creates a KLU configuration from a library location and shared [options]. */
    public constructor(libraryPath: String?, options: KluOptions) : this(
        libraryPath = libraryPath,
        factorizeMin = options.factorizeMin,
        qrFactorizeMin = options.qrFactorizeMin,
        equilibrate = options.equilibrate,
        pivotTolerance = options.pivotTolerance,
        memoryGrowth = options.memoryGrowth,
        amdInitialMemoryFactor = options.amdInitialMemoryFactor,
        initialMemoryFactor = options.initialMemoryFactor,
        maxBtfWork = options.maxBtfWork,
        useBtf = options.useBtf,
        ordering = options.ordering,
        equilibratedScaling = options.equilibratedScaling,
        haltIfSingular = options.haltIfSingular,
    )

    /** Numerical and execution policy, independent of [libraryPath]. */
    public val options: KluOptions
        get() = KluOptions(
            factorizeMin = factorizeMin,
            qrFactorizeMin = qrFactorizeMin,
            equilibrate = equilibrate,
            pivotTolerance = pivotTolerance,
            memoryGrowth = memoryGrowth,
            amdInitialMemoryFactor = amdInitialMemoryFactor,
            initialMemoryFactor = initialMemoryFactor,
            maxBtfWork = maxBtfWork,
            useBtf = useBtf,
            ordering = ordering,
            equilibratedScaling = equilibratedScaling,
            haltIfSingular = haltIfSingular,
        )

    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
        require(qrFactorizeMin == null || qrFactorizeMin >= 0) { "qrFactorizeMin must not be negative" }
        require(pivotTolerance == null || pivotTolerance in 0.0..1.0) {
            "pivotTolerance must be between zero and one"
        }
        require(memoryGrowth == null || memoryGrowth > 0.0) { "memoryGrowth must be positive" }
        require(amdInitialMemoryFactor == null || amdInitialMemoryFactor > 0.0) {
            "amdInitialMemoryFactor must be positive"
        }
        require(initialMemoryFactor == null || initialMemoryFactor > 0.0) {
            "initialMemoryFactor must be positive"
        }
        require(maxBtfWork == null || maxBtfWork.isFinite()) { "maxBtfWork must be finite" }
    }
}

internal fun KluOptions.metadataOptions(): Map<String, String> = buildMap {
    put("factorizeMin", hostSparseDispatchThresholds(factorize = factorizeMin).factorize.toString())
    put("qrFactorizeMin", hostSparseDispatchThresholds(qr = qrFactorizeMin).qr.toString())
    put("equilibrate", equilibrate.toString())
    pivotTolerance?.let { put("pivotTolerance", it.toString()) }
    memoryGrowth?.let { put("memoryGrowth", it.toString()) }
    amdInitialMemoryFactor?.let { put("amdInitialMemoryFactor", it.toString()) }
    initialMemoryFactor?.let { put("initialMemoryFactor", it.toString()) }
    maxBtfWork?.let { put("maxBtfWork", it.toString()) }
    useBtf?.let { put("useBtf", it.toString()) }
    ordering?.let { put("ordering", it.name) }
    put("equilibratedScaling", equilibratedScaling.name)
    haltIfSingular?.let { put("haltIfSingular", it.toString()) }
}

/** Outcome shared by the JVM and Kotlin/Native same-pattern KLU paths. */
internal enum class KluRefactorResult { Success, Incompatible, Singular }
