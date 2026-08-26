package com.eignex.koblas.sparse.host.klu

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
) {
    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
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
