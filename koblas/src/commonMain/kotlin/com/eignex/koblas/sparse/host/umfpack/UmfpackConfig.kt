package com.eignex.koblas.sparse.host.umfpack

/** The row-scaling strategy UMFPACK applies while building numeric factors. */
public enum class UmfpackScaling {
    /** Do not scale rows. */
    NONE,

    /** Divide each row by its absolute-value sum. */
    SUM,

    /** Divide each row by its largest absolute value. */
    MAX,
}

/** Policy for one UMFPACK backend instance. */
public data class UmfpackConfig(
    /** An absolute UMFPACK library path, or the deployment lookup chain when null. */
    val libraryPath: String? = null,
    /** Smallest stored-entry count routed to the native factorization; null keeps the platform default. */
    val factorizeMin: Int? = null,
    /** Maximum residual-correction passes each solve may perform. */
    val iterativeRefinementSteps: Int = 0,
    /** UMFPACK's threshold-pivoting tolerance. */
    val pivotTolerance: Double = 0.1,
    /** Row scaling used while constructing numeric factors requested with `equilibrate = true`. */
    val scaling: UmfpackScaling = UmfpackScaling.SUM,
) {
    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
        require(iterativeRefinementSteps >= 0) { "iterativeRefinementSteps must not be negative" }
        require(pivotTolerance in 0.0..1.0) { "pivotTolerance must be between zero and one" }
    }
}
