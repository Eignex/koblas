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

/** Numerical and execution policy shared by host and bundled UMFPACK providers. */
public data class UmfpackOptions(
    /** Whether the factorization scales rows. */
    val equilibrate: Boolean = false,
    /** Maximum residual-correction passes per solve. */
    val iterativeRefinementSteps: Int = 0,
    /** Threshold-pivoting tolerance. */
    val pivotTolerance: Double = 0.1,
    /** Native row-scaling strategy. */
    val scaling: UmfpackScaling = UmfpackScaling.SUM,
) {
    init {
        require(iterativeRefinementSteps >= 0) { "iterativeRefinementSteps must not be negative" }
        require(pivotTolerance in 0.0..1.0) { "pivotTolerance must be between zero and one" }
    }
}

/** Policy for one UMFPACK backend instance. */
public data class UmfpackConfig(
    /** An absolute UMFPACK library path, or the deployment lookup chain when null. */
    val libraryPath: String? = null,
    /** Whether to scale rows before factorizing and undo it in the solves. */
    val equilibrate: Boolean = false,
    /** Maximum residual-correction passes each solve may perform. */
    val iterativeRefinementSteps: Int = 0,
    /** UMFPACK's threshold-pivoting tolerance. */
    val pivotTolerance: Double = 0.1,
    /** Row scaling used while constructing numeric factors requested with `equilibrate = true`. */
    val scaling: UmfpackScaling = UmfpackScaling.SUM,
) {
    /** Creates a deployment-discovered UMFPACK configuration from shared [options]. */
    public constructor(options: UmfpackOptions) : this(null, options)

    /** Creates a UMFPACK configuration from a library location and shared [options]. */
    public constructor(libraryPath: String?, options: UmfpackOptions) : this(
        libraryPath = libraryPath,
        equilibrate = options.equilibrate,
        iterativeRefinementSteps = options.iterativeRefinementSteps,
        pivotTolerance = options.pivotTolerance,
        scaling = options.scaling,
    )

    /** Numerical and execution policy, independent of [libraryPath]. */
    public val options: UmfpackOptions
        get() = UmfpackOptions(
            equilibrate,
            iterativeRefinementSteps,
            pivotTolerance,
            scaling,
        )

    init {
        require(iterativeRefinementSteps >= 0) { "iterativeRefinementSteps must not be negative" }
        require(pivotTolerance in 0.0..1.0) { "pivotTolerance must be between zero and one" }
    }
}

internal fun UmfpackOptions.metadataOptions(): Map<String, String> = mapOf(
    "equilibrate" to equilibrate.toString(),
    "iterativeRefinementSteps" to iterativeRefinementSteps.toString(),
    "pivotTolerance" to pivotTolerance.toString(),
    "scaling" to scaling.name,
)
