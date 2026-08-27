package com.eignex.koblas.sparse.host.spqr

/**
 * Numerical and execution policy for the SPQR sparse QR.
 *
 * @property ordering the fill-reducing column ordering SPQR applies before factorizing.
 * @property rankTolerance columns with a 2-norm at or below this are zero; negative values are SPQR's own
 *   codes, where the default derives one and `-1.0` disables detection.
 */
public data class SpqrOptions(
    val ordering: SpqrOrdering = SpqrOrdering.DEFAULT,
    val rankTolerance: Double = SPQR_DEFAULT_TOL,
)

/**
 * Where a SPQR binding looks for its library, beside the policy it factorizes under.
 *
 * @property libraryPath an explicit library to load, or null to search the platform chain.
 * @property searchDirectory a directory tried before the platform chain, which is how a bundled artifact
 *   reaches its own SPQR.
 * @property options the numerical policy this binding factorizes under.
 */
public data class SpqrConfig(
    val libraryPath: String? = null,
    val searchDirectory: String? = null,
    val options: SpqrOptions = SpqrOptions(),
)
