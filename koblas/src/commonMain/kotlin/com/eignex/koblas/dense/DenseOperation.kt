package com.eignex.koblas.dense

/**
 * The dense Level 1 operations, named so a caller can ask where one of them executes.
 *
 * Level 1 only. A built-in Level 2 or 3 call is named by [DenseMatrixOperation], which asks a different
 * question: not which kernel serves one run of a vector, but what a whole matrix call executes. An explicit
 * host binding's operations are named by [com.eignex.koblas.vendor.BlasOperation], alongside the CBLAS
 * symbols they resolve to.
 */
public enum class DenseOperation {
    /** dot execution. */
    Dot,

    /** sum execution. */
    Sum,

    /** nrm2 execution. */
    Nrm2,

    /** iamax execution. */
    Iamax,

    /** asum execution. */
    Asum,

    /** axpy execution. */
    Axpy,

    /** scale execution. */
    Scale,

    /** swap execution. */
    Swap,

    /** rot execution. */
    Rot,
}
