package com.eignex.koblas.dense

/**
 * The dense Level 1 operations, named so a caller can ask where one of them executes.
 *
 * Level 1 only: Level 2 and 3 belong to the vendor, and its operations are named by [com.eignex.koblas.vendor.BlasOperation]
 * alongside the CBLAS symbols they resolve to.
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

    /** rotm execution. */
    Rotm,
}
