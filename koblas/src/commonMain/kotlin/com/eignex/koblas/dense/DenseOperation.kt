package com.eignex.koblas.dense

/** The dense Level 1 operations, which name a length crossover and the body a length reaches. */
internal enum class DenseOperation {
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
