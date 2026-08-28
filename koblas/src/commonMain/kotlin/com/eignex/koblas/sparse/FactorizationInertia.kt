package com.eignex.koblas.sparse

/** Pivot-sign counts for a symmetric indefinite factorization. */
public data class FactorizationInertia(
    /** Positive pivots. */
    public val positive: Int,
    /** Negative pivots. */
    public val negative: Int,
    /** Zero pivots. */
    public val zero: Int,
)
