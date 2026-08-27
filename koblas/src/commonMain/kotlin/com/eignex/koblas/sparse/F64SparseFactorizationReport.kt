package com.eignex.koblas.sparse

/** Pivot-sign counts for a symmetric indefinite factorization. */
public data class F64FactorizationInertia(
    /** Positive pivots. */
    public val positive: Int,
    /** Negative pivots. */
    public val negative: Int,
    /** Zero pivots. */
    public val zero: Int,
)

/**
 * Common diagnostics sampled from a sparse factorization. Nullable fields mean the provider does not expose
 * that statistic; a valid zero is therefore never used as an unavailable sentinel. [details] is the stable
 * extension point for provider facts that do not yet have a common field.
 */
public data class F64SparseFactorizationReport(
    /** Backend that owns the factorization. */
    public val provider: String,
    /** Matrix order. */
    public val order: Int,
    /** Stored entries in the factors, or null when unavailable. */
    public val factorNonzeros: Int?,
    /** Smallest-to-largest pivot magnitude ratio, or null when unavailable. */
    public val reciprocalPivotRange: Double?,
    /** Factor entries divided by input entries, or null when the input count is unavailable. */
    public val fillRatio: Double? = null,
    /** Pivot-position to original-row mapping, or null when unavailable. */
    public val rowPermutation: List<Int>? = null,
    /** Pivot-position to original-column mapping, or null when unavailable. */
    public val columnPermutation: List<Int>? = null,
    /** Ordering strategy, or null when unavailable. */
    public val ordering: String? = null,
    /** Scaling strategy, or null when unavailable. */
    public val scaling: String? = null,
    /** Iterative-refinement steps, or null when unavailable or inapplicable. */
    public val refinementSteps: Int? = null,
    /** Provider-reported factor memory in bytes, or null when unavailable. */
    public val memoryBytes: Long? = null,
    /** Symmetric pivot signs, or null when unavailable or inapplicable. */
    public val inertia: F64FactorizationInertia? = null,
    /** Provider-specific stable fields not represented above. */
    public val details: Map<String, String> = emptyMap(),
)

internal fun F64SparseFactorization.basicReport(provider: String): F64SparseFactorizationReport =
    F64SparseFactorizationReport(
        provider = provider,
        order = n,
        factorNonzeros = nnz,
        reciprocalPivotRange = rcond,
    )
