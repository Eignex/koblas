package com.eignex.koblas.dense

/**
 * A built-in dense Level 2 or 3 entry point, which platform host policies key their size crossovers on.
 *
 * The variants a caller selects with a flag are separate entries where they execute differently: a
 * transposed matrix-vector product reduces down each stored column while the untransposed one accumulates
 * those columns into the destination.
 */
internal enum class DenseMatrixOperation {
    /** `y = alpha·A·x + beta·y`, accumulating each stored column into the destination. */
    Gemv,

    /** `y = alpha·Aᵀ·x + beta·y`, reducing each stored column against the input. */
    GemvTransposed,

    /** Selected-triangle symmetric matrix-vector product. */
    Symv,

    /** `A += alpha·x·yᵀ`. */
    Ger,

    /** Rank-one update of a rectangle within an existing matrix, using the owned panel kernels. */
    GerWindow,

    /** `A += alpha·x·xᵀ` in the selected triangle. */
    Syr,

    /** `A += alpha·(x·yᵀ + y·xᵀ)` in the selected triangle. */
    Syr2,

    /** Triangular matrix-vector product, accumulating each stored column into the destination. */
    Trmv,

    /** Triangular matrix-vector product, reducing each stored column against the input. */
    TrmvTransposed,

    /** Triangular solve, substituting forward or backward over the stored columns. */
    Trsv,

    /** Triangular solve whose substitution reduces each stored column against the finished entries. */
    TrsvTransposed,

    /** `C = alpha·op(A)·op(B) + beta·C`. */
    Gemm,

    /** The same product written into one triangle of a square destination. */
    Gemmt,

    /** Symmetric product against a dense block, on either side. */
    Symm,

    /** `C = alpha·op(A)·op(A)ᵀ + beta·C` in the selected triangle. */
    Syrk,

    /** `C = alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + beta·C` in the selected triangle. */
    Syr2k,

    /** Triangular matrix multiply against a dense block. */
    Trmm,

    /** Triangular solve against a dense block. */
    Trsm,
}
