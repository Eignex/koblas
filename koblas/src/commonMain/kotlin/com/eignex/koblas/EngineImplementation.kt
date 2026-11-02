package com.eignex.koblas

/** Arithmetic boundaries a routing configuration can select independently. */
public enum class EngineImplementation(
    /** Unsuffixed LP64 CBLAS entry point, when this contribution is a foreign call. */
    public val entryPoint: String = "",
    /** BLAS level used to group measurements. */
    public val level: Int = 1,
) {
    /** Dense dot product. */
    Dot("cblas_ddot", 1),

    /** Dense sum. */
    Sum,

    /** Dense rescaled norm. */
    Nrm2("cblas_dnrm2", 1),

    /** Dense maximum magnitude index. */
    Iamax("cblas_idamax", 1),

    /** Dense absolute sum. */
    Asum("cblas_dasum", 1),

    /** Dense scaled addition. */
    Axpy("cblas_daxpy", 1),

    /** Dense scaling. */
    Scale("cblas_dscal", 1),

    /** Dense exchange. */
    Swap("cblas_dswap", 1),

    /** Dense rotation. */
    Rot("cblas_drot", 1),

    /** Dense storage copy. */
    Copy("cblas_dcopy"),

    /** Indexed dot against a dense vector. */
    IndexedDot,

    /** Merge of two stored sparse supports. */
    IndexedSparseDot,

    /** Stored sparse updates into dense storage. */
    IndexedAxpy,

    /** Stored sparse assignment into dense storage. */
    IndexedScatter,

    /** Reading a stored support from dense storage. */
    IndexedGather,

    /** Reading and zeroing a stored support. */
    IndexedGatherZero,

    /** Rescaled norm over indexed dense positions. */
    IndexedNrm2,

    /** Panel arithmetic within shared schedules. */
    Panels,

    /** Packed product arithmetic and its packing geometry. */
    Products,

    /** Diagonal substitution within triangular schedules. */
    Triangular,

    /** Whole matrix vector product. */
    Gemv("cblas_dgemv", 2),

    /** Whole transposed matrix vector product. */
    GemvTransposed("cblas_dgemv", 2),

    /** Whole symmetric matrix vector product. */
    Symv("cblas_dsymv", 2),

    /** Whole rank one update. */
    Ger("cblas_dger", 2),

    /** Whole symmetric rank one update. */
    Syr("cblas_dsyr", 2),

    /** Whole symmetric rank two update. */
    Syr2("cblas_dsyr2", 2),

    /** Whole triangular matrix vector product. */
    Trmv("cblas_dtrmv", 2),

    /** Whole transposed triangular matrix vector product. */
    TrmvTransposed("cblas_dtrmv", 2),

    /** Whole triangular vector substitution. */
    Trsv("cblas_dtrsv", 2),

    /** Whole transposed triangular vector substitution. */
    TrsvTransposed("cblas_dtrsv", 2),

    /** Whole matrix product. */
    Gemm("cblas_dgemm", 3),

    /** Whole selected triangle product. */
    Gemmt("cblas_dgemmt", 3),

    /** Whole symmetric product. */
    Symm("cblas_dsymm", 3),

    /** Whole symmetric rank k update. */
    Syrk("cblas_dsyrk", 3),

    /** Whole symmetric rank two k update. */
    Syr2k("cblas_dsyr2k", 3),

    /** Whole triangular matrix product. */
    Trmm("cblas_dtrmm", 3),

    /** Whole triangular matrix substitution. */
    Trsm("cblas_dtrsm", 3),
    ;

    internal val required: Boolean get() = entryPoint.isNotEmpty() && this != Gemmt && !name.endsWith("Transposed")
}
