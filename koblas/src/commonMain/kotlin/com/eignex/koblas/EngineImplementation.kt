package com.eignex.koblas

/** Arithmetic boundaries a routing configuration can select independently. */
public enum class EngineImplementation(
    /** Unsuffixed LP64 CBLAS entry point, when this contribution is a foreign call. */
    internal val entryPoint: String = "",
    internal val matrix: Boolean = false,
) {
    /** Dense dot product. */
    Dot("cblas_ddot"),

    /** Dense sum. */
    Sum,

    /** Dense rescaled norm. */
    Nrm2("cblas_dnrm2"),

    /** Dense maximum magnitude index. */
    Iamax("cblas_idamax"),

    /** Dense absolute sum. */
    Asum("cblas_dasum"),

    /** Dense scaled addition. */
    Axpy("cblas_daxpy"),

    /** Dense scaling. */
    Scale("cblas_dscal"),

    /** Dense exchange. */
    Swap("cblas_dswap"),

    /** Dense rotation. */
    Rot("cblas_drot"),

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
    Gemv("cblas_dgemv", matrix = true),

    /** Whole transposed matrix vector product. */
    GemvTransposed("cblas_dgemv", matrix = true),

    /** Whole symmetric matrix vector product. */
    Symv("cblas_dsymv", matrix = true),

    /** Whole rank one update. */
    Ger("cblas_dger", matrix = true),

    /** Whole symmetric rank one update. */
    Syr("cblas_dsyr", matrix = true),

    /** Whole symmetric rank two update. */
    Syr2("cblas_dsyr2", matrix = true),

    /** Whole triangular matrix vector product. */
    Trmv("cblas_dtrmv", matrix = true),

    /** Whole transposed triangular matrix vector product. */
    TrmvTransposed("cblas_dtrmv", matrix = true),

    /** Whole triangular vector substitution. */
    Trsv("cblas_dtrsv", matrix = true),

    /** Whole transposed triangular vector substitution. */
    TrsvTransposed("cblas_dtrsv", matrix = true),

    /** Whole matrix product. */
    Gemm("cblas_dgemm", matrix = true),

    /** Whole selected triangle product. */
    Gemmt("cblas_dgemmt", matrix = true),

    /** Whole symmetric product. */
    Symm("cblas_dsymm", matrix = true),

    /** Whole symmetric rank k update. */
    Syrk("cblas_dsyrk", matrix = true),

    /** Whole symmetric rank two k update. */
    Syr2k("cblas_dsyr2k", matrix = true),

    /** Whole triangular matrix product. */
    Trmm("cblas_dtrmm", matrix = true),

    /** Whole triangular matrix substitution. */
    Trsm("cblas_dtrsm", matrix = true),
    ;

    internal val family: Boolean get() = this == Panels || this == Products || this == Triangular
    internal val vector: Boolean get() = !matrix && !family
}
