package com.eignex.koblas

/** Norm choices shared by [Vector.norm] and [Matrix.norm]; each operation documents its supported choices. */
public enum class Norm {
    /** Sum of absolute vector entries, or the largest absolute column sum of a matrix. */
    One,

    /** Euclidean vector norm. The spectral matrix norm is not supported. */
    Two,

    /** Maximum absolute vector entry, or the largest absolute row sum of a matrix. */
    Infinity,

    /** Square root of the sum of squared matrix entries. Not supported for vectors. */
    Frobenius,

    /** Maximum absolute entry. Equivalent to [Infinity] for vectors; distinct from it for matrices. */
    MaxAbs,
}

/**
 * Whether the single compute thread the binding requires was confirmed against the library itself.
 *
 * This is evidence, not a setting. Nothing reads it to decide anything and there is no way to change what it
 * reports; a benchmark records it because a run described as single-threaded should say whether that was
 * checked or only asked for.
 */
public enum class ThreadEvidence {
    /** The library reported one compute thread after being held to it. */
    Confirmed,

    /**
     * The library exposes no thread-count entry point, so the requirement could not be read back.
     *
     * Accelerate is the case this exists for: it has no CBLAS thread-count symbol at all, and its only lever is
     * an environment variable read before the process starts, which cannot be observed from inside it.
     */
    Unconfirmed,
}

/** Operands whose shapes do not fit the routine. */
public class DimensionMismatch(message: String) : IllegalArgumentException(message)

/** Arithmetic capabilities supplied by a [KoblasEngine]. */
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

/**
 * Marks direct access to live structural or factorization storage. Mutating these buffers can invalidate
 * later operations; access is intended for backend interoperability and specialized kernels.
 */
@RequiresOptIn(
    message = "This is live internal storage; mutating it can invalidate the owning object",
    level = RequiresOptIn.Level.ERROR,
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CONSTRUCTOR, AnnotationTarget.PROPERTY)
public annotation class UnsafeKoblasApi
