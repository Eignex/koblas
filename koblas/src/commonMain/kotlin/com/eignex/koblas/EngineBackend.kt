package com.eignex.koblas

import com.eignex.koblas.vendor.Blas

/** A source of typed arithmetic implementations, independently of the policy selecting them. */
public sealed interface EngineBackend {
    /** Stable identifier used in routing configurations. */
    public val name: String

    /** Implementations supplied directly by this backend; portable fallbacks are not included. */
    public val implementations: Set<EngineImplementation>
}

/** Arithmetic boundaries a routing configuration can select independently. */
public enum class EngineImplementation {
    /** Dense dot product. */
    Dot,

    /** Dense sum. */
    Sum,

    /** Dense rescaled norm. */
    Nrm2,

    /** Dense maximum magnitude index. */
    Iamax,

    /** Dense absolute sum. */
    Asum,

    /** Dense scaled addition. */
    Axpy,

    /** Dense scaling. */
    Scale,

    /** Dense exchange. */
    Swap,

    /** Dense rotation. */
    Rot,

    /** Indexed dot against a dense vector. */
    IndexedDot,

    /** Panel arithmetic within shared schedules. */
    Panels,

    /** Packed product arithmetic and its packing geometry. */
    Products,

    /** Diagonal substitution within triangular schedules. */
    Triangular,

    /** Whole matrix vector product. */
    Gemv,

    /** Whole transposed matrix vector product. */
    GemvTransposed,

    /** Whole symmetric matrix vector product. */
    Symv,

    /** Whole rank one update. */
    Ger,

    /** Whole symmetric rank one update. */
    Syr,

    /** Whole symmetric rank two update. */
    Syr2,

    /** Whole triangular matrix vector product. */
    Trmv,

    /** Whole transposed triangular matrix vector product. */
    TrmvTransposed,

    /** Whole triangular vector substitution. */
    Trsv,

    /** Whole transposed triangular vector substitution. */
    TrsvTransposed,

    /** Whole matrix product. */
    Gemm,

    /** Whole selected triangle product. */
    Gemmt,

    /** Whole symmetric product. */
    Symm,

    /** Whole symmetric rank k update. */
    Syrk,

    /** Whole symmetric rank two k update. */
    Syr2k,

    /** Whole triangular matrix product. */
    Trmm,

    /** Whole triangular matrix substitution. */
    Trsm,
}

/** Built-in arithmetic sources. Missing optional runtimes are represented by null. */
public object EngineBackends {
    /** Portable coverage for every operation without resolving a host library. */
    public val portable: EngineBackend get() = portableBackend

    /** Owned JVM Vector API arithmetic, or null when unavailable on this runtime. */
    public val simd: EngineBackend? get() = simdBackend()

    /** Adapts an already opened binding without changing the default engine. */
    public fun vendor(binding: Blas): EngineBackend = vendorBackend(binding)
}

internal expect fun simdBackend(): BackendContributions?
internal expect fun vendorBackend(binding: Blas): BackendContributions
