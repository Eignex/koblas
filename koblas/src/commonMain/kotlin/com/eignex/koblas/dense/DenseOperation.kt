package com.eignex.koblas.dense

/** Dense operation identities for inspecting the immutable engine's current dispatch decisions. */
public enum class DenseOperation(internal val key: String, internal val nativeOperation: Int) {
    /** dot execution. */
    Dot("dot", 1),

    /** sum execution. */
    Sum("sum", 7),

    /** ssqd execution. */
    Ssqd("ssqd", 2),

    /** nrm2 execution. */
    Nrm2("nrm2", 6),

    /** iamax execution. */
    Iamax("iamax", 9),

    /** asum execution. */
    Asum("asum", 8),

    /** axpy execution. */
    Axpy("axpy", 3),

    /** scale execution. */
    Scale("scale", 5),

    /** swap execution. */
    Swap("swap", 11),

    /** rot execution. */
    Rot("rot", 15),

    /** rotm execution. */
    Rotm("rotm", 15),

    /** dot4 execution. */
    Dot4("dot4", 12),

    /** axpy4 execution. */
    Axpy4("axpy4", 13),

    /** dot axpy execution. */
    DotAxpy("dot.axpy", 14),

    /** axpy arithmetic execution. */
    AxpyArithmetic("axpy.arithmetic", 4),

    /** gemm tile execution. */
    GemmTile("gemm.tile", 10),

    /** trsm tile execution. */
    TrsmTile("trsm.tile", 23),

    /** gemm trsm tile execution. */
    GemmTrsmTile("gemm.trsm.tile", 24),
    ;

    internal val packed: Boolean get() = this == GemmTile || this == TrsmTile || this == GemmTrsmTile
    internal val panel: Boolean get() = this == Dot4 || this == Axpy4 || this == DotAxpy || this == AxpyArithmetic
}
