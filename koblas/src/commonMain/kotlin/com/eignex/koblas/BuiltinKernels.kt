package com.eignex.koblas

import com.eignex.koblas.dense.Kernels
import com.eignex.koblas.sparse.SparseKernels

/** An exact built-in dense and sparse level-1 implementation pair. */
@ExperimentalKoblasApi
public class BuiltinKernelProvider internal constructor(
    /** Dense level-1 kernels. */
    public val kernels: Kernels,
    /** Sparse level-1 kernels. */
    public val sparseKernels: SparseKernels,
)

/**
 * Built-in level-1 providers for explicit [KoblasContext] configuration and implementation comparisons.
 * A platform-specific provider is null when that implementation cannot run in the current process.
 */
@ExperimentalKoblasApi
public expect object BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public val scalar: BuiltinKernelProvider

    /** Compiled C kernels, or null when they are unavailable. */
    public val c: BuiltinKernelProvider?

    /** JVM Vector API kernels, or null when the Vector API module is unavailable or on a non-JVM target. */
    public val simd: BuiltinKernelProvider?
}
