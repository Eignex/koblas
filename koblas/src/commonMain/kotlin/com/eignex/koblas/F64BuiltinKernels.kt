package com.eignex.koblas

import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.sparse.F64SparseKernels

/** An exact built-in dense and sparse level-1 implementation pair. */
@ExperimentalKoblasApi
public class F64BuiltinKernelProvider internal constructor(
    /** Dense level-1 kernels. */
    public val kernels: F64Kernels,
    /** Sparse level-1 kernels. */
    public val sparseKernels: F64SparseKernels,
)

/**
 * Built-in level-1 providers for explicit [F64Context] configuration and implementation comparisons.
 * A platform-specific provider is null when that implementation cannot run in the current process.
 */
@ExperimentalKoblasApi
public expect object F64BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public val scalar: F64BuiltinKernelProvider

    /** Compiled C kernels, or null when they are unavailable. */
    public val c: F64BuiltinKernelProvider?

    /** JVM Vector API kernels, or null when the Vector API module is unavailable or on a non-JVM target. */
    public val simd: F64BuiltinKernelProvider?
}
