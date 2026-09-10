package com.eignex.koblas

import com.eignex.koblas.dense.DenseKernelFamilies
import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.PackedKernels
import com.eignex.koblas.sparse.SparseKernelFamilies
import com.eignex.koblas.sparse.SparseKernels

/** An exact built-in dense, indexed sparse, and sparse-panel implementation composition. */
public class BuiltinKernelProvider internal constructor(
    internal val denseKernelFamilies: DenseKernelFamilies,
    internal val sparseKernelFamilies: SparseKernelFamilies,
) {
    /** Sparse level-1 kernels. */
    public val sparseKernels: SparseKernels get() = sparseKernelFamilies.vector

    /** Standalone contiguous dense-vector kernels. */
    public val vectorKernels: DenseVectorKernels get() = denseKernelFamilies.vector

    /** Dense matrix-panel arithmetic kernels. */
    public val panelKernels: DensePanelKernels get() = denseKernelFamilies.panel

    /** Packed layout shape and tile arithmetic kernels. */
    public val packedKernels: PackedKernels get() = denseKernelFamilies.packed
}

/** Creates an immutable engine using exactly these built-in kernel families. */
public fun BuiltinKernelProvider.engine(): KoblasContext = KoblasContext(denseKernelFamilies, sparseKernelFamilies)

/**
 * Built-in providers for explicit [KoblasContext] construction and implementation comparisons.
 * A platform-specific provider is null when that implementation cannot run in the current process.
 */
public expect object BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public val scalar: BuiltinKernelProvider

    /** Compiled C kernels, or null when they are unavailable. */
    public val c: BuiltinKernelProvider?

    /** JVM Vector API kernels, or null when the Vector API module is unavailable or on a non-JVM target. */
    public val simd: BuiltinKernelProvider?
}
