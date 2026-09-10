package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.BuiltinBlas
import com.eignex.koblas.dense.DenseKernelFamilies
import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.PackedKernels
import com.eignex.koblas.dense.PackedPanels
import com.eignex.koblas.dense.platformDenseKernelFamilies
import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernelFamilies
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.platformSparseKernelFamilies

/** The immutable platform-selected BLAS engine used by top-level convenience operations. */
public val koblas: KoblasContext = KoblasContext(
    platformDenseKernelFamilies,
    platformSparseKernelFamilies,
)

/**
 * An immutable dense and sparse BLAS engine.
 *
 * The default [koblas] instance is selected once for the platform. Tests and benchmarks can construct an
 * exact scalar, C, or SIMD composition from [BuiltinKernels] without changing process-global state. Each
 * composition binds its dense vector, dense panel, packed tile, indexed sparse, and sparse panel families once.
 */
public class KoblasContext internal constructor(
    internal val denseKernelFamilies: DenseKernelFamilies,
    internal val sparseKernelFamilies: SparseKernelFamilies,
) : Blas by BuiltinBlas(denseKernelFamilies),
    SparseBlas by SparseAlgorithms(
        denseKernelFamilies.vector,
        sparseKernelFamilies.indexed,
        sparseKernelFamilies.panel,
    ) {

    /** Sparse-vector kernels used by sparse convenience operations. */
    public val sparseKernels: SparseKernels get() = sparseKernelFamilies.vector

    /** Standalone contiguous dense-vector kernels. */
    public val vectorKernels: DenseVectorKernels get() = denseKernelFamilies.vector

    /** Dense matrix-panel arithmetic kernels. */
    public val panelKernels: DensePanelKernels get() = denseKernelFamilies.panel

    /** Packed layout shape and tile arithmetic kernels. */
    public val packedKernels: PackedKernels get() = denseKernelFamilies.packed

    /** Packed panel operations bound to this engine's exact packed kernels. */
    public val packedPanels: PackedPanels = PackedPanels(denseKernelFamilies.packed)

    /** Short read-only implementation description for logs and benchmark attribution. */
    public val name: String get() = "built-in/${vectorKernels.name}/${sparseKernels.name}"

    override fun toString(): String = "KoblasContext($name)"
}

/** Built-in engines for implementation comparisons. */
public expect object BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public val scalar: KoblasContext

    /** Compiled C kernels, or null when they are unavailable. */
    public val c: KoblasContext?

    /** JVM Vector API kernels, or null when the Vector API module is unavailable or on a non-JVM target. */
    public val simd: KoblasContext?
}
