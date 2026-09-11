package com.eignex.koblas

import com.eignex.koblas.dense.BuiltinBlas
import com.eignex.koblas.dense.DenseBlas
import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.PackedKernels
import com.eignex.koblas.dense.PackedPanels
import com.eignex.koblas.sparse.IndexedSparseKernels
import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.SparsePanelKernels

/** The immutable platform-selected BLAS engine used by top-level convenience operations. */
public val koblas: KoblasEngine = BuiltinEngines.simd ?: BuiltinEngines.c ?: BuiltinEngines.scalar

/**
 * An immutable dense and sparse BLAS engine.
 *
 * The default [koblas] instance is selected once for the platform. Tests and benchmarks can construct an
 * exact scalar, C, or SIMD composition from [BuiltinEngines] without changing process-global state. Each
 * composition binds its dense vector, dense panel, packed tile, indexed sparse, and sparse panel kernels once.
 */
public class KoblasEngine internal constructor(
    /** Standalone contiguous dense-vector kernels. */
    public val vectorKernels: DenseVectorKernels,
    /** Dense matrix-panel arithmetic kernels. */
    public val panelKernels: DensePanelKernels,
    /** Packed layout shape and tile arithmetic kernels. */
    public val packedKernels: PackedKernels,
    /** Sparse-vector kernels used by sparse convenience operations. */
    public val sparseKernels: SparseKernels,
    internal val indexedSparseKernels: IndexedSparseKernels,
) : DenseBlas by BuiltinBlas(vectorKernels, panelKernels, packedKernels),
    SparseBlas by SparseAlgorithms(
        vectorKernels,
        indexedSparseKernels,
        SparsePanelKernels(vectorKernels, panelKernels),
    ) {
    /** Packed panel operations bound to this engine's exact packed kernels. */
    public val packedPanels: PackedPanels = PackedPanels(packedKernels)

    /** Short read-only implementation description for logs and benchmark attribution. */
    public val name: String get() = "built-in/${vectorKernels.name}/${sparseKernels.name}"

    override fun toString(): String = "KoblasEngine($name)"
}

/** Built-in engines for implementation comparisons. */
public expect object BuiltinEngines {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public val scalar: KoblasEngine

    /** Compiled C kernels, or null when they are unavailable. */
    public val c: KoblasEngine?

    /** JVM Vector API kernels, or null when the Vector API module is unavailable or on a non-JVM target. */
    public val simd: KoblasEngine?
}
