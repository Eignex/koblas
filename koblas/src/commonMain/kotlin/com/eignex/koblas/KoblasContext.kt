package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.BuiltinBlas
import com.eignex.koblas.dense.DenseKernelFamilies
import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.PackedKernels
import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels

/**
 * An immutable dense and sparse BLAS engine.
 *
 * The default [koblas] instance is selected once for the platform. Tests and benchmarks can construct an
 * exact scalar, C, or SIMD composition from [BuiltinKernels] without changing process-global state. Each
 * composition binds its vector, matrix-panel, and packed-tile families once.
 */
public class KoblasContext internal constructor(
    internal val denseKernelFamilies: DenseKernelFamilies,
    /** Sparse-vector kernels used by sparse convenience operations. */
    public val sparseKernels: SparseKernels,
    /** Shared dense matrix algorithms bound to the immutable kernel families. */
    public val blas: Blas = BuiltinBlas(denseKernelFamilies),
    /** Shared sparse matrix algorithms bound to the dense vector and panel families. */
    public val sparseBlas: SparseBlas = SparseAlgorithms(denseKernelFamilies.vector, denseKernelFamilies.panel),
) : Blas by blas,
    SparseBlas by sparseBlas {

    /** Standalone contiguous dense-vector kernels. */
    public val vectorKernels: DenseVectorKernels get() = denseKernelFamilies.vector

    /** Dense matrix-panel arithmetic kernels. */
    public val panelKernels: DensePanelKernels get() = denseKernelFamilies.panel

    /** Packed layout shape and tile arithmetic kernels. */
    public val packedKernels: PackedKernels get() = denseKernelFamilies.packed

    /** Short read-only implementation description for logs and benchmark attribution. */
    override val name: String get() = "built-in/${vectorKernels.name}/${sparseKernels.name}"

    override fun toString(): String = "KoblasContext($name)"
}
