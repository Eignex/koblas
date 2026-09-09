package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.BuiltinBlas
import com.eignex.koblas.dense.Kernels
import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels

/**
 * An immutable dense and sparse BLAS engine.
 *
 * The default [koblas] instance is selected once for the platform. Tests and benchmarks can construct an
 * exact scalar, C, or SIMD engine from [BuiltinKernels] without changing process-global state.
 */
public class KoblasContext internal constructor(
    /** Dense vector and packed-panel kernels used by [blas]. */
    override val kernels: Kernels,
    /** Sparse-vector kernels used by sparse convenience operations. */
    public val sparseKernels: SparseKernels,
    /** Shared dense matrix algorithms bound to [kernels]. */
    public val blas: Blas = BuiltinBlas(kernels),
    /** Shared sparse matrix algorithms bound to [kernels]. */
    public val sparseBlas: SparseBlas = SparseAlgorithms(kernels),
) : Blas by blas,
    SparseBlas by sparseBlas {

    /** Short read-only implementation description for logs and benchmark attribution. */
    override val name: String get() = "built-in/${kernels.name}/${sparseKernels.name}"

    override fun toString(): String = "KoblasContext($name)"
}
