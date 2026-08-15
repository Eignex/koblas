package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.VectorKernels
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLinearAlgebra
import com.eignex.koblas.sparse.SparseVectorKernels

/**
 * Every backend koblas will use for a piece of work, in one object you can hold. Immutable, and itself a
 * [LinearAlgebra] and a [SparseLinearAlgebra] by delegation.
 *
 * @property vectorKernels dense vector-vector routines; every dense inner loop bottoms out here.
 * @property blas dense matrix routines.
 * @property lapack dense factorizations.
 * @property sparseVectorKernels sparse vector-vector routines.
 * @property sparseBlas sparse matrix routines.
 * @property sparseLapack sparse factorizations.
 */
public class KoblasContext(
    override val vectorKernels: VectorKernels,
    public val blas: Blas,
    public val lapack: Lapack,
    public val sparseVectorKernels: SparseVectorKernels,
    public val sparseBlas: SparseBlas,
    public val sparseLapack: SparseLapack,
) : LinearAlgebra,
    Blas by blas,
    Lapack by lapack,
    SparseLinearAlgebra,
    SparseBlas by sparseBlas,
    SparseLapack by sparseLapack {

    /**
     * The distinct names of the backends that do the matrix work, joined, such as `"openblas+reference"`.
     * The vector-kernel halves are left out; [koblasInfo] prints both parts.
     */
    override val name: String
        get() = listOf(blas, lapack, sparseBlas, sparseLapack).map { it.name }.distinct().joinToString("+")

    /** True when every half is koblas's own, so the context calls out to nothing. */
    override val isPortable: Boolean
        get() = vectorKernels.isPortable && blas.isPortable && lapack.isPortable &&
            sparseVectorKernels.isPortable && sparseBlas.isPortable && sparseLapack.isPortable

    /** The strongest half's priority, so a context is at least as preferred as the best thing in it. */
    override val priority: Int
        get() = maxOf(
            vectorKernels.priority,
            blas.priority,
            lapack.priority,
            sparseVectorKernels.priority,
            sparseBlas.priority,
            sparseLapack.priority,
        )

    /**
     * A copy with the named halves replaced and the rest kept. A replaced [vectorKernels] reaches the
     * inherited routines of halves that follow the installed context, which requires [installBackends];
     * a half built around kernels of its own always keeps them.
     */
    public fun with(
        vectorKernels: VectorKernels = this.vectorKernels,
        blas: Blas = this.blas,
        lapack: Lapack = this.lapack,
        sparseVectorKernels: SparseVectorKernels = this.sparseVectorKernels,
        sparseBlas: SparseBlas = this.sparseBlas,
        sparseLapack: SparseLapack = this.sparseLapack,
    ): KoblasContext = KoblasContext(
        vectorKernels = vectorKernels,
        blas = blas,
        lapack = lapack,
        sparseVectorKernels = sparseVectorKernels,
        sparseBlas = sparseBlas,
        sparseLapack = sparseLapack,
    )

    override fun toString(): String = "KoblasContext($name)"
}
