package com.eignex.koblas

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseLapack
import com.eignex.koblas.sparse.F64SparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseVectorKernels

/**
 * Every backend koblas will use for a piece of work, in one object you can hold. Immutable, and itself a
 * [F64LinearAlgebra] and a [F64SparseLinearAlgebra] by delegation.
 *
 * @property vectorKernels dense vector-vector routines; every dense inner loop bottoms out here.
 * @property blas dense matrix routines.
 * @property lapack dense factorizations.
 * @property sparseVectorKernels sparse vector-vector routines.
 * @property sparseBlas sparse matrix routines.
 * @property sparseLapack sparse factorizations.
 */
public class F64Context(
    override val vectorKernels: F64VectorKernels,
    public val blas: F64Blas,
    public val lapack: F64Lapack,
    public val sparseVectorKernels: F64SparseVectorKernels,
    public val sparseBlas: F64SparseBlas,
    public val sparseLapack: F64SparseLapack,
) : F64LinearAlgebra,
    F64Blas by blas,
    F64Lapack by lapack,
    F64SparseLinearAlgebra,
    F64SparseBlas by sparseBlas,
    F64SparseLapack by sparseLapack {

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

    /** True when every half can run, which a context assembled from resolved backends always can. */
    override val isAvailable: Boolean
        get() = vectorKernels.isAvailable && blas.isAvailable && lapack.isAvailable &&
            sparseVectorKernels.isAvailable && sparseBlas.isAvailable && sparseLapack.isAvailable

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
        vectorKernels: F64VectorKernels = this.vectorKernels,
        blas: F64Blas = this.blas,
        lapack: F64Lapack = this.lapack,
        sparseVectorKernels: F64SparseVectorKernels = this.sparseVectorKernels,
        sparseBlas: F64SparseBlas = this.sparseBlas,
        sparseLapack: F64SparseLapack = this.sparseLapack,
    ): F64Context = F64Context(
        vectorKernels = vectorKernels,
        blas = blas,
        lapack = lapack,
        sparseVectorKernels = sparseVectorKernels,
        sparseBlas = sparseBlas,
        sparseLapack = sparseLapack,
    )

    override fun toString(): String = "F64Context($name)"
}
