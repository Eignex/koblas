package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.ReferenceBackend
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
     * A copy with the named halves replaced and the rest kept. Replacing [vectorKernels] rebinds any
     * reference half that has no kernels of its own, so the new kernels reach the work this context does
     * without having to be installed globally. A half built around particular kernels keeps them.
     */
    public fun with(
        vectorKernels: VectorKernels = this.vectorKernels,
        blas: Blas = this.blas,
        lapack: Lapack = this.lapack,
        sparseVectorKernels: SparseVectorKernels = this.sparseVectorKernels,
        sparseBlas: SparseBlas = this.sparseBlas,
        sparseLapack: SparseLapack = this.sparseLapack,
    ): KoblasContext {
        val rebound = if (vectorKernels !== this.vectorKernels) ReferenceBackend(vectorKernels) else null
        return KoblasContext(
            vectorKernels = vectorKernels,
            blas = if (rebound != null && blas is ReferenceBackend && blas.followsContext) rebound else blas,
            lapack = if (rebound != null && lapack is ReferenceBackend && lapack.followsContext) rebound else lapack,
            sparseVectorKernels = sparseVectorKernels,
            sparseBlas = sparseBlas,
            sparseLapack = sparseLapack,
        )
    }

    override fun toString(): String = "KoblasContext($name)"
}
