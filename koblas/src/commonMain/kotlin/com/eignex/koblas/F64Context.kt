package com.eignex.koblas

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Decompositions
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseKernels
import com.eignex.koblas.sparse.F64SparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseLu

/**
 * Every backend koblas will use for a piece of work, in one object you can hold. Immutable, and itself a
 * [F64LinearAlgebra] and a [F64SparseLinearAlgebra] by delegation.
 *
 * @property kernels dense vector-vector routines; every dense inner loop bottoms out here.
 * @property blas dense matrix routines.
 * @property decompositions dense factorizations.
 * @property sparseKernels sparse vector-vector routines.
 * @property sparseBlas sparse matrix routines.
 * @property sparseLu sparse factorizations.
 */
public class F64Context(
    override val kernels: F64Kernels,
    public val blas: F64Blas,
    public val decompositions: F64Decompositions,
    override val sparseKernels: F64SparseKernels,
    public val sparseBlas: F64SparseBlas,
    public val sparseLu: F64SparseLu,
) : F64LinearAlgebra,
    F64Blas by blas,
    F64Decompositions by decompositions,
    F64SparseLinearAlgebra,
    F64SparseBlas by sparseBlas,
    F64SparseLu by sparseLu {

    /**
     * The distinct names of the backends that do the matrix work, joined, such as `"openblas+reference"`.
     * The vector-kernel halves are left out; [koblasInfo] prints both parts.
     */
    override val name: String
        get() = BackendSlot.matrixHalves.map { backendFor(it).name }.distinct().joinToString("+")

    /** True when every half is koblas's own, so the context calls out to nothing. */
    override val isPortable: Boolean get() = BackendSlot.entries.all { backendFor(it).isPortable }

    /** True when every half can run, which a context assembled from resolved backends always can. */
    override val isAvailable: Boolean get() = BackendSlot.entries.all { backendFor(it).isAvailable }

    /** The strongest half's priority, so a context is at least as preferred as the best thing in it. */
    override val priority: Int get() = BackendSlot.entries.maxOf { backendFor(it).priority }

    /**
     * A copy with the named halves replaced and the rest kept. A replaced [kernels] reaches the
     * inherited routines of halves that follow the installed context, which requires [installBackends];
     * a half built around kernels of its own always keeps them.
     */
    public fun with(
        kernels: F64Kernels = this.kernels,
        blas: F64Blas = this.blas,
        decompositions: F64Decompositions = this.decompositions,
        sparseKernels: F64SparseKernels = this.sparseKernels,
        sparseBlas: F64SparseBlas = this.sparseBlas,
        sparseLu: F64SparseLu = this.sparseLu,
    ): F64Context = F64Context(
        kernels = kernels,
        blas = blas,
        decompositions = decompositions,
        sparseKernels = sparseKernels,
        sparseBlas = sparseBlas,
        sparseLu = sparseLu,
    )

    override fun toString(): String = "F64Context($name)"
}
