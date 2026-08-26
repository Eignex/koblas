package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.F64Context

/**
 * The halves of the seam a backend can implement.
 *
 * Each entry knows how to read its half out of a context, so the places that need every half in turn read
 * them from [entries] rather than listing them all again. The constant names are the interface names, which
 * is what makes them usable in a diagnostic. A vector half is one sitting below the matrix routines, as the
 * vector-vector kernels do.
 */
public enum class BackendSlot(
    internal val from: (F64Context) -> Backend,
    internal val vectorHalf: Boolean = false,
    internal val sparse: Boolean = false,
) {
    /** Dense vector-vector routines. */
    F64Kernels({ it.kernels }, vectorHalf = true),

    /** Dense matrix routines. */
    F64Blas({ it.blas }),

    /** Dense factorizations. */
    F64Decompositions({ it.decompositions }),

    /** Sparse vector-vector routines. */
    F64SparseKernels({ it.sparseKernels }, vectorHalf = true, sparse = true),

    /** Sparse matrix routines. */
    F64SparseBlas({ it.sparseBlas }, sparse = true),

    /** Sparse factorizations. */
    F64SparseDecompositions({ it.sparseDecompositions }, sparse = true),

    /** Simplex basis solvers. */
    F64BasisSolvers({ it.basisSolvers }, sparse = true),
    ;

    internal companion object {
        /** The halves a registry has seams for, as a diagnostic listing them the way a sentence would. */
        val names: String = entries.dropLast(1).joinToString(", ") { it.name } + " or " + entries.last().name

        /** The halves that do the matrix work, the ones a context is named after. */
        val matrixHalves: List<BackendSlot> = entries.filterNot { it.vectorHalf }

        /** The halves a dense pin speaks for, and the rest, which a sparse pin speaks for. */
        val denseHalves: Set<BackendSlot> = entries.filterNotTo(mutableSetOf()) { it.sparse }
        val sparseHalves: Set<BackendSlot> = entries.filterTo(mutableSetOf()) { it.sparse }
    }
}
