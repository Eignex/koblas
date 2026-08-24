package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.F64Context

/**
 * The halves of the seam a backend can implement.
 *
 * Each entry knows how to read its half out of a context, so the places that need every half in turn read
 * them from [entries] rather than listing the six again. The constant names are the interface names, which
 * is what makes them usable in a diagnostic. A vector half is one sitting below the matrix routines, as the
 * vector-vector kernels do.
 */
public enum class BackendSlot(internal val from: (F64Context) -> Backend, internal val vectorHalf: Boolean = false) {
    /** Dense vector-vector routines. */
    F64Kernels({ it.kernels }, vectorHalf = true),

    /** Dense matrix routines. */
    F64Blas({ it.blas }),

    /** Dense factorizations. */
    F64Decompositions({ it.lapack }),

    /** Sparse vector-vector routines. */
    F64SparseKernels({ it.sparseKernels }, vectorHalf = true),

    /** Sparse matrix routines. */
    F64SparseBlas({ it.sparseBlas }),

    /** Sparse factorizations. */
    F64SparseLu({ it.sparseLu }),
    ;

    internal companion object {
        /** The halves a registry has seams for, as a diagnostic listing them the way a sentence would. */
        val names: String = entries.dropLast(1).joinToString(", ") { it.name } + " or " + entries.last().name

        /** The halves that do the matrix work, the ones a context is named after. */
        val matrixHalves: List<BackendSlot> = entries.filterNot { it.vectorHalf }
    }
}
