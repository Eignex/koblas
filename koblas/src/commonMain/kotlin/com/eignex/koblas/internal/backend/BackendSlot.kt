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
internal enum class BackendSlot(
    internal val from: (F64Context) -> Backend,
    internal val vectorHalf: Boolean = false,
    internal val sparse: Boolean = false,
    internal val required: Boolean = true,
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

    /** General pivoting sparse LU. */
    F64GeneralSparseLu({ it.generalSparseLu }, sparse = true),

    /** Repeated-pattern sparse LU. */
    F64RepeatedSparseLu(
        { it.repeatedSparseLu ?: com.eignex.koblas.MissingRepeatedSparseLu },
        sparse = true,
        required = false,
    ),

    /** Sparse Cholesky. */
    F64SparseCholesky({ it.sparseCholesky }, sparse = true),

    /** Sparse quasi-definite LDL. */
    F64QuasiDefiniteLdl({ it.quasiDefiniteLdl }, sparse = true),

    /** Sparse QR. */
    F64SparseQr({ it.sparseQr }, sparse = true),

    /** Simplex basis factorizations. */
    F64BasisFactorizations({ it.basisFactorizations }, sparse = true),

    /** Simplex basis solvers. */
    F64BasisSolvers({ it.basisSolvers }, sparse = true),
    ;

    internal companion object {
        /** The halves a registry has seams for, as a diagnostic listing them the way a sentence would. */
        val names: String = entries.dropLast(1).joinToString(", ") { it.name } + " or " + entries.last().name

        /** The halves that do the matrix work, the ones a context is named after. */
        val contextHalves: List<BackendSlot> = entries.filter { it.required }
        val matrixHalves: List<BackendSlot> = contextHalves.filterNot { it.vectorHalf }
        val sparseHalves: Set<BackendSlot> = entries.filterTo(mutableSetOf()) { it.sparse }
    }
}
