package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.BackendRole
import com.eignex.koblas.F64Context
import com.eignex.koblas.MissingRepeatedSparseLu
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Decompositions
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.F64RoutedKernels
import com.eignex.koblas.sparse.F64BasisFactorizations
import com.eignex.koblas.sparse.F64GeneralSparseLu
import com.eignex.koblas.sparse.F64PlatformSparseKernels
import com.eignex.koblas.sparse.F64QuasiDefiniteLdl
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64RepeatedSparseLu
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseCholesky
import com.eignex.koblas.sparse.F64SparseKernels
import com.eignex.koblas.sparse.F64SparseQr
import com.eignex.koblas.sparse.basis.F64BasisSolvers

/**
 * The halves of the seam a backend can implement, and everything either selection path needs to know about
 * one: which interface fills it, how to read it out of a context, what koblas's own implementation is, the
 * keys a deployment pins it with, and whether a specialized provider should leave it alone.
 *
 * One entry per half rather than one table per question. The registry and [com.eignex.koblas.F64ContextBuilder]
 * both select backends, and a half described in two places is a half the two paths can answer differently.
 * The constant names are the interface names, which is what makes them usable in a diagnostic. A vector half
 * is one sitting below the matrix routines, as the vector-vector kernels do.
 */
@Suppress("LongParameterList") // one descriptor per half, deliberately in one table rather than several
internal enum class BackendSlot(
    /** The public role this half fills, one to one. */
    internal val role: BackendRole,
    /** Whether a backend implements this half, which is the type test only the compiler can write. */
    internal val accepts: (Backend) -> Boolean,
    /** Reads this half out of a resolved context. */
    internal val from: (F64Context) -> Backend,
    /** koblas's own implementation, for a selection that names nothing for this half. */
    internal val portableDefault: () -> Backend,
    /** The system property and environment variable a deployment pins this half with. */
    internal val selectionKeys: BackendSelectionKeys,
    /** Roles whose providers are specialized enough that this half is not theirs to fill. */
    internal val supersededBy: Set<BackendRole> = emptySet(),
    internal val vectorHalf: Boolean = false,
    internal val sparse: Boolean = false,
    internal val required: Boolean = true,
) {
    /** Dense vector-vector routines. */
    F64Kernels(
        role = BackendRole.DENSE_KERNELS,
        accepts = { it is F64Kernels },
        from = { it.kernels },
        portableDefault = { F64RoutedKernels(null) },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.dense.kernels",
            "KOBLAS_DENSE_KERNELS_BACKEND",
        ),
        vectorHalf = true,
    ),

    /** Dense matrix routines. */
    F64Blas(
        role = BackendRole.DENSE_BLAS,
        accepts = { it is F64Blas },
        from = { it.blas },
        portableDefault = { F64ReferenceLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.dense.blas",
            "KOBLAS_DENSE_BLAS_BACKEND",
        ),
    ),

    /** Dense factorizations. */
    F64Decompositions(
        role = BackendRole.DENSE_DECOMPOSITIONS,
        accepts = { it is F64Decompositions },
        from = { it.decompositions },
        portableDefault = { F64ReferenceLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.dense.decompositions",
            "KOBLAS_DENSE_DECOMPOSITIONS_BACKEND",
        ),
    ),

    /** Sparse vector-vector routines. */
    F64SparseKernels(
        role = BackendRole.SPARSE_KERNELS,
        accepts = { it is F64SparseKernels },
        from = { it.sparseKernels },
        portableDefault = { F64PlatformSparseKernels },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.kernels",
            "KOBLAS_SPARSE_KERNELS_BACKEND",
        ),
        vectorHalf = true,
        sparse = true,
    ),

    /** Sparse matrix routines. */
    F64SparseBlas(
        role = BackendRole.SPARSE_BLAS,
        accepts = { it is F64SparseBlas },
        from = { it.sparseBlas },
        portableDefault = { F64ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.blas",
            "KOBLAS_SPARSE_BLAS_BACKEND",
        ),
        sparse = true,
    ),

    /**
     * General pivoting sparse LU.
     *
     * A provider that also refactors one pattern or factorizes simplex bases is specialized, and what it
     * offers for ordinary LU is that specialization's own factorization rather than a general one. So an
     * offer of everything such a provider implements leaves this half to a general provider; a caller who
     * wants it here anyway names this role for it.
     */
    F64GeneralSparseLu(
        role = BackendRole.SPARSE_GENERAL_LU,
        accepts = { it is F64GeneralSparseLu },
        from = { it.generalSparseLu },
        portableDefault = { F64ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.general.lu",
            "KOBLAS_SPARSE_GENERAL_LU_BACKEND",
        ),
        supersededBy = setOf(BackendRole.SPARSE_REPEATED_LU, BackendRole.BASIS_FACTORIZATIONS),
        sparse = true,
    ),

    /** Repeated-pattern sparse LU. */
    F64RepeatedSparseLu(
        role = BackendRole.SPARSE_REPEATED_LU,
        accepts = { it is F64RepeatedSparseLu },
        from = { it.repeatedSparseLu ?: MissingRepeatedSparseLu },
        portableDefault = { MissingRepeatedSparseLu },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.repeated.lu",
            "KOBLAS_SPARSE_REPEATED_LU_BACKEND",
        ),
        sparse = true,
        required = false,
    ),

    /** Sparse Cholesky. */
    F64SparseCholesky(
        role = BackendRole.SPARSE_CHOLESKY,
        accepts = { it is F64SparseCholesky },
        from = { it.sparseCholesky },
        portableDefault = { F64ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.cholesky",
            "KOBLAS_SPARSE_CHOLESKY_BACKEND",
        ),
        sparse = true,
    ),

    /** Sparse quasi-definite LDL. */
    F64QuasiDefiniteLdl(
        role = BackendRole.SPARSE_QUASI_DEFINITE_LDL,
        accepts = { it is F64QuasiDefiniteLdl },
        from = { it.quasiDefiniteLdl },
        portableDefault = { F64ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.quasi-definite-ldl",
            "KOBLAS_SPARSE_QUASI_DEFINITE_LDL_BACKEND",
        ),
        sparse = true,
    ),

    /** Sparse QR. */
    F64SparseQr(
        role = BackendRole.SPARSE_QR,
        accepts = { it is F64SparseQr },
        from = { it.sparseQr },
        portableDefault = { F64ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.qr",
            "KOBLAS_SPARSE_QR_BACKEND",
        ),
        sparse = true,
    ),

    /** Simplex basis factorizations. */
    F64BasisFactorizations(
        role = BackendRole.BASIS_FACTORIZATIONS,
        accepts = { it is F64BasisFactorizations },
        from = { it.basisFactorizations },
        portableDefault = { F64ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.basis.factorizations",
            "KOBLAS_BASIS_FACTORIZATIONS_BACKEND",
        ),
        sparse = true,
    ),

    /** Simplex basis solvers. */
    F64BasisSolvers(
        role = BackendRole.BASIS_SOLVERS,
        accepts = { it is F64BasisSolvers },
        from = { it.basisSolvers },
        portableDefault = { F64ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.basis.solvers",
            "KOBLAS_BASIS_SOLVERS_BACKEND",
        ),
        sparse = true,
    ),
    ;

    /**
     * Whether this half takes [backend] from an offer of every half it implements, which is both selection
     * paths' default. False for a half [supersededBy] something [backend] also implements, so a specialized
     * provider keeps its own halves and leaves the general one alone.
     */
    internal fun acceptsOffer(backend: Backend): Boolean =
        accepts(backend) && supersededBy.none { it.slot.accepts(backend) }

    internal companion object {
        private val byRole: Map<BackendRole, BackendSlot> = entries.associateBy { it.role }

        init {
            check(byRole.size == BackendRole.entries.size) {
                "every BackendRole needs a half: ${BackendRole.entries - byRole.keys} have none"
            }
        }

        /** The half filling [role]. */
        fun of(role: BackendRole): BackendSlot = byRole.getValue(role)

        /** The halves a registry has seams for, as a diagnostic listing them the way a sentence would. */
        val names: String = entries.dropLast(1).joinToString(", ") { it.name } + " or " + entries.last().name

        /** The halves that do the matrix work, the ones a context is named after. */
        val contextHalves: List<BackendSlot> = entries.filter { it.required }
        val matrixHalves: List<BackendSlot> = contextHalves.filterNot { it.vectorHalf }
        val sparseHalves: Set<BackendSlot> = entries.filterTo(mutableSetOf()) { it.sparse }
    }
}

/** The half filling [BackendRole], which is the same thing under the name the registry keys on. */
internal val BackendRole.slot: BackendSlot get() = BackendSlot.of(this)
