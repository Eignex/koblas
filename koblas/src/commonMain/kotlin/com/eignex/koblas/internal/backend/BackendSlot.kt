package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.BackendRole
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Kernels
import com.eignex.koblas.dense.PlatformKernels
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.sparse.GeneralSparseLu
import com.eignex.koblas.sparse.PlatformSparseKernels
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.basis.BasisSolvers

/**
 * The halves of the seam a backend can implement, and everything either selection path needs to know about
 * one: which interface fills it, how to read it out of a context, what koblas's own implementation is, the
 * keys a deployment pins it with, and whether a specialized provider should leave it alone.
 *
 * One entry per half rather than one table per question. The registry and [com.eignex.koblas.ContextBuilder]
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
    internal val from: (KoblasContext) -> Backend,
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
    Kernels(
        role = BackendRole.DENSE_KERNELS,
        accepts = { it is Kernels },
        from = { it.kernels },
        portableDefault = { PlatformKernels },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.dense.kernels",
            "KOBLAS_DENSE_KERNELS_BACKEND",
        ),
        vectorHalf = true,
    ),

    /** Dense matrix routines. */
    Blas(
        role = BackendRole.DENSE_BLAS,
        accepts = { it is Blas },
        from = { it.blas },
        portableDefault = { ReferenceBlas },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.dense.blas",
            "KOBLAS_DENSE_BLAS_BACKEND",
        ),
    ),

    /** Sparse vector-vector routines. */
    SparseKernels(
        role = BackendRole.SPARSE_KERNELS,
        accepts = { it is SparseKernels },
        from = { it.sparseKernels },
        portableDefault = { PlatformSparseKernels },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.kernels",
            "KOBLAS_SPARSE_KERNELS_BACKEND",
        ),
        vectorHalf = true,
        sparse = true,
    ),

    /** Sparse matrix routines. */
    SparseBlas(
        role = BackendRole.SPARSE_BLAS,
        accepts = { it is SparseBlas },
        from = { it.sparseBlas },
        portableDefault = { ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.blas",
            "KOBLAS_SPARSE_BLAS_BACKEND",
        ),
        sparse = true,
    ),

    /**
     * General pivoting sparse LU.
     *
     * A host library that also refactors one pattern or factorizes simplex bases is specialized, and what it
     * offers for ordinary LU is that specialization's own factorization rather than a general one. So an
     * offer of everything such a library implements leaves this half to a general provider; a caller who
     * wants it here anyway names this role for it. [acceptsOffer] says which providers that covers.
     */
    GeneralSparseLu(
        role = BackendRole.SPARSE_GENERAL_LU,
        accepts = { it is GeneralSparseLu },
        from = { it.generalSparseLu },
        portableDefault = { ReferenceSparseLinearAlgebra },
        selectionKeys = BackendSelectionKeys(
            "koblas.backend.sparse.general.lu",
            "KOBLAS_SPARSE_GENERAL_LU_BACKEND",
        ),
        sparse = true,
    ),

    /** Simplex basis solvers. */
    BasisSolvers(
        role = BackendRole.BASIS_SOLVERS,
        accepts = { it is BasisSolvers },
        from = { it.basisSolvers },
        portableDefault = { ReferenceSparseLinearAlgebra },
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
     * host library keeps its own halves and leaves the general one alone.
     *
     * koblas's own implementations are exempt, since [Backend.isPortable] is what tells a binding to a host
     * library from an implementation written here. The portable reference fills the basis-factorization half
     * beside a general LU that really is general, and dropping it from the general half would hand the role
     * to a second copy of itself.
     */
    internal fun acceptsOffer(backend: Backend): Boolean = accepts(backend) &&
        (backend.isPortable || supersededBy.none { it.slot.accepts(backend) })

    internal companion object {
        private val byRole: Map<BackendRole, BackendSlot> = entries.associateBy { it.role }

        init {
            check(byRole.keys == BackendRole.entries.toSet()) {
                "every BackendRole needs a half: ${BackendRole.entries - byRole.keys} have none"
            }
            // Two halves claiming one role leave the loser unreachable through [slot], and the coverage
            // check above cannot see it because the winner still answers for the role.
            check(byRole.size == entries.size) {
                "every BackendRole needs only one half: ${entries.filterNot { byRole[it.role] === it }} " +
                    "share a role with an earlier half"
            }
        }

        /** The half filling [role]. */
        fun of(role: BackendRole): BackendSlot = byRole.getValue(role)

        /** The halves a registry has seams for, as a diagnostic listing them the way a sentence would. */
        val names: String = entries.dropLast(1).joinToString(", ") { it.name } + " or " + entries.last().name

        /** The halves a context always has one of, so the optional ones cannot speak for a whole context. */
        val contextHalves: List<BackendSlot> = entries.filter { it.required }

        /** The halves that do the matrix work, the ones a context is named after. */
        val matrixHalves: List<BackendSlot> = contextHalves.filterNot { it.vectorHalf }
        val sparseHalves: Set<BackendSlot> = entries.filterTo(mutableSetOf()) { it.sparse }
    }
}

/** The half filling [BackendRole], which is the same thing under the name the registry keys on. */
internal val BackendRole.slot: BackendSlot get() = BackendSlot.of(this)
