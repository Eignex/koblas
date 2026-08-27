package com.eignex.koblas

import com.eignex.koblas.dense.F64RoutedKernels
import com.eignex.koblas.internal.backend.BackendSlot

/** A public role in an [F64Context], independent of the registry's internal seam representation. */
public enum class BackendRole {
    /** Dense vector-vector kernels. */
    DENSE_KERNELS,

    /** Dense matrix operations. */
    DENSE_BLAS,

    /** Dense factorizations and their solves. */
    DENSE_DECOMPOSITIONS,

    /** Sparse vector-vector kernels. */
    SPARSE_KERNELS,

    /** Sparse matrix operations. */
    SPARSE_BLAS,

    /** Sparse factorizations and their solves. */
    SPARSE_DECOMPOSITIONS,

    /** General pivoting sparse LU for unrelated patterns. */
    SPARSE_GENERAL_LU,

    /** Sparse LU that reuses one pattern's symbolic work. */
    SPARSE_REPEATED_LU,

    /** Symmetric positive-definite sparse Cholesky. */
    SPARSE_CHOLESKY,

    /** Symmetric sparse `L * D * L^T`. */
    SPARSE_LDL,

    /** Sparse QR for least-squares solves. */
    SPARSE_QR,

    /** Simplex basis factorizations with column replacement. */
    BASIS_FACTORIZATIONS,

    /** Simplex basis solvers. */
    BASIS_SOLVERS,
}

/**
 * Structured status for the backend selected for one [role].
 *
 * @property role the context role this backend fills.
 * @property provider the backend's diagnostic name.
 * @property priority the preference used when this backend was selected.
 * @property available whether the provider reports that it can run on this host.
 * @property portable whether this is koblas's own implementation.
 * @property accelerated whether this role contains an external provider.
 * @property metadata optional structured runtime facts reported by the provider.
 */
public data class BackendStatus(
    val role: BackendRole,
    val provider: String,
    val priority: Int,
    val available: Boolean,
    val portable: Boolean,
    val accelerated: Boolean,
    val metadata: BackendMetadata,
)

/**
 * A stable, structured snapshot of every backend selected in an [F64Context].
 *
 * @property backends one entry for every [BackendRole], in declaration order.
 */
public data class F64ContextStatus(val backends: List<BackendStatus>) {
    init {
        require(
            backends.size == BackendRole.entries.size &&
                backends.map { it.role }.toSet() == BackendRole.entries.toSet(),
        ) {
            "context status must contain every backend role exactly once"
        }
    }

    /** The selected backend status for [role]. */
    public operator fun get(role: BackendRole): BackendStatus = backends.first { it.role == role }
}

/** The backend installed for [role]. */
public fun F64Context.backendFor(role: BackendRole): Backend = when (role) {
    BackendRole.DENSE_KERNELS -> kernels
    BackendRole.DENSE_BLAS -> blas
    BackendRole.DENSE_DECOMPOSITIONS -> decompositions
    BackendRole.SPARSE_KERNELS -> sparseKernels
    BackendRole.SPARSE_BLAS -> sparseBlas
    BackendRole.SPARSE_DECOMPOSITIONS -> sparseDecompositions
    BackendRole.SPARSE_GENERAL_LU -> generalSparseLu
    BackendRole.SPARSE_REPEATED_LU -> repeatedSparseLu ?: MissingRepeatedSparseLu
    BackendRole.SPARSE_CHOLESKY -> sparseCholesky
    BackendRole.SPARSE_LDL -> sparseLdl
    BackendRole.SPARSE_QR -> sparseQr
    BackendRole.BASIS_FACTORIZATIONS -> basisFactorizations
    BackendRole.BASIS_SOLVERS -> basisSolvers
}

/** Whether [role] is filled by something other than koblas's own portable implementation. */
public fun F64Context.isAccelerated(role: BackendRole): Boolean = when (val backend = backendFor(role)) {
    is F64RoutedKernels -> backend.host != null
    else -> !backend.isPortable
}

/** A structured snapshot of every selected backend half. */
public val F64Context.status: F64ContextStatus
    get() = F64ContextStatus(
        BackendRole.entries.map { role ->
            val backend = backendFor(role)
            BackendStatus(
                role = role,
                provider = backend.name,
                priority = backend.priority,
                available = backend.isAvailable,
                portable = backend.isPortable,
                accelerated = isAccelerated(role),
                metadata = (backend as? BackendMetadataProvider)?.backendMetadata ?: BackendMetadata(),
            )
        },
    )

/** The roles still running koblas's own portable implementation, in declaration order. */
public val F64Context.portableRoles: Set<BackendRole>
    get() = BackendRole.entries.filterNot { isAccelerated(it) }.toSet()

/** Throws unless [role] and every one of [otherRoles] is filled by an accelerated backend. */
public fun F64Context.requireAccelerated(role: BackendRole, vararg otherRoles: BackendRole) {
    val roles = listOf(role, *otherRoles)
    val fallen = roles.filterNot { isAccelerated(it) }
    check(fallen.isEmpty()) {
        val detail = fallen.joinToString(", ") { "$it=${backendFor(it).name}" }
        accelerationFailure(detail)
    }
}

/** The backend installed in [slot]. */
public fun F64Context.backendFor(slot: BackendSlot): Backend = slot.from(this)

/** Whether [slot] is filled by something other than koblas's own portable implementation. */
public fun F64Context.isAccelerated(slot: BackendSlot): Boolean = when (val backend = backendFor(slot)) {
    is F64RoutedKernels -> backend.host != null
    else -> !backend.isPortable
}

/** The slots still running koblas's own portable implementation, in declaration order. */
public val F64Context.portableSlots: Set<BackendSlot>
    get() = BackendSlot.entries.filterNot { isAccelerated(it) }.toSet()

/** Throws unless every one of [slots] is filled by an accelerated backend. */
public fun F64Context.requireAccelerated(vararg slots: BackendSlot) {
    val fallen = slots.filterNot { isAccelerated(it) }
    check(fallen.isEmpty()) {
        val detail = fallen.joinToString(", ") { "$it=${backendFor(it).name}" }
        accelerationFailure(detail)
    }
}

private fun F64Context.accelerationFailure(detail: String): String =
    "koblas fell back to portable implementations for: $detail. " +
        "Either the host library is missing (libopenblas/liblapacke on Linux, brew install openblas on " +
        "macOS), the backend artifact is not on the classpath, or nothing has been registered for that " +
        "role yet. Resolved: backend=$name, kernels=${kernels.name}"
