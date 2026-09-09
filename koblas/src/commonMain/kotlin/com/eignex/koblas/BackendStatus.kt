package com.eignex.koblas

import com.eignex.koblas.internal.backend.slot

/** A public role in an [KoblasContext], independent of the registry's internal seam representation. */
public enum class BackendRole {
    /** Dense vector-vector kernels. */
    DENSE_KERNELS,

    /** Dense matrix operations. */
    DENSE_BLAS,

    /** Sparse vector-vector kernels. */
    SPARSE_KERNELS,

    /** Sparse matrix operations. */
    SPARSE_BLAS,

    /** General pivoting sparse LU for unrelated patterns. */
    SPARSE_GENERAL_LU,

    /** Sparse LU that reuses one pattern's symbolic work. */
    SPARSE_REPEATED_LU,

    /** Symmetric positive-definite sparse Cholesky. */
    SPARSE_CHOLESKY,

    /** Symmetric sparse quasi-definite, numerically unpivoted `L * D * L^T`. */
    SPARSE_QUASI_DEFINITE_LDL,

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
 * @property unavailableReason why the provider cannot run, when it says. Null when it is available or
 * reports no reason.
 */
public data class BackendStatus(
    val role: BackendRole,
    val provider: String,
    val priority: Int,
    val available: Boolean,
    val portable: Boolean,
    val accelerated: Boolean,
    val metadata: BackendMetadata,
    val unavailableReason: String? = null,
)

/**
 * A stable, structured snapshot of every backend selected in an [KoblasContext].
 *
 * @property backends one entry for every [BackendRole], in declaration order.
 */
public data class ContextStatus(val backends: List<BackendStatus>) {
    init {
        // Checked position by position rather than through two sets: the producers all build this in
        // declaration order, and comparing sets allocated a list and two of them on a type diagnostics read
        // in a loop. Holding the order is also what lets [get] index instead of scan.
        require(backends.size == BackendRole.entries.size) {
            "context status must contain every backend role exactly once"
        }
        for (i in backends.indices) {
            require(backends[i].role == BackendRole.entries[i]) {
                "context status must list the roles in declaration order"
            }
        }
    }

    /** The selected backend status for [role]. */
    public operator fun get(role: BackendRole): BackendStatus = backends[role.ordinal]
}

/** The backend installed for [role]. */
public fun KoblasContext.backendFor(role: BackendRole): Backend = role.slot.from(this)

/** Whether [role] is filled by something other than koblas's own portable implementation. */
public fun KoblasContext.isAccelerated(role: BackendRole): Boolean = !backendFor(role).isPortable

/** Shared by every half that reports no metadata of its own, so reading a status allocates none. */
internal val NO_METADATA: BackendMetadata = BackendMetadata()

/**
 * The selected backend for one [role].
 *
 * Reading a single role off [status] would build every role, which is what a routed dispatch used to do on
 * every operation it inspected.
 */
internal fun KoblasContext.statusFor(role: BackendRole): BackendStatus =
    backendStatus(role, backendFor(role), accelerated = isAccelerated(role))

/**
 * A status for [role] read off [backend].
 *
 * [accelerated] is the one field a backend cannot answer for itself: whether the half it fills counts as
 * accelerated is a fact about the context that selected it. A caller holding a context passes
 * [KoblasContext.isAccelerated]; one reporting its own route has only the backend, and answers from whether it
 * is portable.
 */
internal fun backendStatus(role: BackendRole, backend: Backend, accelerated: Boolean): BackendStatus = BackendStatus(
    role = role,
    provider = backend.name,
    priority = backend.priority,
    available = backend.isAvailable,
    portable = backend.isPortable,
    accelerated = accelerated,
    metadata = (backend as? BackendMetadataProvider)?.backendMetadata ?: NO_METADATA,
    unavailableReason = backend.unavailableReason,
)

/** A structured snapshot of every selected backend half. */
public val KoblasContext.status: ContextStatus
    get() = ContextStatus(BackendRole.entries.map { statusFor(it) })

/** The roles still running koblas's own portable implementation, in declaration order. */
public val KoblasContext.portableRoles: Set<BackendRole>
    get() = BackendRole.entries.filterNot { isAccelerated(it) }.toSet()

/** Throws unless [role] and every one of [otherRoles] is filled by an accelerated backend. */
public fun KoblasContext.requireAccelerated(role: BackendRole, vararg otherRoles: BackendRole) {
    val roles = listOf(role, *otherRoles)
    val fallen = roles.filterNot { isAccelerated(it) }
    check(fallen.isEmpty()) {
        val detail = fallen.joinToString(", ") { "$it=${backendFor(it).name}" }
        accelerationFailure(detail)
    }
}

private fun KoblasContext.accelerationFailure(detail: String): String =
    "koblas fell back to portable implementations for: $detail. " +
        "Either the host library is missing (libopenblas on Linux, brew install openblas on " +
        "macOS), the backend artifact is not on the classpath, or nothing has been registered for that " +
        "role yet. Resolved: backend=$name, kernels=${kernels.name}"
