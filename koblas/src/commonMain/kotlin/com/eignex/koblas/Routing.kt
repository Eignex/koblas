package com.eignex.koblas

/** An operation whose runtime route can be inspected before it is executed. */
public sealed interface F64RouteQuery {
    /** The context role that owns this operation. */
    public val role: BackendRole

    /** A dense matrix-vector product over a matrix of [rows] by [cols]. */
    public data class DenseGemv(val rows: Int, val cols: Int) : F64RouteQuery {
        override val role: BackendRole get() = BackendRole.DENSE_BLAS

        init {
            requireNonNegative(rows, "rows")
            requireNonNegative(cols, "cols")
        }
    }

    /** A dense product with result shape `[m] x [n]` and inner dimension [k]. */
    public data class DenseGemm(val m: Int, val n: Int, val k: Int) : F64RouteQuery {
        override val role: BackendRole get() = BackendRole.DENSE_BLAS

        init {
            requireNonNegative(m, "m")
            requireNonNegative(n, "n")
            requireNonNegative(k, "k")
        }
    }

    /** A dense LU factorization with [order] DGETRF pivot steps. */
    public data class DenseLu(val order: Int) : F64RouteQuery {
        override val role: BackendRole get() = BackendRole.DENSE_DECOMPOSITIONS

        init {
            requireNonNegative(order, "order")
        }
    }

    /**
     * A sparse-times-dense product whose sparse operand has [storedEntries]. [right] puts it on the right;
     * [transposeDense] transposes the dense operand.
     */
    public data class SparseDenseGemm(
        val storedEntries: Int,
        val right: Boolean = false,
        val transposeDense: Boolean = false,
    ) : F64RouteQuery {
        override val role: BackendRole get() = BackendRole.SPARSE_BLAS

        init {
            requireNonNegative(storedEntries, "storedEntries")
        }
    }

    /**
     * A product over a caller-held prepared sparse descriptor.
     *
     * @property storedEntries entries retained by the prepared descriptor.
     * @property kind the product family whose amortized gate is inspected.
     */
    public data class PreparedSparseProduct(val storedEntries: Int, val kind: PreparedSparseProductKind) :
        F64RouteQuery {
        override val role: BackendRole get() = BackendRole.SPARSE_BLAS

        init {
            requireNonNegative(storedEntries, "storedEntries")
        }
    }

    /**
     * An in-place operation against a caller-supplied sparse triangle, either eliminating it ([kind] of
     * [SparseTriangularKind.SOLVE]) or applying it directly ([kind] of [SparseTriangularKind.MULTIPLY]).
     *
     * @property storedEntries stored entries in the triangular matrix.
     * @property kind whether the triangle is solved against or multiplied by.
     * @property rightHandSides independent vectors processed in this call.
     * @property lower whether the lower triangle is used rather than the upper triangle.
     * @property right whether the triangle is applied from the right.
     * @property transpose whether the triangle is transposed.
     * @property unitDiagonal whether the stored diagonal is ignored and treated as one.
     */
    public data class SparseTriangular(
        val storedEntries: Int,
        val kind: SparseTriangularKind,
        val rightHandSides: Int = 1,
        val lower: Boolean = true,
        val right: Boolean = false,
        val transpose: Boolean = false,
        val unitDiagonal: Boolean = false,
    ) : F64RouteQuery {
        override val role: BackendRole get() = BackendRole.SPARSE_BLAS

        init {
            requireNonNegative(storedEntries, "storedEntries")
            requireNonNegative(rightHandSides, "rightHandSides")
        }
    }

    /** A sparse QR factorization of a matrix with [storedEntries] stored entries. */
    public data class SparseQr(val storedEntries: Int) : F64RouteQuery {
        override val role: BackendRole get() = BackendRole.SPARSE_QR

        init {
            requireNonNegative(storedEntries, "storedEntries")
        }
    }

    /** A general sparse LU factorization of a matrix with [storedEntries] stored entries. */
    public data class SparseLu(val storedEntries: Int) : F64RouteQuery {
        override val role: BackendRole get() = BackendRole.SPARSE_GENERAL_LU

        init {
            requireNonNegative(storedEntries, "storedEntries")
        }
    }
}

/** Operation families whose amortized prepared-descriptor crossovers are measured independently. */
public enum class PreparedSparseProductKind {
    /** One dense right-hand side. */
    GEMV,

    /** Several dense right-hand sides in one call. */
    DENSE_GEMM,

    /** A second sparse operand and a sparse result. */
    SPARSE_GEMM,
}

/** Whether an [F64RouteQuery.SparseTriangular] eliminates the triangle or applies it directly. */
public enum class SparseTriangularKind {
    /** The triangle is solved against, as in `trsv`/`trsm`. */
    SOLVE,

    /** The triangle is multiplied by, as in `trmv`/`trmm`. */
    MULTIPLY,
}

/** Where an inspected operation will execute. */
public enum class BackendExecution {
    /** A call through a native or otherwise external provider. */
    NATIVE,

    /** Koblas's own implementation. */
    PORTABLE,

    /** The selected binding cannot execute on this host and has no automatic fallback for this call. */
    UNAVAILABLE,

    /** The selected third-party backend does not expose operation-level routing information. */
    UNKNOWN,
}

/** Why an inspected operation takes its reported [BackendExecution]. */
public enum class BackendRouteReason {
    /** The selected backend is portable. */
    SELECTED_PORTABLE,

    /** The selected binding is unavailable; [BackendExecution] says whether a fallback exists. */
    BACKEND_UNAVAILABLE,

    /** The problem is below a third-party backend's native crossover. */
    BELOW_THRESHOLD,

    /** This argument form is deliberately handled by the portable implementation. */
    UNSUPPORTED_ARGUMENTS,

    /** The selected provider has no native implementation of this operation. */
    UNSUPPORTED_OPERATION,

    /** The selected backend will execute this operation natively. */
    NATIVE_ROUTE,

    /** The selected backend does not report a route for this operation. */
    NOT_REPORTED,
}

/** The quantity a backend may compare with a dispatch threshold. */
public enum class DispatchMetric {
    /** A matrix dimension. */
    DIMENSION,

    /** The `m * n * k` work estimate of a level-3 operation. */
    LEVEL3_WORK,

    /** Stored entries in a sparse matrix. */
    STORED_ENTRIES,
}

/**
 * A backend-specific gate reported by operation-level routing diagnostics.
 *
 * @property metric the quantity the backend compares.
 * @property actual the value for the inspected problem shape.
 * @property minimum the smallest value the backend routes externally.
 */
public data class DispatchGate(val metric: DispatchMetric, val actual: Long, val minimum: Long)

/**
 * A structured prediction of where [query] will execute under the selected context.
 *
 * @property query the inspected operation and problem shape.
 * @property selected the provider selected for the operation's semantic role.
 * @property execution where the operation is predicted to run.
 * @property executor the provider expected to execute the operation.
 * @property reason why this route was selected.
 * @property gate a backend-specific threshold comparison, when one is reported.
 */
public data class BackendRoute(
    val query: F64RouteQuery,
    val selected: BackendStatus,
    val execution: BackendExecution,
    val executor: String,
    val reason: BackendRouteReason,
    val gate: DispatchGate? = null,
) {
    /** Whether this operation is predicted to leave koblas's portable implementation. */
    public val accelerated: Boolean get() = execution == BackendExecution.NATIVE
}

/** Optional operation-level routing diagnostics implemented by backends that make per-call decisions. */
public interface F64RoutingBackend : Backend {
    /** The route for [query], or null when this backend does not report that operation. */
    public fun route(query: F64RouteQuery): BackendRoute?
}

/** Predicts where [query] will execute without performing the operation. */
public fun F64Context.route(query: F64RouteQuery): BackendRoute {
    val backend = backendFor(query.role)
    val selected = status[query.role]
    if (backend.isPortable) {
        return BackendRoute(
            query,
            selected,
            BackendExecution.PORTABLE,
            backend.name,
            BackendRouteReason.SELECTED_PORTABLE,
        )
    }
    val reported = (backend as? F64RoutingBackend)?.route(query)
    return reported?.copy(selected = selected) ?: BackendRoute(
        query,
        selected,
        BackendExecution.UNKNOWN,
        backend.name,
        BackendRouteReason.NOT_REPORTED,
    )
}

/** What [backend] reports for [query]'s half, as every route records it. */
private fun selectedStatus(query: F64RouteQuery, backend: Backend): BackendStatus = BackendStatus(
    query.role,
    backend.name,
    backend.priority,
    backend.isAvailable,
    backend.isPortable,
    accelerated = !backend.isPortable,
    (backend as? BackendMetadataProvider)?.backendMetadata ?: BackendMetadata(),
)

/** Builds the native route or unavailable fallback shared by the host adapters. */
internal fun nativeRoute(
    query: F64RouteQuery,
    backend: Backend,
    portableExecutor: String = "reference",
    fallbackWhenUnavailable: Boolean = true,
): BackendRoute {
    val selected = selectedStatus(query, backend)
    return when {
        !backend.isAvailable && fallbackWhenUnavailable -> BackendRoute(
            query,
            selected,
            BackendExecution.PORTABLE,
            portableExecutor,
            BackendRouteReason.BACKEND_UNAVAILABLE,
        )

        !backend.isAvailable -> BackendRoute(
            query,
            selected,
            BackendExecution.UNAVAILABLE,
            backend.name,
            BackendRouteReason.BACKEND_UNAVAILABLE,
        )

        else -> BackendRoute(
            query,
            selected,
            BackendExecution.NATIVE,
            backend.name,
            BackendRouteReason.NATIVE_ROUTE,
        )
    }
}

/** Builds a reported portable route that does not depend on a threshold. */
internal fun portableRoute(
    query: F64RouteQuery,
    backend: Backend,
    portableExecutor: String,
    reason: BackendRouteReason,
): BackendRoute = BackendRoute(
    query,
    selectedStatus(query, backend),
    BackendExecution.PORTABLE,
    portableExecutor,
    reason,
)

/** Saturates a displayed work estimate instead of overflowing it. */
internal fun saturatedProduct(a: Int, b: Int, c: Int): Long {
    if (a == 0 || b == 0 || c == 0) return 0
    val ab = a.toLong() * b
    if (ab > Long.MAX_VALUE / c) return Long.MAX_VALUE
    return ab * c
}

private fun requireNonNegative(value: Int, name: String) {
    require(value >= 0) { "$name must not be negative" }
}
