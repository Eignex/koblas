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

    /** A dense LU factorization of a square matrix of [order]. */
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

    /** A general sparse LU factorization of a matrix with [storedEntries] stored entries. */
    public data class SparseLu(val storedEntries: Int) : F64RouteQuery {
        override val role: BackendRole get() = BackendRole.SPARSE_DECOMPOSITIONS

        init {
            requireNonNegative(storedEntries, "storedEntries")
        }
    }
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

    /** The problem is below the binding's measured native crossover. */
    BELOW_THRESHOLD,

    /** This argument form is deliberately handled by the portable implementation. */
    UNSUPPORTED_ARGUMENTS,

    /** The selected backend will execute this operation natively. */
    NATIVE_ROUTE,

    /** The selected backend does not report a route for this operation. */
    NOT_REPORTED,
}

/** The quantity compared with a dispatch threshold. */
public enum class DispatchMetric {
    /** A matrix dimension. */
    DIMENSION,

    /** The `m * n * k` work estimate of a level-3 operation. */
    LEVEL3_WORK,

    /** Stored entries in a sparse matrix. */
    STORED_ENTRIES,
}

/**
 * The measured gate used to make a route decision.
 *
 * @property metric the quantity the backend compares.
 * @property actual the value for the inspected problem shape.
 * @property minimum the smallest value routed to the external provider.
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
 * @property gate the threshold comparison, when the route depends on one.
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

/** Builds the native or threshold fallback decision shared by the host adapters. */
internal fun thresholdRoute(
    query: F64RouteQuery,
    backend: Backend,
    portableExecutor: String,
    gate: DispatchGate,
    dispatches: Boolean,
    fallbackWhenUnavailable: Boolean = true,
): BackendRoute {
    val selected = BackendStatus(
        query.role,
        backend.name,
        backend.priority,
        backend.isAvailable,
        backend.isPortable,
        accelerated = !backend.isPortable,
        (backend as? BackendMetadataProvider)?.backendMetadata ?: BackendMetadata(),
    )
    return when {
        !dispatches -> BackendRoute(
            query,
            selected,
            BackendExecution.PORTABLE,
            portableExecutor,
            BackendRouteReason.BELOW_THRESHOLD,
            gate,
        )

        !backend.isAvailable && fallbackWhenUnavailable -> BackendRoute(
            query,
            selected,
            BackendExecution.PORTABLE,
            portableExecutor,
            BackendRouteReason.BACKEND_UNAVAILABLE,
            gate,
        )

        !backend.isAvailable -> BackendRoute(
            query,
            selected,
            BackendExecution.UNAVAILABLE,
            backend.name,
            BackendRouteReason.BACKEND_UNAVAILABLE,
            gate,
        )

        else -> BackendRoute(
            query,
            selected,
            BackendExecution.NATIVE,
            backend.name,
            BackendRouteReason.NATIVE_ROUTE,
            gate,
        )
    }
}

/** The saturated product used only for displaying a level-3 gate without overflowing. */
internal fun saturatedProduct(a: Int, b: Int, c: Int): Long {
    if (a == 0 || b == 0 || c == 0) return 0
    val ab = a.toLong() * b
    if (ab > Long.MAX_VALUE / c) return Long.MAX_VALUE
    return ab * c
}

private fun requireNonNegative(value: Int, name: String) {
    require(value >= 0) { "$name must not be negative" }
}
