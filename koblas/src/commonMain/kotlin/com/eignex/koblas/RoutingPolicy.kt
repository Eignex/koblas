package com.eignex.koblas

/** How an explicit [F64Context] constrains operation-level dispatch. */
public enum class F64DispatchPolicy {
    /** Use the selected provider's measured routing behavior. */
    AUTO,

    /** Reject an inspected operation unless it is known to execute outside the portable implementation. */
    NATIVE_ONLY,

    /** Resolve every role to koblas's portable implementation. */
    PORTABLE_ONLY,
}

/** What an automatic context does when an inspected operation is not known to execute natively. */
public enum class F64FallbackPolicy {
    /** Execute without reporting the fallback. */
    ALLOW,

    /** Report the route to the context's warning handler, then execute. */
    WARN,

    /** Reject the operation before invoking its backend. */
    THROW,
}

/** The action an explicit context will take for an inspected [route]. */
public enum class BackendPolicyDecision {
    /** Execute the operation. */
    EXECUTE,

    /** Report the route and then execute the operation. */
    WARN,

    /** Reject the operation before its backend is invoked. */
    REJECT,
}

/**
 * A route together with the action imposed by its context policy.
 *
 * @property route the provider's operation-level prediction.
 * @property decision the action the context will take before dispatch.
 */
public data class F64RoutePlan(val route: BackendRoute, val decision: BackendPolicyDecision)

/** Raised before dispatch when an explicit context rejects [route]. */
public class BackendRouteRejectedException(public val route: BackendRoute) :
    IllegalStateException(
        "${route.query} would execute as ${route.execution} through ${route.executor}: ${route.reason}",
    )

/** Applies this context's dispatch and fallback policy to [query] without executing it. */
public fun F64Context.plan(query: F64RouteQuery): F64RoutePlan {
    val route = route(query)
    val decision = when (dispatchPolicy) {
        F64DispatchPolicy.NATIVE_ONLY -> if (route.execution == BackendExecution.NATIVE) {
            BackendPolicyDecision.EXECUTE
        } else {
            BackendPolicyDecision.REJECT
        }

        F64DispatchPolicy.PORTABLE_ONLY -> if (route.execution == BackendExecution.PORTABLE) {
            BackendPolicyDecision.EXECUTE
        } else {
            BackendPolicyDecision.REJECT
        }

        F64DispatchPolicy.AUTO -> when {
            route.execution == BackendExecution.NATIVE -> BackendPolicyDecision.EXECUTE
            fallbackPolicy == F64FallbackPolicy.ALLOW -> BackendPolicyDecision.EXECUTE
            fallbackPolicy == F64FallbackPolicy.WARN -> BackendPolicyDecision.WARN
            else -> BackendPolicyDecision.REJECT
        }
    }
    return F64RoutePlan(route, decision)
}

internal val F64Context.enforcesRoutingPolicy: Boolean
    get() = dispatchPolicy != F64DispatchPolicy.AUTO || fallbackPolicy != F64FallbackPolicy.ALLOW

internal fun F64Context.beforeDispatch(query: F64RouteQuery) {
    val plan = plan(query)
    when (plan.decision) {
        BackendPolicyDecision.EXECUTE -> Unit
        BackendPolicyDecision.WARN -> fallbackWarning(plan.route)
        BackendPolicyDecision.REJECT -> throw BackendRouteRejectedException(plan.route)
    }
}
