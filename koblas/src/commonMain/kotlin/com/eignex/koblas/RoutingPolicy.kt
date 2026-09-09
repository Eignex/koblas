package com.eignex.koblas

/** How an explicit [KoblasContext] constrains operation-level dispatch. */
public enum class DispatchPolicy {
    /** Use the selected provider's measured routing behavior. */
    AUTO,

    /** Reject an inspected operation unless it is known to execute outside the portable implementation. */
    NATIVE_ONLY,

    /** Resolve every role to koblas's portable implementation. */
    PORTABLE_ONLY,
}

/** What an automatic context does when an inspected operation is not known to execute natively. */
public enum class FallbackPolicy {
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
public data class RoutePlan(val route: BackendRoute, val decision: BackendPolicyDecision)

/** Raised before dispatch when an explicit context rejects [route]. */
public class BackendRouteRejectedException(public val route: BackendRoute) :
    IllegalStateException(
        "${route.query} would execute as ${route.execution} through ${route.executor}: ${route.reason}",
    )

/** Applies this context's dispatch and fallback policy to [query] without executing it. */
public fun KoblasContext.plan(query: RouteQuery): RoutePlan {
    val route = route(query)
    val decision = when (dispatchPolicy) {
        DispatchPolicy.NATIVE_ONLY -> if (route.execution == BackendExecution.NATIVE) {
            BackendPolicyDecision.EXECUTE
        } else {
            BackendPolicyDecision.REJECT
        }

        DispatchPolicy.PORTABLE_ONLY -> if (route.execution == BackendExecution.PORTABLE) {
            BackendPolicyDecision.EXECUTE
        } else {
            BackendPolicyDecision.REJECT
        }

        DispatchPolicy.AUTO -> when {
            route.execution == BackendExecution.NATIVE -> BackendPolicyDecision.EXECUTE
            fallbackPolicy == FallbackPolicy.ALLOW -> BackendPolicyDecision.EXECUTE
            fallbackPolicy == FallbackPolicy.WARN -> BackendPolicyDecision.WARN
            else -> BackendPolicyDecision.REJECT
        }
    }
    return RoutePlan(route, decision)
}

internal val KoblasContext.enforcesRoutingPolicy: Boolean
    get() = dispatchPolicy != DispatchPolicy.AUTO || fallbackPolicy != FallbackPolicy.ALLOW

internal fun KoblasContext.beforeDispatch(query: RouteQuery) {
    val plan = plan(query)
    when (plan.decision) {
        BackendPolicyDecision.EXECUTE -> Unit
        BackendPolicyDecision.WARN -> fallbackWarning(plan.route)
        BackendPolicyDecision.REJECT -> throw BackendRouteRejectedException(plan.route)
    }
}
