package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.internal.backend.slot
import com.eignex.koblas.sparse.*

/**
 * An immutable resolver for an explicit [KoblasContext], initially containing only portable implementations.
 * Every `with` method returns a new builder, so configurations can be safely retained and shared.
 */
public class ContextBuilder private constructor(
    private val selections: Map<BackendRole, Backend>,
    /** The operation-level dispatch requirement of the resolved context. */
    public val dispatchPolicy: DispatchPolicy,
    /** The action for non-native inspected routes in automatic mode. */
    public val fallbackPolicy: FallbackPolicy,
    private val fallbackWarning: ((BackendRoute) -> Unit)?,
) {
    /** Creates a resolver seeded with koblas's portable implementations for every role. */
    public constructor() : this(
        portableSelections(),
        DispatchPolicy.AUTO,
        FallbackPolicy.ALLOW,
        null,
    )

    /** Returns a resolver seeded from the exact role selections and policies of [context]. */
    public constructor(context: KoblasContext) : this(
        BackendRole.entries.associateWith(context::backendFor),
        context.dispatchPolicy,
        context.fallbackPolicy,
        context.fallbackWarning,
    )

    /** Selects [backend] for [role], without consulting global registration or priority. */
    public fun withBackend(role: BackendRole, backend: Backend): ContextBuilder {
        require(role.slot.accepts(backend)) { "${backend.name} does not implement $role" }
        return copy(selections = selections + (role to backend))
    }

    /**
     * Selects [backend] for every role it implements, without consulting global registration or priority.
     * A specialized provider keeps the roles that specialization is about and leaves the general ones it
     * also implements to a general provider, which is what registration does with the same object. The role
     * overload selects it for a general role anyway.
     */
    public fun withBackend(backend: Backend): ContextBuilder {
        val roles = BackendSlot.entries.filter { it.acceptsOffer(backend) }.map { it.role }
        require(roles.isNotEmpty()) { "${backend.name} implements no  backend role" }
        return copy(selections = selections + roles.associateWith { backend })
    }

    /** Selects both level-1 halves from an exact built-in [provider]. */
    @ExperimentalKoblasApi
    public fun withBuiltinKernels(provider: BuiltinKernelProvider): ContextBuilder =
        withBackend(BackendRole.DENSE_KERNELS, provider.kernels)
            .withBackend(BackendRole.SPARSE_KERNELS, provider.sparseKernels)

    /** Returns a resolver using [policy] for operation-level dispatch. */
    public fun withDispatchPolicy(policy: DispatchPolicy): ContextBuilder = copy(dispatchPolicy = policy)

    /** Returns a resolver using [policy] for automatic fallbacks. */
    public fun withFallbackPolicy(policy: FallbackPolicy): ContextBuilder = copy(fallbackPolicy = policy)

    /** Returns a resolver that sends warning routes to [handler]. */
    public fun onFallback(handler: (BackendRoute) -> Unit): ContextBuilder = copy(fallbackWarning = handler)

    /** Resolves a new immutable context without reading or mutating the process-wide registry. */
    public fun resolve(): KoblasContext {
        require(fallbackPolicy != FallbackPolicy.WARN || fallbackWarning != null) {
            "WARN fallback policy requires an onFallback handler"
        }
        val resolved = if (dispatchPolicy == DispatchPolicy.PORTABLE_ONLY) portableSelections() else selections
        // Taken exactly as chosen, and `portable halves retain their contexts selected dense kernels`
        // pins that even a one-element operation reaches it.
        val kernels = resolved.getValue(BackendRole.DENSE_KERNELS) as Kernels
        val denseReference = ReferenceBackend(kernels)
        val sparseReference = ReferenceSparseBackend(kernels)
        return KoblasContext(
            kernels = kernels,
            blas = resolved.boundReference(BackendRole.DENSE_BLAS, denseReference) as Blas,
            sparseKernels = resolved.getValue(BackendRole.SPARSE_KERNELS) as SparseKernels,
            sparseBlas = resolved.boundReference(BackendRole.SPARSE_BLAS, sparseReference) as SparseBlas,
            dispatchPolicy = dispatchPolicy,
            fallbackPolicy = fallbackPolicy,
            fallbackWarning = fallbackWarning ?: {},
        )
    }

    @Suppress("LongParameterList") // mirrors the four immutable builder fields
    private fun copy(
        selections: Map<BackendRole, Backend> = this.selections,
        dispatchPolicy: DispatchPolicy = this.dispatchPolicy,
        fallbackPolicy: FallbackPolicy = this.fallbackPolicy,
        fallbackWarning: ((BackendRoute) -> Unit)? = this.fallbackWarning,
    ): ContextBuilder = ContextBuilder(
        selections.toMap(),
        dispatchPolicy,
        fallbackPolicy,
        fallbackWarning,
    )
}

private fun Map<BackendRole, Backend>.boundReference(role: BackendRole, configured: Backend): Backend {
    val selected = getValue(role)
    return if (selected is RebindableBackend && !selected.hasOwnKernels) configured else selected
}

private fun portableSelections(): Map<BackendRole, Backend> =
    BackendSlot.entries.associate { it.role to it.portableDefault() }
