package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.internal.backend.slot
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.basis.F64BasisSolvers

/**
 * An immutable resolver for an explicit [F64Context], initially containing only portable implementations.
 * Every `with` method returns a new builder, so configurations can be safely retained and shared.
 */
public class F64ContextBuilder private constructor(
    private val selections: Map<BackendRole, Backend>,
    /** The operation-level dispatch requirement of the resolved context. */
    public val dispatchPolicy: F64DispatchPolicy,
    /** The action for non-native inspected routes in automatic mode. */
    public val fallbackPolicy: F64FallbackPolicy,
    private val fallbackWarning: ((BackendRoute) -> Unit)?,
) {
    /** Creates a resolver seeded with koblas's portable implementations for every role. */
    public constructor() : this(
        portableSelections(),
        F64DispatchPolicy.AUTO,
        F64FallbackPolicy.ALLOW,
        null,
    )

    /** Returns a resolver seeded from the exact role selections and policies of [context]. */
    public constructor(context: F64Context) : this(
        BackendRole.entries.associateWith(context::backendFor),
        context.dispatchPolicy,
        context.fallbackPolicy,
        context.fallbackWarning,
    )

    /** Selects [backend] for [role], without consulting global registration or priority. */
    public fun withBackend(role: BackendRole, backend: Backend): F64ContextBuilder {
        require(role.slot.accepts(backend)) { "${backend.name} does not implement $role" }
        return copy(selections = selections + (role to backend))
    }

    /**
     * Selects [backend] for every role it implements, without consulting global registration or priority.
     * A specialized provider keeps the roles that specialization is about and leaves the general ones it
     * also implements to a general provider, which is what registration does with the same object. The role
     * overload selects it for a general role anyway.
     */
    public fun withBackend(backend: Backend): F64ContextBuilder {
        val roles = BackendSlot.entries.filter { it.acceptsOffer(backend) }.map { it.role }
        require(roles.isNotEmpty()) { "${backend.name} implements no F64 backend role" }
        return copy(selections = selections + roles.associateWith { backend })
    }

    /** Selects both level-1 halves from an exact built-in [provider]. */
    @ExperimentalKoblasApi
    public fun withBuiltinKernels(provider: F64BuiltinKernelProvider): F64ContextBuilder =
        withBackend(BackendRole.DENSE_KERNELS, provider.kernels)
            .withBackend(BackendRole.SPARSE_KERNELS, provider.sparseKernels)

    /** Returns a resolver using [policy] for operation-level dispatch. */
    public fun withDispatchPolicy(policy: F64DispatchPolicy): F64ContextBuilder = copy(dispatchPolicy = policy)

    /** Returns a resolver using [policy] for automatic fallbacks. */
    public fun withFallbackPolicy(policy: F64FallbackPolicy): F64ContextBuilder = copy(fallbackPolicy = policy)

    /** Returns a resolver that sends warning routes to [handler]. */
    public fun onFallback(handler: (BackendRoute) -> Unit): F64ContextBuilder = copy(fallbackWarning = handler)

    /** Resolves a new immutable context without reading or mutating the process-wide registry. */
    public fun resolve(): F64Context {
        require(fallbackPolicy != F64FallbackPolicy.WARN || fallbackWarning != null) {
            "WARN fallback policy requires an onFallback handler"
        }
        val resolved = if (dispatchPolicy == F64DispatchPolicy.PORTABLE_ONLY) portableSelections() else selections
        val kernels = resolved.getValue(BackendRole.DENSE_KERNELS) as F64Kernels
        val denseReference = F64ReferenceBackend(kernels)
        val sparseReference = F64ReferenceSparseBackend(kernels)
        val generalLu = resolved.semantic<F64GeneralSparseLu>(BackendRole.SPARSE_GENERAL_LU, sparseReference)
        val cholesky = resolved.semantic<F64SparseCholesky>(BackendRole.SPARSE_CHOLESKY, sparseReference)
        val quasiDefiniteLdl =
            resolved.semantic<F64QuasiDefiniteLdl>(BackendRole.SPARSE_QUASI_DEFINITE_LDL, sparseReference)
        val qr = resolved.semantic<F64SparseQr>(BackendRole.SPARSE_QR, sparseReference)
        val sparseRoles = F64SparseDecompositionRoles(generalLu, cholesky, quasiDefiniteLdl, qr)
        val repeated = resolved[BackendRole.SPARSE_REPEATED_LU] as? F64RepeatedSparseLu
        val basisFactorizations =
            resolved.semantic<F64BasisFactorizations>(BackendRole.BASIS_FACTORIZATIONS, sparseReference)
        return F64Context(
            kernels = kernels,
            blas = resolved.boundReference(BackendRole.DENSE_BLAS, denseReference) as F64Blas,
            decompositions = resolved.boundReference(
                BackendRole.DENSE_DECOMPOSITIONS,
                denseReference,
            ) as F64Decompositions,
            sparseKernels = resolved.getValue(BackendRole.SPARSE_KERNELS) as F64SparseKernels,
            sparseBlas = resolved.boundReference(BackendRole.SPARSE_BLAS, sparseReference) as F64SparseBlas,
            sparseDecompositions = sparseRoles,
            basisSolvers = resolved.boundReference(BackendRole.BASIS_SOLVERS, sparseReference) as F64BasisSolvers,
            dispatchPolicy = dispatchPolicy,
            fallbackPolicy = fallbackPolicy,
            fallbackWarning = fallbackWarning ?: {},
            roles = SparseRoles(
                generalLu = generalLu,
                repeatedLu = repeated,
                cholesky = cholesky,
                quasiDefiniteLdl = quasiDefiniteLdl,
                qr = qr,
                basisFactorizations = basisFactorizations,
            ),
        )
    }

    @Suppress("LongParameterList") // mirrors the four immutable builder fields
    private fun copy(
        selections: Map<BackendRole, Backend> = this.selections,
        dispatchPolicy: F64DispatchPolicy = this.dispatchPolicy,
        fallbackPolicy: F64FallbackPolicy = this.fallbackPolicy,
        fallbackWarning: ((BackendRoute) -> Unit)? = this.fallbackWarning,
    ): F64ContextBuilder = F64ContextBuilder(
        selections.toMap(),
        dispatchPolicy,
        fallbackPolicy,
        fallbackWarning,
    )
}

private fun Map<BackendRole, Backend>.boundReference(role: BackendRole, configured: Backend): Backend {
    val selected = getValue(role)
    return when (selected) {
        F64ReferenceLinearAlgebra, F64ReferenceSparseLinearAlgebra -> configured
        else -> selected
    }
}

private fun portableSelections(): Map<BackendRole, Backend> =
    BackendSlot.entries.associate { it.role to it.portableDefault() }

/**
 * The provider selected for [role], as the interface the role is defined by. The type holds because
 * [F64ContextBuilder.withBackend] takes no selection its role does not accept, so the error is a guard on a
 * selection map built some other way rather than something a caller can reach.
 */
private inline fun <reified T : Backend> Map<BackendRole, Backend>.semantic(
    role: BackendRole,
    reference: F64ReferenceSparseBackend,
): T {
    val backend = boundReference(role, reference)
    return backend as? T ?: error("${backend.name} does not implement $role")
}
