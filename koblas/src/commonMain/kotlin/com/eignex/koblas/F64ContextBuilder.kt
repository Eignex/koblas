package com.eignex.koblas

import com.eignex.koblas.dense.*
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
        require(role.accepts(backend)) { "${backend.name} does not implement $role" }
        val updated = if (role == BackendRole.SPARSE_DECOMPOSITIONS) {
            selections + mapOf(
                BackendRole.SPARSE_DECOMPOSITIONS to backend,
                BackendRole.SPARSE_GENERAL_LU to backend,
                BackendRole.SPARSE_CHOLESKY to backend,
                BackendRole.SPARSE_LDL to backend,
            )
        } else {
            selections + (role to backend)
        }
        return copy(selections = updated)
    }

    /**
     * Selects [backend] for every role it implements, without consulting global registration or priority.
     * Prefer the role overload when a specialized provider should not also fill its general capabilities.
     */
    public fun withBackend(backend: Backend): F64ContextBuilder {
        val roles = BackendRole.entries.filter { it.accepts(backend) }
        require(roles.isNotEmpty()) { "${backend.name} implements no F64 backend role" }
        return copy(selections = selections + roles.associateWith { backend })
    }

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
        val generalLu = resolved.semanticGeneralLu(sparseReference)
        val cholesky = resolved.semanticCholesky(sparseReference)
        val ldl = resolved.semanticLdl(sparseReference)
        val qr = resolved.semanticQr(sparseReference)
        val sparseRoles = F64SparseDecompositionRoles(generalLu, cholesky, ldl, qr)
        val repeated = resolved[BackendRole.SPARSE_REPEATED_LU] as? F64RepeatedSparseLu
        val basisFactorizations = resolved.semanticBasisFactorizations(sparseReference)
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
            generalSparseLu = generalLu,
            repeatedSparseLu = repeated,
            sparseCholesky = cholesky,
            sparseLdl = ldl,
            sparseQr = qr,
            basisFactorizations = basisFactorizations,
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

private fun portableSelections(): Map<BackendRole, Backend> = mapOf(
    BackendRole.DENSE_KERNELS to F64RoutedKernels(null),
    BackendRole.DENSE_BLAS to F64ReferenceLinearAlgebra,
    BackendRole.DENSE_DECOMPOSITIONS to F64ReferenceLinearAlgebra,
    BackendRole.SPARSE_KERNELS to F64PlatformSparseKernels,
    BackendRole.SPARSE_BLAS to F64ReferenceSparseLinearAlgebra,
    BackendRole.SPARSE_DECOMPOSITIONS to F64ReferenceSparseLinearAlgebra,
    BackendRole.SPARSE_GENERAL_LU to F64ReferenceSparseLinearAlgebra,
    BackendRole.SPARSE_REPEATED_LU to MissingRepeatedSparseLu,
    BackendRole.SPARSE_CHOLESKY to F64ReferenceSparseLinearAlgebra,
    BackendRole.SPARSE_LDL to F64ReferenceSparseLinearAlgebra,
    BackendRole.SPARSE_QR to F64ReferenceSparseLinearAlgebra,
    BackendRole.BASIS_FACTORIZATIONS to F64ReferenceSparseLinearAlgebra,
    BackendRole.BASIS_SOLVERS to F64ReferenceSparseLinearAlgebra,
)

private fun BackendRole.accepts(backend: Backend): Boolean = when (this) {
    BackendRole.DENSE_KERNELS -> backend is F64Kernels

    BackendRole.DENSE_BLAS -> backend is F64Blas

    BackendRole.DENSE_DECOMPOSITIONS -> backend is F64Decompositions

    BackendRole.SPARSE_KERNELS -> backend is F64SparseKernels

    BackendRole.SPARSE_BLAS -> backend is F64SparseBlas

    BackendRole.SPARSE_DECOMPOSITIONS -> backend is F64SparseDecompositions

    BackendRole.SPARSE_GENERAL_LU ->
        backend is F64GeneralSparseLu ||
            (
                backend is F64SparseDecompositions &&
                    backend !is F64RepeatedSparseLu &&
                    backend !is F64BasisFactorizations
                )

    BackendRole.SPARSE_REPEATED_LU -> backend is F64RepeatedSparseLu

    BackendRole.SPARSE_CHOLESKY ->
        backend is F64SparseCholesky ||
            (backend is F64SparseDecompositions && backend !is F64BasisFactorizations)

    BackendRole.SPARSE_LDL ->
        backend is F64SparseLdl ||
            (backend is F64SparseDecompositions && backend !is F64BasisFactorizations)

    BackendRole.SPARSE_QR ->
        backend is F64SparseQr ||
            (backend is F64SparseDecompositions && backend !is F64BasisFactorizations)

    BackendRole.BASIS_FACTORIZATIONS -> backend is F64BasisFactorizations

    BackendRole.BASIS_SOLVERS -> backend is F64BasisSolvers
}

private fun Map<BackendRole, Backend>.semanticGeneralLu(reference: F64ReferenceSparseBackend): F64GeneralSparseLu =
    when (val backend = getValue(BackendRole.SPARSE_GENERAL_LU)) {
        F64ReferenceSparseLinearAlgebra -> reference
        is F64GeneralSparseLu -> backend
        else -> error("${backend.name} does not implement general sparse LU")
    }

private fun Map<BackendRole, Backend>.semanticCholesky(reference: F64ReferenceSparseBackend): F64SparseCholesky =
    when (val backend = getValue(BackendRole.SPARSE_CHOLESKY)) {
        F64ReferenceSparseLinearAlgebra -> reference
        is F64SparseCholesky -> backend
        else -> error("${backend.name} does not implement sparse Cholesky")
    }

private fun Map<BackendRole, Backend>.semanticLdl(reference: F64ReferenceSparseBackend): F64SparseLdl =
    when (val backend = getValue(BackendRole.SPARSE_LDL)) {
        F64ReferenceSparseLinearAlgebra -> reference
        is F64SparseLdl -> backend
        else -> error("${backend.name} does not implement sparse LDL")
    }

private fun Map<BackendRole, Backend>.semanticQr(reference: F64ReferenceSparseBackend): F64SparseQr =
    when (val backend = getValue(BackendRole.SPARSE_QR)) {
        F64ReferenceSparseLinearAlgebra -> reference
        is F64SparseQr -> backend
        else -> error("${backend.name} does not implement sparse QR")
    }

private fun Map<BackendRole, Backend>.semanticBasisFactorizations(
    reference: F64ReferenceSparseBackend,
): F64BasisFactorizations = when (val backend = getValue(BackendRole.BASIS_FACTORIZATIONS)) {
    F64ReferenceSparseLinearAlgebra -> reference
    is F64BasisFactorizations -> backend
    else -> error("${backend.name} does not implement basis factorizations")
}
