package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import kotlin.concurrent.Volatile

@Volatile
private var installedSparseVectorKernels: SparseVectorKernels? = null

@Volatile
private var registeredSparseVectorKernels: SparseVectorKernels? = null

@Volatile
private var installedSparse: SparseLinearAlgebra? = null

@Volatile
private var registeredSparseBlas: SparseBlas? = null

@Volatile
private var registeredSparseLapack: SparseLapack? = null

/**
 * The composed sparse backend, rebuilt only when a registration changes, for the same reason the dense one
 * is cached: [sparseKoblas] is read on every dispatched call, so composing on demand would allocate per
 * call.
 */
@Volatile
private var resolvedSparse: SparseLinearAlgebra? = null

/** The active sparse matrix backend: an [installSparseLinearAlgebra] override when set, else the strongest
 *  registered half of each kind, else the portable [ReferenceSparseLinearAlgebra]. */
val sparseKoblas: SparseLinearAlgebra
    get() = installedSparse ?: resolvedSparse ?: ReferenceSparseLinearAlgebra

/**
 * The active sparse level-1 kernels.
 *
 * Never null, unlike the dense `activeVectorKernels`. That one is nullable because its fallback is a
 * compiled-in primitive rather than an object, so `null` has to mean "use the compiled kernel"; here the
 * fallback is [ReferenceSparseLinearAlgebra] and there is no threshold to gate.
 */
val sparseVectorKernels: SparseVectorKernels
    get() = installedSparseVectorKernels ?: registeredSparseVectorKernels ?: ReferenceSparseLinearAlgebra

/** Pairs a [SparseBlas] and a [SparseLapack] when the two halves come from different places. */
private class ComposedSparseBackend(private val blas: SparseBlas, private val lapack: SparseLapack) :
    SparseLinearAlgebra,
    SparseBlas by blas,
    SparseLapack by lapack {
    override val name: String get() = if (blas.name == lapack.name) blas.name else "${blas.name}+${lapack.name}"
    override val priority: Int get() = maxOf(blas.priority, lapack.priority)
}

private fun recomposeSparse() {
    val blas = registeredSparseBlas
    val lapack = registeredSparseLapack
    resolvedSparse = when {
        blas == null && lapack == null -> null

        blas === lapack && blas is SparseLinearAlgebra -> blas

        else -> ComposedSparseBackend(
            blas ?: ReferenceSparseLinearAlgebra,
            lapack ?: ReferenceSparseLinearAlgebra,
        )
    }
}

/**
 * Overrides which sparse backend [sparseKoblas] resolves to, taking precedence over registration. Passing
 * null restores automatic selection. The counterpart of `installLinearAlgebra`; not synchronized against
 * operations in flight.
 */
fun installSparseLinearAlgebra(backend: SparseLinearAlgebra?) {
    installedSparse = backend
}

/** Overrides which [SparseVectorKernels] is active; null restores automatic selection. */
fun installSparseVectorKernels(backend: SparseVectorKernels?) {
    installedSparseVectorKernels = backend
}

/** Offers the matrix half of [backend] for automatic selection, ranked by [Backend.priority]. */
fun registerSparseBlas(backend: SparseBlas) {
    val current = registeredSparseBlas
    if (current == null || backend.priority > current.priority) {
        registeredSparseBlas = backend
        recomposeSparse()
    }
}

/** Offers the factorization half of [backend] for automatic selection; see [registerSparseBlas]. */
fun registerSparseLapack(backend: SparseLapack) {
    val current = registeredSparseLapack
    if (current == null || backend.priority > current.priority) {
        registeredSparseLapack = backend
        recomposeSparse()
    }
}

/** Offers [backend] as both sparse matrix halves; the shorthand for registering one object twice. */
fun registerSparseLinearAlgebra(backend: SparseLinearAlgebra) {
    registerSparseBlas(backend)
    registerSparseLapack(backend)
}

/** Offers [backend] for automatic selection as the sparse level-1 kernels. */
fun registerSparseVectorKernels(backend: SparseVectorKernels) {
    val current = registeredSparseVectorKernels
    if (current == null || backend.priority > current.priority) {
        registeredSparseVectorKernels = backend
    }
}

/** What this runtime resolved on the sparse seams, for startup logging — e.g. `"sparse=reference"`. */
val sparseKoblasInfo: String
    get() = "sparse=${sparseKoblas.name}, sparseVector=${sparseVectorKernels.name}"

/** Test hook: clears sparse registration so selection tests are order-independent. */
internal fun resetRegisteredSparse() {
    installedSparse = null
    installedSparseVectorKernels = null
    registeredSparseBlas = null
    registeredSparseLapack = null
    registeredSparseVectorKernels = null
    recomposeSparse()
}
