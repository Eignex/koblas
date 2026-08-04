package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.ComposedSeam
import com.eignex.koblas.Seam
import com.eignex.koblas.SparseMatrix

// Which sparse backend is active, and how it got there — the counterpart of `dense/Registry.kt`, and
// deliberately the same file in two packages: same seams, same verbs, same fallback rule, so knowing one
// side is knowing the other.
//
// Two differences from the dense side, both structural rather than accidental. There is no
// platform-discovery hook, because no target ships a sparse backend yet; when one does, it registers here
// through the same verbs. And `sparseVectorKernels` never returns null, because unlike the dense level-1
// half its fallback is an object — see `SparseVectorKernels`.

private val sparseBlasSeam = Seam<SparseBlas>(::recompose)

private val sparseLapackSeam = Seam<SparseLapack>(::recompose)

private val sparseVectorKernelSeam = Seam<SparseVectorKernels>()

private val sparseSeam = ComposedSeam<SparseLinearAlgebra>(default = ReferenceSparseLinearAlgebra)

/** The active sparse matrix backend: an [installSparseLinearAlgebra] override when set, else the strongest
 *  registered half of each kind, else the portable [ReferenceSparseLinearAlgebra]. */
val sparseKoblas: SparseLinearAlgebra get() = sparseSeam.active

/**
 * The active sparse level-1 kernels.
 *
 * Never null, unlike the dense level-1 seam. That one is nullable because its fallback is a compiled-in
 * primitive rather than an object, so `null` has to mean "use the compiled kernel"; here the fallback is
 * [ReferenceSparseLinearAlgebra] and there is no threshold to gate.
 */
val sparseVectorKernels: SparseVectorKernels
    get() = sparseVectorKernelSeam.active ?: ReferenceSparseLinearAlgebra

/** Pairs a [SparseBlas] and a [SparseLapack] when the two halves come from different places. */
private class ComposedSparseBackend(private val blas: SparseBlas, private val lapack: SparseLapack) :
    SparseLinearAlgebra,
    SparseBlas by blas,
    SparseLapack by lapack {
    override val name: String get() = if (blas.name == lapack.name) blas.name else "${blas.name}+${lapack.name}"
    override val priority: Int get() = maxOf(blas.priority, lapack.priority)
}

/** Recomputes the composed backend from the two halves, falling back to the reference for either. */
private fun recompose() {
    val blas = sparseBlasSeam.active
    val lapack = sparseLapackSeam.active
    sparseSeam.resolve(
        when {
            blas == null && lapack == null -> null

            blas === lapack && blas is SparseLinearAlgebra -> blas

            else -> ComposedSparseBackend(
                blas ?: ReferenceSparseLinearAlgebra,
                lapack ?: ReferenceSparseLinearAlgebra,
            )
        },
    )
}

/** What this runtime resolved on the sparse seams, for startup logging — e.g. `"sparse=reference"`. */
val sparseKoblasInfo: String
    get() = "sparse=${sparseKoblas.name}, sparseVector=${sparseVectorKernels.name}"

/**
 * Overrides which sparse backend [sparseKoblas] resolves to, taking precedence over registration. Passing
 * null restores automatic selection. The counterpart of `installLinearAlgebra`; not synchronized against
 * operations in flight.
 */
fun installSparseLinearAlgebra(backend: SparseLinearAlgebra?) {
    sparseSeam.install(backend)
}

/** Overrides which [SparseVectorKernels] is active; null restores automatic selection. */
fun installSparseVectorKernels(backend: SparseVectorKernels?) {
    sparseVectorKernelSeam.install(backend)
}

/** Offers the matrix half of [backend] for automatic selection, ranked by [Backend.priority]. */
fun registerSparseBlas(backend: SparseBlas) {
    sparseBlasSeam.register(backend)
}

/** Offers the factorization half of [backend] for automatic selection; see [registerSparseBlas]. */
fun registerSparseLapack(backend: SparseLapack) {
    sparseLapackSeam.register(backend)
}

/** Offers [backend] as both sparse matrix halves; the shorthand for registering one object twice. */
fun registerSparseLinearAlgebra(backend: SparseLinearAlgebra) {
    registerSparseBlas(backend)
    registerSparseLapack(backend)
}

/** Offers [backend] for automatic selection as the sparse level-1 kernels. */
fun registerSparseVectorKernels(backend: SparseVectorKernels) {
    sparseVectorKernelSeam.register(backend)
}

/** Test hook: clears the sparse matrix-seam override and registration so selection tests are
 *  order-independent. */
internal fun resetRegisteredSparseLinearAlgebra() {
    sparseSeam.reset()
    sparseBlasSeam.reset()
    sparseLapackSeam.reset()
}

/** Test hook: clears the sparse level-1 override and registration; see
 *  [resetRegisteredSparseLinearAlgebra]. */
internal fun resetRegisteredSparseVectorKernels() {
    sparseVectorKernelSeam.reset()
}

/**
 * Factorize this sparse matrix with the active backend ([sparseKoblas]) — the sparse counterpart of
 * `DenseMatrix.lu()`, carrying the same name for the same operation on the other storage.
 */
fun SparseMatrix.lu(equilibrate: Boolean = false): SparseFactorization = sparseKoblas.factor(this, equilibrate)

/**
 * `this · x`, or `thisᵀ · x` when [transpose], with the active backend ([sparseKoblas]).
 *
 * An extension rather than a member of `SparseMatrix`, matching `DenseMatrix.matMul`: the containers are
 * storage, and which backend multiplies them is a separate concern that would otherwise make the root
 * package depend on this one.
 */
fun SparseMatrix.gemv(x: DoubleArray, transpose: Boolean = false): DoubleArray = sparseKoblas.gemv(this, x, transpose)
