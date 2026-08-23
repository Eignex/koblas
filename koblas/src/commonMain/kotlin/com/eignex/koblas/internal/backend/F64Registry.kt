package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.F64Context
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.F64RoutedVectorKernels
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.sparse.F64PlatformSparseVectorKernels
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.F64SparseVectorKernels
import kotlin.concurrent.Volatile
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicReference
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.concurrent.atomics.incrementAndFetch

/**
 * The six double-precision seams and the context they compose. One element type's registrations live in one
 * of these, so an element type added later brings its own rather than widening this one: its halves are
 * different interfaces, and a backend offered to the public registry belongs to whichever registry recognises
 * it. [Seam] and the ranking are shared, so what differs between two registries is only the six types.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class F64Registry {
    private val vectorKernelSeam = Seam<F64VectorKernels>(::recompose)
    private val blasSeam = Seam<F64Blas>(::recompose)
    private val lapackSeam = Seam<F64Lapack>(::recompose)
    private val sparseVectorKernelSeam = Seam<F64SparseVectorKernels>(::recompose)
    private val sparseBlasSeam = Seam<F64SparseBlas>(::recompose)
    private val sparseLuSeam = Seam<F64SparseLu>(::recompose)

    @Volatile
    private var installed: F64Context? = null

    /** A context and the [changes] count it was assembled at, published as one so neither can be read alone. */
    private class Resolution(val at: Int, val context: F64Context)

    /** Bumped by every seam change, so a stale [resolution] can be told from a current one. */
    private val changes = AtomicInt(0)

    private val resolution = AtomicReference<Resolution?>(null)

    /**
     * The context in force: an [install] override when set, else what the seams resolved to.
     *
     * Assembled on demand against a change count rather than eagerly on every registration. Eager rebuilds
     * race: two threads registering into different halves could each assemble from a partial view, and the
     * later write would drop the other's registration with nothing left to trigger a rebuild. Reading the
     * count before assembling means a stamp that still matches cannot have missed a change, and a stamp
     * that no longer matches costs a rebuild rather than a wrong answer.
     */
    val active: F64Context get() {
        installed?.let { return it }
        val at = changes.load()
        resolution.load()?.let { if (it.at == at) return it.context }
        val fresh = Resolution(at, assemble())
        resolution.store(fresh)
        return fresh.context
    }

    /** Overrides [active] wholesale; null restores automatic selection. */
    fun install(context: F64Context?) {
        installed = context
    }

    /**
     * Offers [backend] to every seam whose half it implements, and reports whether any took it. Explicit
     * offers outrank automatic ones. A caller that gets false has offered something this element type has no
     * seam for.
     */
    fun offer(backend: Backend, explicit: Boolean): Boolean {
        var offered = false
        if (backend is F64VectorKernels) {
            vectorKernelSeam.register(backend, explicit)
            offered = true
        }
        if (backend is F64Blas) {
            blasSeam.register(backend, explicit)
            offered = true
        }
        if (backend is F64Lapack) {
            lapackSeam.register(backend, explicit)
            offered = true
        }
        if (backend is F64SparseVectorKernels) {
            sparseVectorKernelSeam.register(backend, explicit)
            offered = true
        }
        if (backend is F64SparseBlas) {
            sparseBlasSeam.register(backend, explicit)
            offered = true
        }
        if (backend is F64SparseLu) {
            sparseLuSeam.register(backend, explicit)
            offered = true
        }
        return offered
    }

    /** Clears the override and every registration, leaving the portable fallbacks. */
    fun reset() {
        installed = null
        vectorKernelSeam.reset()
        blasSeam.reset()
        lapackSeam.reset()
        sparseVectorKernelSeam.reset()
        sparseBlasSeam.reset()
        sparseLuSeam.reset()
    }

    /** Builds a context from the currently registered halves, falling back to the portable reference. */
    private fun assemble(): F64Context = F64Context(
        vectorKernels = F64RoutedVectorKernels(vectorKernelSeam.active),
        blas = blasSeam.active ?: F64ReferenceLinearAlgebra,
        lapack = lapackSeam.active ?: F64ReferenceLinearAlgebra,
        sparseVectorKernels = sparseVectorKernelSeam.active ?: F64PlatformSparseVectorKernels,
        sparseBlas = sparseBlasSeam.active ?: F64ReferenceSparseLinearAlgebra,
        sparseLu = sparseLuSeam.active ?: F64ReferenceSparseLinearAlgebra,
    )

    private fun recompose() {
        changes.incrementAndFetch()
    }

    companion object {
        /** The halves this registry has seams for, used in diagnostics when nothing matched. */
        const val HALF_NAMES: String = "F64VectorKernels, F64Blas, F64Lapack, F64SparseVectorKernels, " +
            "F64SparseBlas or F64SparseLu"
    }
}
