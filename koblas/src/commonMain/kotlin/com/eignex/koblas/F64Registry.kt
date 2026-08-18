package com.eignex.koblas

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.F64RoutedVectorKernels
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.sparse.F64PlatformSparseVectorKernels
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseLapack
import com.eignex.koblas.sparse.F64SparseVectorKernels
import kotlin.concurrent.Volatile

/**
 * The six double-precision seams and the context they compose. One element type's registrations live in one
 * of these, so an element type added later brings its own rather than widening this one: its halves are
 * different interfaces, and a backend offered to [registerBackend] belongs to whichever registry recognises
 * it. [Seam] and the ranking are shared, so what differs between two registries is only the six types.
 */
internal class F64Registry {
    private val vectorKernelSeam = Seam<F64VectorKernels>(::recompose)
    private val blasSeam = Seam<F64Blas>(::recompose)
    private val lapackSeam = Seam<F64Lapack>(::recompose)
    private val sparseVectorKernelSeam = Seam<F64SparseVectorKernels>(::recompose)
    private val sparseBlasSeam = Seam<F64SparseBlas>(::recompose)
    private val sparseLapackSeam = Seam<F64SparseLapack>(::recompose)

    @Volatile
    private var installed: F64Context? = null

    /** The context assembled from the winning halves, rebuilt only when a registration changes. */
    @Volatile
    private var resolved: F64Context = assemble()

    /** The context in force: an [install] override when set, else what the seams resolved to. */
    val active: F64Context get() = installed ?: resolved

    /** Overrides [active] wholesale; null restores automatic selection. */
    fun install(context: F64Context?) {
        installed = context
    }

    /**
     * Offers [backend] to every seam whose half it implements, and reports whether any took it. A caller
     * that gets false has offered something this element type has no seam for.
     */
    fun offer(backend: Backend): Boolean {
        var offered = false
        if (backend is F64VectorKernels) {
            vectorKernelSeam.register(backend)
            offered = true
        }
        if (backend is F64Blas) {
            blasSeam.register(backend)
            offered = true
        }
        if (backend is F64Lapack) {
            lapackSeam.register(backend)
            offered = true
        }
        if (backend is F64SparseVectorKernels) {
            sparseVectorKernelSeam.register(backend)
            offered = true
        }
        if (backend is F64SparseBlas) {
            sparseBlasSeam.register(backend)
            offered = true
        }
        if (backend is F64SparseLapack) {
            sparseLapackSeam.register(backend)
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
        sparseLapackSeam.reset()
    }

    /** Builds a context from the currently registered halves, falling back to the portable reference. */
    private fun assemble(): F64Context = F64Context(
        vectorKernels = F64RoutedVectorKernels(vectorKernelSeam.active),
        blas = blasSeam.active ?: F64ReferenceLinearAlgebra,
        lapack = lapackSeam.active ?: F64ReferenceLinearAlgebra,
        sparseVectorKernels = sparseVectorKernelSeam.active ?: F64PlatformSparseVectorKernels,
        sparseBlas = sparseBlasSeam.active ?: F64ReferenceSparseLinearAlgebra,
        sparseLapack = sparseLapackSeam.active ?: F64ReferenceSparseLinearAlgebra,
    )

    private fun recompose() {
        resolved = assemble()
    }

    companion object {
        /** The halves this registry has seams for, as [registerBackend] names them when nothing matched. */
        const val HALF_NAMES: String = "F64VectorKernels, F64Blas, F64Lapack, F64SparseVectorKernels, " +
            "F64SparseBlas or F64SparseLapack"
    }
}
