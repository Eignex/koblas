package com.eignex.koblas

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64PlatformVectorKernels
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.F64RoutedVectorKernels
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.dense.registerPlatformBackends
import com.eignex.koblas.sparse.F64PlatformSparseVectorKernels
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseLapack
import com.eignex.koblas.sparse.F64SparseVectorKernels
import kotlin.concurrent.Volatile

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

/** Runs platform discovery exactly once, on the first [koblas] read. */
private val discovery: Unit by lazy { registerPlatformBackends() }

/**
 * The process-wide default context: an [installBackends] override when set, else whatever registered
 * itself, else the portable reference implementations. Every free function in koblas uses this.
 */
public val koblas: F64Context
    get() {
        discovery
        return installed ?: resolved
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

/** What this runtime resolved, for startup logging (e.g. `"backend=openblas, kernels=simd(8 lanes)"`). */
public val koblasInfo: String get() = "backend=${koblas.name}, kernels=${koblas.vectorKernels.name}"

/**
 * Offers [backend] for automatic selection as every half it implements. It becomes active only while no
 * [installBackends] override is set and nothing stronger was offered for the same half.
 *
 * @throws IllegalArgumentException if [backend] implements none of the six halves, which would otherwise
 *   register nothing and look like it worked.
 */
public fun registerBackend(backend: Backend) {
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
    require(offered) {
        "${backend.name} implements none of F64VectorKernels, F64Blas, F64Lapack, " +
            "F64SparseVectorKernels, F64SparseBlas or F64SparseLapack, so there is nothing to register it as"
    }
}

/**
 * Overrides the context [koblas] returns; null restores automatic selection. Not synchronized with
 * operations in flight, so install during startup, before other threads run.
 */
public fun installBackends(context: F64Context?) {
    installed = context
}

/** Test hook: clears every override and registration, so selection tests are order-independent. */
internal fun resetBackends() {
    installed = null
    vectorKernelSeam.reset()
    blasSeam.reset()
    lapackSeam.reset()
    sparseVectorKernelSeam.reset()
    sparseBlasSeam.reset()
    sparseLapackSeam.reset()
}

/** The kernels the compiled-in path uses when nothing is registered, for tests that need to name them. */
internal val platformKernels: F64VectorKernels get() = F64PlatformVectorKernels
