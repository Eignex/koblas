package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.PlatformVectorKernels
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.RoutedVectorKernels
import com.eignex.koblas.dense.VectorKernels
import com.eignex.koblas.dense.registerPlatformBackends
import com.eignex.koblas.sparse.PlatformSparseVectorKernels
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseVectorKernels
import kotlin.concurrent.Volatile

private val vectorKernelSeam = Seam<VectorKernels>(::recompose)

private val blasSeam = Seam<Blas>(::recompose)

private val lapackSeam = Seam<Lapack>(::recompose)

private val sparseVectorKernelSeam = Seam<SparseVectorKernels>(::recompose)

private val sparseBlasSeam = Seam<SparseBlas>(::recompose)

private val sparseLapackSeam = Seam<SparseLapack>(::recompose)

@Volatile
private var installed: KoblasContext? = null

/** The context assembled from the winning halves, rebuilt only when a registration changes. */
@Volatile
private var resolved: KoblasContext = assemble()

/** Runs platform discovery exactly once, on the first [koblas] read. */
private val discovery: Unit by lazy { registerPlatformBackends() }

/**
 * The process-wide default context: an [installBackends] override when set, else whatever registered
 * itself, else the portable reference implementations. Every free function in koblas uses this.
 */
public val koblas: KoblasContext
    get() {
        discovery
        return installed ?: resolved
    }

/** Builds a context from the currently registered halves, falling back to the portable reference. */
private fun assemble(): KoblasContext = KoblasContext(
    vectorKernels = RoutedVectorKernels(vectorKernelSeam.active),
    blas = blasSeam.active ?: ReferenceLinearAlgebra,
    lapack = lapackSeam.active ?: ReferenceLinearAlgebra,
    sparseVectorKernels = sparseVectorKernelSeam.active ?: PlatformSparseVectorKernels,
    sparseBlas = sparseBlasSeam.active ?: ReferenceSparseLinearAlgebra,
    sparseLapack = sparseLapackSeam.active ?: ReferenceSparseLinearAlgebra,
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
    if (backend is VectorKernels) {
        vectorKernelSeam.register(backend)
        offered = true
    }
    if (backend is Blas) {
        blasSeam.register(backend)
        offered = true
    }
    if (backend is Lapack) {
        lapackSeam.register(backend)
        offered = true
    }
    if (backend is SparseVectorKernels) {
        sparseVectorKernelSeam.register(backend)
        offered = true
    }
    if (backend is SparseBlas) {
        sparseBlasSeam.register(backend)
        offered = true
    }
    if (backend is SparseLapack) {
        sparseLapackSeam.register(backend)
        offered = true
    }
    require(offered) {
        "${backend.name} implements none of VectorKernels, Blas, Lapack, SparseVectorKernels, SparseBlas " +
            "or SparseLapack, so there is nothing to register it as"
    }
}

/**
 * Overrides the context [koblas] returns; null restores automatic selection. Not synchronized with
 * operations in flight, so install during startup, before other threads run.
 */
public fun installBackends(context: KoblasContext?) {
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
internal val platformKernels: VectorKernels get() = PlatformVectorKernels
