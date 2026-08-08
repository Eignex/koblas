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

// Which backends are active, and how they got there. One file for both storages rather than a near-identical
// one per package, because a `KoblasContext` spans both and the verbs do not need to know which is which.
//
// Three things happen here. Six `Seam`s each rank the offers for one half. `koblas` assembles the winners
// into the default context. And the whole thing is discovered once, lazily, on first read.
//
// What went away: `ComposedSeam` and the two `Composed*Backend` delegation classes. Pairing halves into a
// whole was all they did, and a context is that pairing. So is the separate whole-vs-half precedence -
// there is now one rule, `installed ?: resolved`.

private val vectorKernelSeam = Seam<VectorKernels>(::recompose)

private val blasSeam = Seam<Blas>(::recompose)

private val lapackSeam = Seam<Lapack>(::recompose)

private val sparseVectorKernelSeam = Seam<SparseVectorKernels>(::recompose)

private val sparseBlasSeam = Seam<SparseBlas>(::recompose)

private val sparseLapackSeam = Seam<SparseLapack>(::recompose)

@Volatile
private var installed: KoblasContext? = null

/**
 * The context assembled from the winning halves, rebuilt only when a registration changes.
 *
 * A cached field rather than an assembly performed on demand: [koblas] is read on every dispatched call and
 * by every inner loop, so building a context per read would allocate per read.
 */
@Volatile
private var resolved: KoblasContext = assemble()

/** Runs platform discovery exactly once, on the first [koblas] read. */
private val discovery: Unit by lazy { registerPlatformBackends() }

/**
 * The process-wide default context: an [installBackends] override when set, else whatever registered
 * itself, else the portable reference implementations.
 *
 * Every free function in koblas uses this, so a program that never mentions contexts still gets the host
 * BLAS its platform found. Hold a [KoblasContext] of your own when the choice should be explicit instead.
 */
val koblas: KoblasContext
    get() {
        // One volatile read so the platform's own backends had their chance to register. It runs exactly
        // once, so a test that clears the registry stays cleared rather than having discovery repopulate it.
        discovery
        return installed ?: resolved
    }

/** Builds a context from the currently registered halves, falling back to the portable reference. */
private fun assemble(): KoblasContext = KoblasContext(
    // The compiled-in kernels are always there; a registered backend is consulted above a length.
    vectorKernels = RoutedVectorKernels(vectorKernelSeam.active),
    blas = blasSeam.active ?: ReferenceLinearAlgebra,
    lapack = lapackSeam.active ?: ReferenceLinearAlgebra,
    // Compiled in for this target, as on the dense side; a registered backend outranks it.
    sparseVectorKernels = sparseVectorKernelSeam.active ?: PlatformSparseVectorKernels,
    sparseBlas = sparseBlasSeam.active ?: ReferenceSparseLinearAlgebra,
    sparseLapack = sparseLapackSeam.active ?: ReferenceSparseLinearAlgebra,
)

private fun recompose() {
    resolved = assemble()
}

/** What this runtime resolved, for startup logging — e.g. `"backend=openblas, kernels=simd(8 lanes)"`. */
val koblasInfo: String get() = "backend=${koblas.name}, kernels=${koblas.vectorKernels.name}"

/**
 * Offers [backend] for automatic selection as every half it implements.
 *
 * One verb for all six halves. A backend is offered, not imposed: it becomes active only while no
 * [installBackends] override is set and nothing stronger was offered for the same half, ranked by
 * [Backend.priority] — openblas over cblas over the reference. Halves are ranked independently, so a host
 * with CBLAS but no LAPACKE still accelerates what it can, and passing an object that implements several
 * interfaces offers it for each.
 *
 * This is how backend artifacts activate themselves: the JVM's classpath discovery and the native
 * startup registration both come through here, so there is one ranking and one fallback rather than a
 * parallel path.
 *
 * @throws IllegalArgumentException if [backend] implements none of the six halves, which would otherwise
 *   register nothing and look like it worked.
 */
fun registerBackend(backend: Backend) {
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
 * Overrides the context [koblas] returns, taking precedence over every automatic mechanism — registration
 * and platform discovery alike. Passing null restores automatic selection.
 *
 * `installBackends(koblas.with(blas = mine))` is the usual shape: start from what resolved and change one
 * half. Note that this freezes the others as they are now, which is what an override means — later
 * registrations do not show through it.
 *
 * Visible to subsequent [koblas] reads but not synchronized with operations in flight: install during
 * startup, before other threads run.
 */
fun installBackends(context: KoblasContext?) {
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

/**
 * The kernels the compiled-in path uses when nothing is registered, for tests that need to name them.
 *
 * Exposed here rather than making [PlatformVectorKernels] public: a caller has no reason to pick the
 * compiled kernels over `koblas.vectorKernels`, which is the same thing plus any host acceleration.
 */
internal val platformKernels: VectorKernels get() = PlatformVectorKernels
