package com.eignex.koblas

import com.eignex.koblas.dense.PlatformVectorKernels
import com.eignex.koblas.dense.ReferenceBackend
import com.eignex.koblas.dense.RoutedVectorKernels
import com.eignex.koblas.sparse.PlatformSparseVectorKernels
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra

/** The halves of the seam a backend can implement. */
public enum class BackendSlot {
    /** Dense vector-vector routines. */
    VectorKernels,

    /** Dense matrix routines. */
    Blas,

    /** Dense factorizations. */
    Lapack,

    /** Sparse vector-vector routines. */
    SparseVectorKernels,

    /** Sparse matrix routines. */
    SparseBlas,

    /** Sparse factorizations. */
    SparseLapack,
}

/** The backend installed in [slot]. */
public fun KoblasContext.backendFor(slot: BackendSlot): Backend = when (slot) {
    BackendSlot.VectorKernels -> vectorKernels
    BackendSlot.Blas -> blas
    BackendSlot.Lapack -> lapack
    BackendSlot.SparseVectorKernels -> sparseVectorKernels
    BackendSlot.SparseBlas -> sparseBlas
    BackendSlot.SparseLapack -> sparseLapack
}

/**
 * Whether [slot] is filled by something other than koblas's own portable implementation. Accelerated means a
 * host library is involved; the compiled-in SIMD kernels count as portable however fast they are.
 */
public fun KoblasContext.isAccelerated(slot: BackendSlot): Boolean = when (val backend = backendFor(slot)) {
    is RoutedVectorKernels -> backend.host != null
    PlatformVectorKernels -> false
    PlatformSparseVectorKernels -> false
    is ReferenceBackend -> false
    ReferenceSparseLinearAlgebra -> false
    else -> true
}

/**
 * The slots still running koblas's own portable implementation, in declaration order. All three sparse slots
 * are reported on every target today, since koblas ships no host sparse backend.
 */
public val KoblasContext.portableSlots: Set<BackendSlot>
    get() = BackendSlot.entries.filterNot { isAccelerated(it) }.toSet()

/**
 * Throws unless every one of [slots] is filled by an accelerated backend. The fallback is silent, so a
 * deployment that expected OpenBLAS and did not get it looks healthy and runs several times slower.
 *
 * @throws IllegalStateException naming each unaccelerated slot and what is filling it.
 */
public fun KoblasContext.requireAccelerated(vararg slots: BackendSlot) {
    val fallen = slots.filterNot { isAccelerated(it) }
    check(fallen.isEmpty()) {
        val detail = fallen.joinToString(", ") { "$it=${backendFor(it).name}" }
        "koblas fell back to portable implementations for: $detail. " +
            "Either the host library is missing (libopenblas/liblapacke on Linux, brew install openblas on " +
            "macOS), the backend artifact is not on the classpath, or nothing has been registered for that " +
            "slot yet. Resolved: backend=$name, kernels=${vectorKernels.name}"
    }
}
