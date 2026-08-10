package com.eignex.koblas

import com.eignex.koblas.dense.PlatformVectorKernels
import com.eignex.koblas.dense.ReferenceBackend
import com.eignex.koblas.dense.RoutedVectorKernels
import com.eignex.koblas.sparse.PlatformSparseVectorKernels
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra

/**
 * One replaceable half of a [KoblasContext], named so a caller can ask about it.
 *
 * koblas resolves backends silently: a machine without the host library runs the portable kernels and gets
 * the right answers, slower. That is the right default — a program should not fail to start because OpenBLAS
 * is missing — but it makes the difference invisible, and these names are what make it checkable.
 */
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

/** The backend currently filling [slot]. */
public fun KoblasContext.backendFor(slot: BackendSlot): Backend = when (slot) {
    BackendSlot.VectorKernels -> vectorKernels
    BackendSlot.Blas -> blas
    BackendSlot.Lapack -> lapack
    BackendSlot.SparseVectorKernels -> sparseVectorKernels
    BackendSlot.SparseBlas -> sparseBlas
    BackendSlot.SparseLapack -> sparseLapack
}

/**
 * Whether [slot] is filled by something other than koblas's own portable implementation.
 *
 * Determined by what the backend *is*, not by [Backend.priority]: priority is an ordering hint a backend
 * declares about itself, and koblas's routed vector kernels have no meaningful one to report. For the
 * vector slot, accelerated means a registered backend is being consulted for long runs — the compiled-in
 * SIMD kernels are still portable koblas, however fast they are, since they are what you get with no host
 * library present.
 */
public fun KoblasContext.isAccelerated(slot: BackendSlot): Boolean = when (val backend = backendFor(slot)) {
    // The routed pair is portable exactly when nothing is registered behind it.
    is RoutedVectorKernels -> backend.host != null

    PlatformVectorKernels -> false

    // Compiled in for the target, like the dense kernels above: SIMD where the platform has it, and still
    // koblas's own portable implementation rather than a host library.
    PlatformSparseVectorKernels -> false

    is ReferenceBackend -> false

    ReferenceSparseLinearAlgebra -> false

    else -> true
}

/**
 * The slots still running koblas's own portable implementation, in declaration order.
 *
 * Empty means everything is accelerated. Print it at startup, or assert on it — see [requireAccelerated] for
 * the failing-fast version.
 *
 * Note what this will report today on every target: all three sparse slots. koblas has the sparse seams but
 * no host sparse backend behind them yet, so `SparseBlas` and friends being portable is the current state of
 * the library rather than anything wrong with the machine.
 */
public val KoblasContext.portableSlots: Set<BackendSlot>
    get() = BackendSlot.entries.filterNot { isAccelerated(it) }.toSet()

/**
 * Throws unless every one of [slots] is filled by an accelerated backend.
 *
 * The point of this function is that koblas's fallback is silent by design. Discovery finds no host library,
 * registration stays empty, the portable kernels answer correctly, and nothing says so. A deployment that
 * expected OpenBLAS and did not get it therefore looks healthy and runs several times slower. Call this at
 * startup when that is not acceptable:
 *
 * ```kotlin
 * koblas.requireAccelerated(BackendSlot.Blas, BackendSlot.Lapack)
 * ```
 *
 * The slots are explicit on purpose, with no "all of them" default. Which slots *can* be accelerated is a
 * property of the target and of what koblas currently ships — no host sparse backend exists yet, so a
 * default covering all six would fail everywhere, and a default covering "the dense three" would quietly
 * rot the day a sparse one lands. Naming what you depend on keeps the check meaningful.
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
            // This context's own summary, not the global `koblasInfo`: the receiver may be a custom context,
            // and reporting what the process resolved would describe something the caller never used.
            "slot yet. Resolved: backend=$name, kernels=${vectorKernels.name}"
    }
}
