package com.eignex.koblas

import com.eignex.koblas.dense.F64RoutedVectorKernels

/** The halves of the backend seam a context reports on. */
public typealias BackendSlot = com.eignex.koblas.internal.backend.BackendSlot

/** The backend installed in [slot]. */
public fun F64Context.backendFor(slot: BackendSlot): Backend = when (slot) {
    BackendSlot.F64VectorKernels -> vectorKernels
    BackendSlot.F64Blas -> blas
    BackendSlot.F64Lapack -> lapack
    BackendSlot.F64SparseVectorKernels -> sparseVectorKernels
    BackendSlot.F64SparseBlas -> sparseBlas
    BackendSlot.F64SparseLapack -> sparseLapack
}

/** Whether [slot] is filled by something other than koblas's own portable implementation. */
public fun F64Context.isAccelerated(slot: BackendSlot): Boolean = when (val backend = backendFor(slot)) {
    is F64RoutedVectorKernels -> backend.host != null
    else -> !backend.isPortable
}

/** The slots still running koblas's own portable implementation, in declaration order. */
public val F64Context.portableSlots: Set<BackendSlot>
    get() = BackendSlot.entries.filterNot { isAccelerated(it) }.toSet()

/** Throws unless every one of [slots] is filled by an accelerated backend. */
public fun F64Context.requireAccelerated(vararg slots: BackendSlot) {
    val fallen = slots.filterNot { isAccelerated(it) }
    check(fallen.isEmpty()) {
        val detail = fallen.joinToString(", ") { "$it=${backendFor(it).name}" }
        "koblas fell back to portable implementations for: $detail. " +
            "Either the host library is missing (libopenblas/liblapacke on Linux, brew install openblas on " +
            "macOS), the backend artifact is not on the classpath, or nothing has been registered for that " +
            "slot yet. Resolved: backend=$name, kernels=${vectorKernels.name}"
    }
}
