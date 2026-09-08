package com.eignex.koblas.bench

/** Fails a benchmark fork when a steady-state kernel unexpectedly allocates managed memory. */
internal expect fun verifyNearZeroManagedAllocation(label: String, operation: () -> Unit)

internal fun reportAllocatingWorkload(label: String, reason: String) {
    println("allocation: operation=$label expectation=workload-dependent reason=$reason")
}
