package com.eignex.koblas.bench

internal actual fun verifyNearZeroManagedAllocation(label: String, operation: () -> Unit) {
    operation()
    println("allocation: operation=$label expectation=near-zero measured=unavailable target=native")
}
