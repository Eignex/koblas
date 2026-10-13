package com.eignex.koblas.vendor

/** Deterministic finite operands shared by vendor conformance and concurrency tests. */
internal fun vendorValues(size: Int, seed: Int = 1): DoubleArray {
    var state = seed
    return DoubleArray(size) {
        state = state * 1_103_515_245 + 12_345
        ((state ushr 8) % 1000) / 250.0 - 2.0
    }
}
