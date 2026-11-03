package com.eignex.koblas.engine

/** Scoped temporary arrays; arithmetic does not own their retention policy. */
internal interface ArraySupplier {
    fun take(size: Int): DoubleArray
    fun takeI32(size: Int): IntArray
    fun release(buffer: DoubleArray)
    fun release(buffer: IntArray)
}

/** Always return loans, including when operand reads or arithmetic throw. */
internal inline fun <T> ArraySupplier?.borrow(size: Int, block: (DoubleArray) -> T): T {
    val buffer = this?.take(size) ?: DoubleArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

internal inline fun <T> ArraySupplier?.borrowI32(size: Int, block: (IntArray) -> T): T {
    val buffer = this?.takeI32(size) ?: IntArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

private val EMPTY_BUFFER = DoubleArray(0)

/** Unused panels consume neither a loan nor a retained length. */
internal inline fun <T> ArraySupplier?.borrowOptional(size: Int, block: (DoubleArray) -> T): T =
    if (size == 0) block(EMPTY_BUFFER) else borrow(size, block)
