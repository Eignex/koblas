package com.eignex.koblas.internal.kernels

/** A cross target without a native toolchain retains its explicit Kotlin scalar implementation. */
internal actual object NativeProbe {
    actual fun query(kind: Int, index: Int, kernelId: Int): IntArray? = null
}
