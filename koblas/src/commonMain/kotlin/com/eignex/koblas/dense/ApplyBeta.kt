package com.eignex.koblas.dense

/** `v = beta * v` over the [len] entries from [off], honoring the `beta == 0` overwrite convention. */
internal fun applyBeta(k: DenseVectorKernels, v: DoubleArray, off: Int, len: Int, beta: Double) {
    when {
        beta == 0.0 -> v.fill(0.0, off, off + len)
        beta != 1.0 -> k.scale(v, off, beta, len)
    }
}
