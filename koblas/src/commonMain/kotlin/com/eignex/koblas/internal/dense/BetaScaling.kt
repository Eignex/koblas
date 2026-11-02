package com.eignex.koblas.internal.dense

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.*

/** `v = beta * v` over the [len] entries from [off], honoring the `beta == 0` overwrite convention. */
internal fun applyBeta(k: KoblasEngine, v: DoubleArray, off: Int, len: Int, beta: Double, stride: Int = 1) {
    when {
        beta == 0.0 -> if (stride == 1) {
            v.fill(
                0.0,
                off,
                off + len,
            )
        } else {
            for (i in 0 until len) v[off + i * stride] = 0.0
        }

        beta != 1.0 -> k.scale(v, off, beta, len, stride)
    }
}
