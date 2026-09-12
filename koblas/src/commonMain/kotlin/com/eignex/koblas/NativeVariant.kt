package com.eignex.koblas

/** Exact ordinary native implementation, independent of Kotlin scalar and JVM Vector API kernels. */
public enum class NativeVariant(internal val id: Int) {
    /** C arithmetic compiled with vectorization disabled. */
    SCALAR(1),

    /** Ordinary 128-bit x86 SSE2 arithmetic. */
    SSE2(2),

    /** Ordinary 256-bit x86 AVX2 arithmetic, requiring OS-managed YMM state. */
    AVX2(3),

    /** Ordinary 128-bit Arm Advanced SIMD arithmetic. */
    NEON(4),
}
