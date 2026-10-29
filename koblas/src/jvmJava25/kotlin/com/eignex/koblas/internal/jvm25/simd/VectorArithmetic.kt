package com.eignex.koblas.internal.jvm25.simd

import jdk.incubator.vector.DoubleVector

/**
 * One multiply-add, fused where [hardwareFusedMultiplyAdd] found the instruction and two operations
 * where it did not.
 *
 * Inline because a helper returning a [DoubleVector] that the JIT declines to inline makes the vector a
 * heap object, allocating per loop iteration or tile depth step. Kotlin inlining keeps the vector in the
 * caller and lets the JIT fold the hardware flag so only one arithmetic form reaches the loop.
 */
@Suppress("NOTHING_TO_INLINE")
internal inline fun multiplyAdd(x: DoubleVector, y: DoubleVector, accumulator: DoubleVector): DoubleVector =
    if (hardwareFusedMultiplyAdd) x.fma(y, accumulator) else x.mul(y).add(accumulator)
