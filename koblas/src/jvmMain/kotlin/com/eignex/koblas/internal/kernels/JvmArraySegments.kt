package com.eignex.koblas.internal.kernels

import java.lang.foreign.MemorySegment
import java.lang.ref.WeakReference

/**
 * Per-thread weak cache for heap-array segments passed repeatedly to critical FFM downcalls.
 *
 * `MemorySegment.ofArray` creates a wrapper on every call and a downcall prevents escape analysis from removing it.
 * Kernel callers normally retain a small fixed set of arrays, so caching those wrappers removes a managed object per
 * operand without retaining the potentially large arrays after their caller releases them. A collected entry is
 * recreated on its next use; misses are setup/allocation work rather than the warmed kernel path.
 */
internal object JvmArraySegments {
    private val local = ThreadLocal.withInitial(::ArraySegmentCache)

    fun of(array: DoubleArray): MemorySegment = local.get().of(array)

    fun of(array: IntArray): MemorySegment = local.get().of(array)
}

private class ArraySegmentCache {
    private companion object {
        const val CAPACITY = 32
    }

    private val arrays = arrayOfNulls<WeakReference<Any>>(CAPACITY)
    private val segments = arrayOfNulls<WeakReference<MemorySegment>>(CAPACITY)
    private var replacement = 0

    fun of(array: DoubleArray): MemorySegment = find(array) ?: remember(array, MemorySegment.ofArray(array))

    fun of(array: IntArray): MemorySegment = find(array) ?: remember(array, MemorySegment.ofArray(array))

    private fun find(array: Any): MemorySegment? {
        for (slot in arrays.indices) {
            if (arrays[slot]?.get() === array) {
                val segment = segments[slot]?.get()
                if (segment != null) return segment
            }
        }
        return null
    }

    private fun remember(array: Any, segment: MemorySegment): MemorySegment {
        arrays[replacement] = WeakReference(array)
        segments[replacement] = WeakReference(segment)
        replacement = (replacement + 1) % CAPACITY
        return segment
    }
}
