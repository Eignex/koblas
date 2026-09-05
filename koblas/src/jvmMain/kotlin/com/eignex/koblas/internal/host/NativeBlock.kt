package com.eignex.koblas.internal.host

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG

/** @property segment what the downcalls take, since `java.lang.foreign` is how the JVM reaches native memory. */
internal actual class NativeBlock(val segment: MemorySegment) {
    actual fun sized(bytes: Long): NativeBlock = NativeBlock(segment.reinterpret(bytes))

    actual fun putInt(offset: Long, value: Int) {
        segment.set(JAVA_INT, offset, value)
    }

    actual fun getInt(offset: Long): Int = segment.get(JAVA_INT, offset)

    actual fun putSize(offset: Long, value: Long) {
        segment.set(JAVA_LONG, offset, value)
    }

    actual fun getSize(offset: Long): Long = segment.get(JAVA_LONG, offset)

    actual fun putDouble(offset: Long, value: Double) {
        segment.set(JAVA_DOUBLE, offset, value)
    }

    actual fun getDouble(offset: Long): Double = segment.get(JAVA_DOUBLE, offset)

    actual fun putPointer(offset: Long, value: NativeBlock?) {
        segment.set(ADDRESS, offset, value?.segment ?: MemorySegment.NULL)
    }

    actual fun getPointer(offset: Long, bytes: Long): NativeBlock? {
        val target = segment.get(ADDRESS, offset)
        // A pointer a library wrote carries no length, so reading through it needs the span the caller means.
        return if (target.address() == 0L) null else NativeBlock(target.reinterpret(bytes))
    }

    actual fun readInts(count: Int): IntArray {
        val out = IntArray(count)
        MemorySegment.copy(segment, JAVA_INT, 0L, out, 0, count)
        return out
    }

    actual fun readDoubles(count: Int): DoubleArray {
        val out = DoubleArray(count)
        MemorySegment.copy(segment, JAVA_DOUBLE, 0L, out, 0, count)
        return out
    }

    actual fun writeDoubles(source: DoubleArray, count: Int) {
        MemorySegment.copy(source, 0, segment, JAVA_DOUBLE, 0L, count)
    }
}

/** [NativeBlockAllocator] over one [Arena], whose scope decides when the blocks go away. */
internal class ArenaBlocks(private val arena: Arena) : NativeBlockAllocator {
    override fun block(bytes: Long): NativeBlock = NativeBlock(arena.allocate(bytes))

    override fun ints(capacity: Int, source: IntArray, count: Int): NativeBlock {
        val out = arena.allocate(JAVA_INT, capacity.toLong())
        MemorySegment.copy(source, 0, out, JAVA_INT, 0L, count)
        return NativeBlock(out)
    }

    override fun doubles(capacity: Int, source: DoubleArray, count: Int): NativeBlock {
        val out = arena.allocate(JAVA_DOUBLE, capacity.toLong())
        MemorySegment.copy(source, 0, out, JAVA_DOUBLE, 0L, count)
        return NativeBlock(out)
    }
}
