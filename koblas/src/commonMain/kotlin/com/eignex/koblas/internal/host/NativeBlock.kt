package com.eignex.koblas.internal.host

/**
 * A span of native memory addressed by byte offset, which is what a C struct is to a binding that has no
 * struct type of its own.
 *
 * The libraries koblas binds take their operands as structs, and both platforms fill the same fields at the
 * same offsets by two different mechanisms: `java.lang.foreign` on the JVM, cinterop pointers on
 * Kotlin/Native. This is that mechanism behind one seam, so a struct layout is written down once and read
 * back once rather than in each binding twice over.
 *
 * A block carries no length of its own on either platform, so a read that needs one says how far it intends
 * to read: [sized] for a block a call handed back, [getPointer] for one a field points at. Allocation and
 * ownership stay with the platform, behind [NativeBlockAllocator].
 */
internal expect class NativeBlock {
    /** The same block, known to span [bytes], for a descriptor a call returned without a length. */
    fun sized(bytes: Long): NativeBlock

    /** Writes one `int` field. */
    fun putInt(offset: Long, value: Int)

    /** Reads one `int` field. */
    fun getInt(offset: Long): Int

    /** Writes one `size_t` field. */
    fun putSize(offset: Long, value: Long)

    /** Reads one `size_t` field. */
    fun getSize(offset: Long): Long

    /** Writes one `double` field. */
    fun putDouble(offset: Long, value: Double)

    /** Reads one `double` field. */
    fun getDouble(offset: Long): Double

    /** Writes one pointer field, null included. */
    fun putPointer(offset: Long, value: NativeBlock?)

    /** Reads one pointer field as a block spanning [bytes], or null where the library left it unset. */
    fun getPointer(offset: Long, bytes: Long): NativeBlock?

    /** The first [count] ints this block holds. */
    fun readInts(count: Int): IntArray

    /** The first [count] doubles this block holds. */
    fun readDoubles(count: Int): DoubleArray

    /** The first [count] doubles this block holds, into [destination], allocating nothing. */
    fun readDoublesInto(destination: DoubleArray, count: Int)

    /** Writes [count] doubles from [source] into this block. */
    fun writeDoubles(source: DoubleArray, count: Int)
}

/**
 * Where a binding gets its native memory, which is the one part of struct marshalling that stays platform
 * specific: the JVM has arenas with a scope, Kotlin/Native has `malloc` and a scoped allocator over it.
 */
internal interface NativeBlockAllocator {
    /** [bytes] of memory for a struct. */
    fun block(bytes: Long): NativeBlock

    /** Room for [capacity] ints, holding the first [count] of [source]. */
    fun ints(capacity: Int, source: IntArray, count: Int): NativeBlock

    /** Room for [capacity] doubles, holding the first [count] of [source]. */
    fun doubles(capacity: Int, source: DoubleArray, count: Int): NativeBlock
}
