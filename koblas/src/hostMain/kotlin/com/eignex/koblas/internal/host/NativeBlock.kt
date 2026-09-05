@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.internal.host

import kotlinx.cinterop.*

/** @property pointer what the cinterop declarations take, since that is how Kotlin/Native reaches memory. */
internal actual class NativeBlock(val pointer: CPointer<ByteVar>) {
    /** A pointer already spans whatever is behind it, so the intended length changes nothing here. */
    actual fun sized(bytes: Long): NativeBlock = this

    actual fun putInt(offset: Long, value: Int) {
        field<IntVar>(offset).value = value
    }

    actual fun getInt(offset: Long): Int = field<IntVar>(offset).value

    actual fun putSize(offset: Long, value: Long) {
        field<LongVar>(offset).value = value
    }

    actual fun getSize(offset: Long): Long = field<LongVar>(offset).value

    actual fun putDouble(offset: Long, value: Double) {
        field<DoubleVar>(offset).value = value
    }

    actual fun getDouble(offset: Long): Double = field<DoubleVar>(offset).value

    actual fun putPointer(offset: Long, value: NativeBlock?) {
        field<COpaquePointerVar>(offset).value = value?.pointer
    }

    actual fun getPointer(offset: Long, bytes: Long): NativeBlock? =
        field<COpaquePointerVar>(offset).value?.let { NativeBlock(it.reinterpret()) }

    actual fun readInts(count: Int): IntArray {
        val ints = pointer.reinterpret<IntVar>()
        return IntArray(count) { ints[it] }
    }

    actual fun readDoubles(count: Int): DoubleArray {
        val doubles = pointer.reinterpret<DoubleVar>()
        return DoubleArray(count) { doubles[it] }
    }

    actual fun readDoublesInto(destination: DoubleArray, count: Int) {
        val doubles = pointer.reinterpret<DoubleVar>()
        for (i in 0 until count) destination[i] = doubles[i]
    }

    actual fun writeDoubles(source: DoubleArray, count: Int) {
        val doubles = pointer.reinterpret<DoubleVar>()
        for (i in 0 until count) doubles[i] = source[i]
    }

    private inline fun <reified T : CVariable> field(offset: Long): T = (pointer + offset)!!.reinterpret<T>().pointed
}

/** [NativeBlockAllocator] over the process heap, whose blocks the caller frees itself. */
internal object HeapBlocks : NativeBlockAllocator {
    override fun block(bytes: Long): NativeBlock = NativeBlock(nativeHeap.allocArray(bytes))

    override fun ints(capacity: Int, source: IntArray, count: Int): NativeBlock {
        val out = nativeHeap.allocArray<IntVar>(capacity)
        for (i in 0 until count) out[i] = source[i]
        return NativeBlock(out.reinterpret())
    }

    override fun doubles(capacity: Int, source: DoubleArray, count: Int): NativeBlock {
        val out = nativeHeap.allocArray<DoubleVar>(capacity)
        for (i in 0 until count) out[i] = source[i]
        return NativeBlock(out.reinterpret())
    }
}

/** [NativeBlockAllocator] over a [MemScope], whose blocks go away when the scope does. */
internal class ScopedBlocks(private val scope: MemScope) : NativeBlockAllocator {
    override fun block(bytes: Long): NativeBlock = NativeBlock(scope.allocArray(bytes))

    override fun ints(capacity: Int, source: IntArray, count: Int): NativeBlock {
        val out = scope.allocArray<IntVar>(capacity)
        for (i in 0 until count) out[i] = source[i]
        return NativeBlock(out.reinterpret())
    }

    override fun doubles(capacity: Int, source: DoubleArray, count: Int): NativeBlock {
        val out = scope.allocArray<DoubleVar>(capacity)
        for (i in 0 until count) out[i] = source[i]
        return NativeBlock(out.reinterpret())
    }
}
