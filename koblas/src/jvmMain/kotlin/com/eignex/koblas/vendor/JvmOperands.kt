package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import kotlin.math.abs

/**
 * Operands crossing into native memory for the duration of one call.
 *
 * Every JVM vendor call copies. That is a consequence of binding without [java.lang.foreign.Linker.Option
 * .critical], which is the right trade for an opaque vendor routine but means a heap segment is not accepted,
 * so there is no in-place path to fall back to. The copy is not hidden: it is what the route's transfer
 * adapter names, and a measurement of the public call includes it.
 *
 * The arena is confined to the calling thread and closed on the way out, including on an exception, so no
 * segment outlives the call that made it and concurrent calls share nothing.
 */
internal class NativeVector(
    /** The native copy handed to BLAS. */
    val segment: MemorySegment,
    /** The BLAS increment, which keeps the vector's sign. */
    val increment: Int,
    private val vector: DenseVector,
    private val base: Int,
    private val span: Int,
) {
    /**
     * Copies the operand back over the storage it came from, entry by entry for a strided vector.
     *
     * The span between a strided vector's entries is not its to write. Copying it back would be harmless for
     * a vector that owns its storage alone, and wrong for two operands of one call that interleave in one
     * array, as two rows of a column-major matrix do: each would put the other's pre-call snapshot back over
     * the result the call just produced. A unit step has no gaps, so it keeps the bulk copy.
     */
    fun writeBack() {
        if (span == 0) return
        val step = abs(vector.stride)
        if (step == 1) {
            MemorySegment.copy(segment, JAVA_DOUBLE, 0L, vector.data, base, span)
            return
        }
        var target = base
        var source = 0L
        repeat(vector.size) {
            vector.data[target] = segment.getAtIndex(JAVA_DOUBLE, source)
            target += step
            source += step
        }
    }
}

/**
 * Copies [vector] into native memory.
 *
 * The copy starts at the lowest address the vector reaches, not at its logical first entry, because that is
 * what BLAS expects: a negatively stepped vector is walked from the low end and the last entry reached is
 * treated as the logical first. Keeping the sign on the increment reproduces the vector exactly.
 */
internal fun Arena.stage(vector: DenseVector): NativeVector {
    val base = baseIndex(vector)
    val span = if (vector.size == 0) 0 else (vector.size - 1) * abs(vector.stride) + 1
    val segment = allocate(JAVA_DOUBLE, maxOf(1, span).toLong())
    if (span > 0) MemorySegment.copy(vector.data, base, segment, JAVA_DOUBLE, 0L, span)
    return NativeVector(segment, vector.stride, vector, base, span)
}

/** A matrix operand in native memory, always the whole contiguous column-major block. */
internal class NativeMatrix(
    /** The native copy handed to BLAS. */
    val segment: MemorySegment,
    /** The leading dimension, which for contiguous column-major storage is the row count. */
    val leadingDimension: Int,
    private val matrix: DenseMatrix,
) {
    /** Copies the operand back over the storage it came from. */
    fun writeBack() {
        if (matrix.data.isEmpty()) return
        MemorySegment.copy(segment, JAVA_DOUBLE, 0L, matrix.data, 0, matrix.data.size)
    }
}

/**
 * Copies [matrix] into native memory.
 *
 * A dense matrix is its whole buffer, column-major, so the leading dimension is its row count and there is no
 * offset to honour or padding to preserve. For a matrix a call reads under an implicit unit diagonal the copy
 * still includes the diagonal storage: the vendor is told the diagonal is implicit and does not load it, and
 * nothing is written back for an input-only operand, so that storage stays as the caller left it.
 */
internal fun Arena.stage(matrix: DenseMatrix): NativeMatrix {
    val span = matrix.data.size
    val segment = allocate(JAVA_DOUBLE, maxOf(1, span).toLong())
    if (span > 0) MemorySegment.copy(matrix.data, 0, segment, JAVA_DOUBLE, 0L, span)
    return NativeMatrix(segment, maxOf(1, matrix.rows), matrix)
}

/**
 * The lowest storage index the vector touches, which is the pointer BLAS is handed.
 *
 * For a positive stride that is the vector's own offset. For a negative one it is the far end, because BLAS
 * walks a negatively stepped vector from the lowest address upward and treats the last element it reaches as
 * the logical first.
 */
internal fun baseIndex(vector: DenseVector): Int =
    if (vector.stride >= 0) vector.offset else vector.offset + (vector.size - 1) * vector.stride
