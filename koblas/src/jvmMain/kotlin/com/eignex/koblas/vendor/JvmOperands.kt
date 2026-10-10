package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import kotlin.math.abs

/**
 * Operands copied into native memory for one call.
 *
 * Non-critical downcalls cannot accept heap segments. The route names the transfer adapter so
 * measurements include this copy. A confined arena closes even on exceptions; concurrent calls
 * share no segments.
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
     * Write back only the vector's entries. Copying gaps in a strided span could overwrite another
     * interleaved operand's result with its pre-call snapshot. Unit strides allow a bulk copy.
     */
    fun writeBack() {
        if (span == 0) return
        val step = abs(vector.stride)
        if (step == 1) {
            MemorySegment.copy(segment, JAVA_DOUBLE, 0L, vector.values, base, span)
            return
        }
        var target = base
        var source = 0L
        repeat(vector.size) {
            vector.values[target] = segment.getAtIndex(JAVA_DOUBLE, source)
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
    if (span > 0) MemorySegment.copy(vector.values, base, segment, JAVA_DOUBLE, 0L, span)
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
        if (matrix.values.isEmpty()) return
        MemorySegment.copy(segment, JAVA_DOUBLE, 0L, matrix.values, 0, matrix.values.size)
    }
}

/**
 * Copy the contiguous column-major buffer with the row count as leading dimension. Implicit
 * unit diagonals remain in the copy, but the vendor's diagonal flag prevents reading them;
 * input-only operands are not written back.
 */
internal fun Arena.stage(matrix: DenseMatrix): NativeMatrix {
    val span = matrix.values.size
    val segment = allocate(JAVA_DOUBLE, maxOf(1, span).toLong())
    if (span > 0) MemorySegment.copy(matrix.values, 0, segment, JAVA_DOUBLE, 0L, span)
    return NativeMatrix(segment, maxOf(1, matrix.rows), matrix)
}

/**
 * Lowest storage index passed to BLAS. Negative increments retain their sign but start at the
 * low address, allowing BLAS to reconstruct the logical first entry at the high end.
 */
internal fun baseIndex(vector: DenseVector): Int =
    if (vector.stride >= 0) vector.offset else vector.offset + (vector.size - 1) * vector.stride
