package com.eignex.koblas.internal.jvm25

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
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
    /** The packed BLAS increment, which keeps the vector's sign except for a singleton. */
    val increment: Int,
    private val vector: DenseVector,
    private val base: Int,
) {
    /**
     * Write back only the vector's entries. Copying gaps in a strided span could overwrite another
     * interleaved operand's result with its pre-call snapshot. Unit strides allow a bulk copy.
     */
    fun writeBack() {
        if (vector.size == 0) return
        val step = if (vector.size == 1) 1 else abs(vector.stride)
        if (step == 1) {
            MemorySegment.copy(segment, JAVA_DOUBLE, 0L, vector.values, base, vector.size)
            return
        }
        var target = base
        var source = 0L
        repeat(vector.size) {
            vector.values[target] = segment.getAtIndex(JAVA_DOUBLE, source)
            target += step
            source++
        }
    }
}

/**
 * Packs only the addressed entries of [vector] into native memory in ascending physical order.
 *
 * The copy starts at the lowest address the vector reaches, not at its logical first entry, because that is
 * what BLAS expects: a negatively stepped vector is walked from the low end and the last entry reached is
 * treated as the logical first. A signed unit increment reproduces that traversal over the packed entries.
 * A singleton uses increment one because its stride is never traversed.
 */
internal fun Arena.stage(vector: DenseVector): NativeVector {
    val base = baseIndex(vector)
    val segment = allocate(JAVA_DOUBLE, maxOf(1, vector.size).toLong())
    val step = if (vector.size <= 1) 1 else abs(vector.stride)
    if (vector.size > 0) {
        if (step == 1) {
            MemorySegment.copy(vector.values, base, segment, JAVA_DOUBLE, 0L, vector.size)
        } else {
            var source = base
            repeat(vector.size) { target ->
                segment.setAtIndex(JAVA_DOUBLE, target.toLong(), vector.values[source])
                source += step
            }
        }
    }
    val increment = if (vector.size <= 1 || vector.stride > 0) 1 else -1
    return NativeVector(segment, increment, vector, base)
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
    if (vector.size <= 1 || vector.stride >= 0) vector.offset else vector.offset + (vector.size - 1) * vector.stride
