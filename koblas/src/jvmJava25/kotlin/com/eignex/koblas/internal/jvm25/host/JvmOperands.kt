package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.*
import com.eignex.koblas.internal.host.baseIndex
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.operations.*
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import kotlin.math.abs

/**
 * Operands copied into native memory for one call.
 *
 * Non-critical downcalls cannot accept heap segments, so measured vendor calls include this copy.
 * A confined arena closes even on exceptions; concurrent calls share no segments.
 */
internal class NativeVector(
    /** The native copy handed to BLAS. */
    val segment: MemorySegment,
    /** The packed BLAS increment, which keeps the vector's sign except for a singleton. */
    val increment: Int,
    private val values: DoubleArray,
    private val size: Int,
    private val stride: Int,
    private val base: Int,
) {
    /**
     * Write back only the vector's entries. Copying gaps in a strided span could overwrite another
     * interleaved operand's result with its pre-call snapshot. Unit strides allow a bulk copy.
     */
    fun writeBack() {
        if (size == 0) return
        val step = if (size == 1) 1 else abs(stride)
        if (step == 1) {
            MemorySegment.copy(segment, JAVA_DOUBLE, 0L, values, base, size)
            return
        }
        var target = base
        var source = 0L
        repeat(size) {
            values[target] = segment.getAtIndex(JAVA_DOUBLE, source)
            target += step
            source++
        }
    }
}

/**
 * Packs only the addressed entries of the vector window into native memory in ascending physical order.
 *
 * The copy starts at the lowest address the vector reaches, not at its logical first entry, because that is
 * what BLAS expects: a negatively stepped vector is walked from the low end and the last entry reached is
 * treated as the logical first. A signed unit increment reproduces that traversal over the packed entries.
 * A singleton uses increment one because its stride is never traversed.
 */
internal fun Arena.stageVector(
    values: DoubleArray,
    offset: Int,
    size: Int,
    stride: Int,
    read: Boolean = true,
): NativeVector {
    val base = baseIndex(offset, stride, size)
    val segment = allocate(JAVA_DOUBLE, maxOf(1, size).toLong())
    val step = if (size <= 1) 1 else abs(stride)
    if (read && size > 0) {
        if (step == 1) {
            MemorySegment.copy(values, base, segment, JAVA_DOUBLE, 0L, size)
        } else {
            var source = base
            repeat(size) { target ->
                segment.setAtIndex(JAVA_DOUBLE, target.toLong(), values[source])
                source += step
            }
        }
    }
    val increment = if (size <= 1 || stride > 0) 1 else -1
    return NativeVector(segment, increment, values, size, stride, base)
}

/** A logical matrix region copied to one call's native arena. */
internal class NativeMatrix(
    val segment: MemorySegment,
    val leadingDimension: Int,
    private val values: DoubleArray,
    private val offset: Int,
    private val rows: Int,
    private val columns: Int,
    private val sourceLeadingDimension: Int,
    private val structure: MatrixStructure,
) {
    /** Writes only selected logical entries, preserving padding and the other triangle. */
    fun writeBack() {
        for (column in 0 until columns) {
            for (row in structure.storedRowStart(column, rows) until structure.storedRowEnd(column, rows)) {
                values[offset + row + column * sourceLeadingDimension] =
                    segment.getAtIndex(JAVA_DOUBLE, (row + column * rows).toLong())
            }
        }
    }
}

/** Packs the logical region; structure controls both transfer and writeback. */
internal fun Arena.stageMatrix(
    values: DoubleArray,
    offset: Int,
    rows: Int,
    columns: Int,
    leadingDimension: Int,
    structure: MatrixStructure = MatrixStructure.General,
    read: Boolean = true,
): NativeMatrix {
    val span = rows.toLong() * columns
    val segment = allocate(JAVA_DOUBLE, maxOf(1L, span))
    if (read) {
        for (column in 0 until columns) {
            for (row in structure.storedRowStart(column, rows) until structure.storedRowEnd(column, rows)) {
                segment.setAtIndex(
                    JAVA_DOUBLE,
                    (row + column * rows).toLong(),
                    values[offset + row + column * leadingDimension],
                )
            }
        }
    }
    return NativeMatrix(segment, maxOf(1, rows), values, offset, rows, columns, leadingDimension, structure)
}
