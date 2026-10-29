package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrixRegion
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
internal fun Arena.stage(vector: DenseVector, read: Boolean = true): NativeVector {
    val base = baseIndex(vector)
    val segment = allocate(JAVA_DOUBLE, maxOf(1, vector.size).toLong())
    val step = if (vector.size <= 1) 1 else abs(vector.stride)
    if (read && vector.size > 0) {
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

/** A logical matrix region copied to one call's native arena. */
internal class NativeMatrix(
    val segment: MemorySegment,
    val leadingDimension: Int,
    private val matrix: DenseMatrixRegion,
    private val structure: MatrixStructure,
) {
    /** Writes only selected logical entries, preserving padding and the other triangle. */
    fun writeBack() {
        for (column in 0 until matrix.cols) {
            for (row in 0 until matrix.rows) {
                if (storedEntry(structure, row, column)) {
                    matrix.values[matrix.offset + row + column * matrix.leadingDimension] =
                        segment.getAtIndex(JAVA_DOUBLE, (row + column * matrix.rows).toLong())
                }
            }
        }
    }
}

/** Packs the logical region; structure controls both transfer and writeback. */
internal fun Arena.stage(
    matrix: DenseMatrixRegion,
    structure: MatrixStructure = MatrixStructure.General,
    read: Boolean = true,
): NativeMatrix {
    val span = matrix.rows.toLong() * matrix.cols
    val segment = allocate(JAVA_DOUBLE, maxOf(1L, span))
    if (read) {
        for (column in 0 until matrix.cols) {
            for (row in 0 until matrix.rows) {
                if (storedEntry(structure, row, column)) {
                    segment.setAtIndex(
                        JAVA_DOUBLE,
                        (row + column * matrix.rows).toLong(),
                        matrix.values[matrix.offset + row + column * matrix.leadingDimension],
                    )
                }
            }
        }
    }
    return NativeMatrix(segment, maxOf(1, matrix.rows), matrix, structure)
}

/** Which entries the binding may transfer, including implicit unit-diagonal exclusion. */
private fun storedEntry(structure: MatrixStructure, row: Int, column: Int): Boolean = when (structure) {
    MatrixStructure.General -> true
    MatrixStructure.SymmetricLower, MatrixStructure.TriangularLower -> row >= column
    MatrixStructure.SymmetricUpper, MatrixStructure.TriangularUpper -> row <= column
    MatrixStructure.UnitLower -> row > column
    MatrixStructure.UnitUpper -> row < column
}

/**
 * Lowest storage index passed to BLAS. Negative increments retain their sign but start at the
 * low address, allowing BLAS to reconstruct the logical first entry at the high end.
 */
internal fun baseIndex(vector: DenseVector): Int =
    if (vector.size <= 1 || vector.stride >= 0) vector.offset else vector.offset + (vector.size - 1) * vector.stride
