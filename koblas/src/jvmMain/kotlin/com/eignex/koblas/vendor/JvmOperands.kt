package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
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
    /** The BLAS increment, which keeps the window's sign. */
    val increment: Int,
    private val window: VectorWindow,
    private val base: Int,
    private val span: Int,
) {
    /**
     * Copies the operand back over the storage it came from, entry by entry for a strided window.
     *
     * The span between the window's entries is not the window's to write. Copying it back would be harmless
     * for a window that owns its storage alone, and wrong for two operands of one call that interleave in one
     * array, as two rows of a column-major matrix do: each would put the other's pre-call snapshot back over
     * the result the call just produced. A unit step has no gaps, so it keeps the bulk copy.
     */
    fun writeBack() {
        if (span == 0) return
        val step = abs(window.stride)
        if (step == 1) {
            MemorySegment.copy(segment, JAVA_DOUBLE, 0L, window.data, base, span)
            return
        }
        var target = base
        var source = 0L
        repeat(window.size) {
            window.data[target] = segment.getAtIndex(JAVA_DOUBLE, source)
            target += step
            source += step
        }
    }
}

/**
 * Copies [window] into native memory.
 *
 * The copy starts at the lowest address the window reaches, not at its logical first entry, because that is
 * what BLAS expects: a negatively stepped vector is walked from the low end and the last entry reached is
 * treated as the logical first. Keeping the sign on the increment reproduces the window exactly.
 */
internal fun Arena.stage(window: VectorWindow): NativeVector {
    val base = baseIndex(window)
    val span = if (window.size == 0) 0 else (window.size - 1) * abs(window.stride) + 1
    val segment = allocate(JAVA_DOUBLE, maxOf(1, span).toLong())
    if (span > 0) MemorySegment.copy(window.data, base, segment, JAVA_DOUBLE, 0L, span)
    return NativeVector(segment, window.stride, window, base, span)
}

/** A matrix operand in native memory, either as a span of its own storage or as a packed block. */
internal class NativeMatrix(
    /** The native copy handed to BLAS. */
    val segment: MemorySegment,
    /** The leading dimension that goes with [addressing]. */
    val leadingDimension: Int,
    /** How the window reached BLAS, which decides how it comes back. */
    val addressing: Addressing,
    private val window: MatrixWindow,
    private val base: Int,
    private val span: Int,
) {
    /** Copies the operand back over the storage it came from, unpacking a staged block on the way. */
    fun writeBack() {
        if (span == 0) return
        if (addressing != Addressing.Staged) {
            MemorySegment.copy(segment, JAVA_DOUBLE, 0L, window.data, base, span)
            return
        }
        val packed = DoubleArray(span)
        MemorySegment.copy(segment, JAVA_DOUBLE, 0L, packed, 0, span)
        unstageFrom(packed, window)
    }
}

/**
 * Copies [window] into native memory under [addressing].
 *
 * A directly addressed window keeps its own leading dimension and travels as the span from its offset to its
 * far corner, padding included. A staged one is packed column-major with its structure already applied, which
 * is why a packed operand is described to BLAS as column-major whatever the call's layout turns out to be.
 *
 * For a window with an implicit unit diagonal the span copied includes the diagonal storage the operation will
 * not read. Those values cannot reach the result: the vendor is told the diagonal is implicit and does not
 * load it. Nothing is written back for an input-only operand, so that storage stays as the caller left it.
 */
internal fun Arena.stage(window: MatrixWindow, addressing: Addressing): NativeMatrix {
    if (addressing == Addressing.Staged) {
        val span = window.rows * window.columns
        val packed = DoubleArray(span)
        stageInto(window, packed)
        val segment = allocate(JAVA_DOUBLE, maxOf(1, span).toLong())
        if (span > 0) MemorySegment.copy(packed, 0, segment, JAVA_DOUBLE, 0L, span)
        return NativeMatrix(segment, maxOf(1, window.rows), addressing, window, 0, span)
    }
    val span = reachableSpan(window)
    val segment = allocate(JAVA_DOUBLE, maxOf(1, span).toLong())
    if (span > 0) MemorySegment.copy(window.data, window.offset, segment, JAVA_DOUBLE, 0L, span)
    return NativeMatrix(
        segment,
        leadingDimension(window, addressing),
        addressing,
        window,
        window.offset,
        span,
    )
}
