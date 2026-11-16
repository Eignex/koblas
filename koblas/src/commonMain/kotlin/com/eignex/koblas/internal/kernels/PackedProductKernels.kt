@file:Suppress("LongParameterList") // Primitive arithmetic windows avoid per-call descriptors.

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*

/** Packed product arithmetic and its register tile geometry. */
internal interface PackedProductKernels : KernelProvider {
    /** The bodies a [productBlock] of these extents reaches, in the order its row sweep takes them. */
    fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> =
        if (rows <= 0 || columns <= 0 || depth <= 0) emptyList() else listOf(name)

    /**
     * Destination rows one register tile accumulates, which is the group a left panel is packed in.
     *
     * Positive and fixed for the life of an implementation.
     */
    val tileRows: Int

    /** Destination columns one register tile accumulates, which is the group a right panel is packed in. */
    val tileColumns: Int

    /**
     * Whether a product of these logical extents is worth packing into this engine's tiles at all.
     *
     * Packing costs a pass over both operands and buys contiguous, tile-shaped reads for the whole product.
     * The arithmetic grows with `rows · columns · depth` and the copy with `rows · depth + depth · columns`,
     * so a product with any small extent pays for a copy it cannot amortise, and a matrix-vector shaped call
     * pays for one it can never amortise at all. Where this answers false the caller runs the product as
     * panel work over the operands where they are, which is what avoids copying both of them into tiles. It
     * is not a promise that nothing is copied: that path may still gather one strided coefficient column,
     * borrow a destination column to accumulate into, and stage an operand that shares the destination.
     *
     * The crossover is local: a tile that finishes more arithmetic per loaded value pays off a copy sooner.
     */
    fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean

    /**
     * `C = alpha · A · B + beta · C` over one `rows` by `columns` window, accumulating `depth` steps.
     *
     * [packedA] holds the left operand in [tileRows]-wide groups and [packedB] the right in
     * [tileColumns]-wide ones. Within a group, depth steps are adjacent blocks of that width, so entry
     * `(i, p)` of the left operand sits at `aOffset + (i / tileRows) · aGroupStride + p · tileRows +
     * i % tileRows`, and entry `(p, j)` of the right at `bOffset + (j / tileColumns) · bGroupStride +
     * p · tileColumns + j % tileColumns`.
     *
     * The group strides are `depth · tileRows` and `depth · tileColumns` for transient panels.
     *
     * The destination window starts at [cOffset] with columns [ldc] apart. With no rows or no columns
     * nothing happens and nothing is read. With no depth the product is empty, so the window becomes
     * `beta · C` and neither packed panel is read.
     */
    fun productBlock(
        alpha: Double,
        packedA: DoubleArray,
        aOffset: Int,
        aGroupStride: Int,
        packedB: DoubleArray,
        bOffset: Int,
        bGroupStride: Int,
        rows: Int,
        columns: Int,
        depth: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
    ): Unit
}
