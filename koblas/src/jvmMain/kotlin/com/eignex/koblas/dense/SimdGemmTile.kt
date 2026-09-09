package com.eignex.koblas.dense

import jdk.incubator.vector.DoubleVector

/** JVM Vector API implementation of the packed GEMM microkernel. */
internal object SimdGemmTile {
    private val species = DoubleVector.SPECIES_PREFERRED
    private val lanes = species.length()

    /** Rows of C the tile covers: two vectors deep, which still leaves the loop accumulators in registers. */
    val rows: Int = 2 * lanes

    /** Columns of C the tile covers. */
    const val COLUMNS: Int = 4

    /** Accumulates [depth] steps of a packed product in registers and adds the result into C once. */
    @Suppress("LongParameterList")
    fun addProduct(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
    ) {
        var c00 = DoubleVector.zero(species)
        var c10 = DoubleVector.zero(species)
        var c01 = DoubleVector.zero(species)
        var c11 = DoubleVector.zero(species)
        var c02 = DoubleVector.zero(species)
        var c12 = DoubleVector.zero(species)
        var c03 = DoubleVector.zero(species)
        var c13 = DoubleVector.zero(species)
        var ap = aOff
        var bp = bOff
        for (p in 0 until depth) {
            val a0 = DoubleVector.fromArray(species, packedA, ap)
            val a1 = DoubleVector.fromArray(species, packedA, ap + lanes)
            var coefficient = DoubleVector.broadcast(species, packedB[bp])
            c00 = a0.fma(coefficient, c00)
            c10 = a1.fma(coefficient, c10)
            coefficient = DoubleVector.broadcast(species, packedB[bp + 1])
            c01 = a0.fma(coefficient, c01)
            c11 = a1.fma(coefficient, c11)
            coefficient = DoubleVector.broadcast(species, packedB[bp + 2])
            c02 = a0.fma(coefficient, c02)
            c12 = a1.fma(coefficient, c12)
            coefficient = DoubleVector.broadcast(species, packedB[bp + 3])
            c03 = a0.fma(coefficient, c03)
            c13 = a1.fma(coefficient, c13)
            ap += rows
            bp += COLUMNS
        }
        addColumn(c, cOff, ldc, 0, c00, c10)
        addColumn(c, cOff, ldc, 1, c01, c11)
        addColumn(c, cOff, ldc, 2, c02, c12)
        addColumn(c, cOff, ldc, 3, c03, c13)
    }

    /** Subtracts a full packed product from C while keeping every intermediate tile value in registers. */
    @Suppress("LongParameterList")
    fun subtractProduct(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
    ) {
        var c00 = DoubleVector.fromArray(species, c, cOff)
        var c10 = DoubleVector.fromArray(species, c, cOff + lanes)
        var c01 = DoubleVector.fromArray(species, c, cOff + rows)
        var c11 = DoubleVector.fromArray(species, c, cOff + rows + lanes)
        var c02 = DoubleVector.fromArray(species, c, cOff + 2 * rows)
        var c12 = DoubleVector.fromArray(species, c, cOff + 2 * rows + lanes)
        var c03 = DoubleVector.fromArray(species, c, cOff + 3 * rows)
        var c13 = DoubleVector.fromArray(species, c, cOff + 3 * rows + lanes)
        var ap = aOff
        var bp = bOff
        for (p in 0 until depth) {
            val a0 = DoubleVector.fromArray(species, packedA, ap)
            val a1 = DoubleVector.fromArray(species, packedA, ap + lanes)
            var coefficient = DoubleVector.broadcast(species, -packedB[bp])
            c00 = a0.fma(coefficient, c00)
            c10 = a1.fma(coefficient, c10)
            coefficient = DoubleVector.broadcast(species, -packedB[bp + 1])
            c01 = a0.fma(coefficient, c01)
            c11 = a1.fma(coefficient, c11)
            coefficient = DoubleVector.broadcast(species, -packedB[bp + 2])
            c02 = a0.fma(coefficient, c02)
            c12 = a1.fma(coefficient, c12)
            coefficient = DoubleVector.broadcast(species, -packedB[bp + 3])
            c03 = a0.fma(coefficient, c03)
            c13 = a1.fma(coefficient, c13)
            ap += rows
            bp += COLUMNS
        }
        storeColumn(c, cOff, 0, c00, c10)
        storeColumn(c, cOff, 1, c01, c11)
        storeColumn(c, cOff, 2, c02, c12)
        storeColumn(c, cOff, 3, c03, c13)
    }

    /** Subtracts a logical edge without reading or writing its padded output lanes. */
    @Suppress("LongParameterList")
    fun subtractProductEdge(
        depth: Int,
        validRows: Int,
        validColumns: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
    ) {
        val lowMask = species.indexInRange(0, validRows)
        val highMask = species.indexInRange(lanes, validRows)
        var c00 = DoubleVector.fromArray(species, c, cOff, lowMask)
        var c10 = DoubleVector.fromArray(species, c, cOff + lanes, highMask)
        var c01 = if (validColumns > 1) {
            DoubleVector.fromArray(species, c, cOff + rows, lowMask)
        } else {
            DoubleVector.zero(species)
        }
        var c11 = if (validColumns > 1) {
            DoubleVector.fromArray(species, c, cOff + rows + lanes, highMask)
        } else {
            DoubleVector.zero(species)
        }
        var c02 = if (validColumns > 2) {
            DoubleVector.fromArray(species, c, cOff + 2 * rows, lowMask)
        } else {
            DoubleVector.zero(species)
        }
        var c12 = if (validColumns > 2) {
            DoubleVector.fromArray(species, c, cOff + 2 * rows + lanes, highMask)
        } else {
            DoubleVector.zero(species)
        }
        var c03 = if (validColumns > 3) {
            DoubleVector.fromArray(species, c, cOff + 3 * rows, lowMask)
        } else {
            DoubleVector.zero(species)
        }
        var c13 = if (validColumns > 3) {
            DoubleVector.fromArray(species, c, cOff + 3 * rows + lanes, highMask)
        } else {
            DoubleVector.zero(species)
        }
        var ap = aOff
        var bp = bOff
        for (p in 0 until depth) {
            val a0 = DoubleVector.fromArray(species, packedA, ap)
            val a1 = DoubleVector.fromArray(species, packedA, ap + lanes)
            var coefficient = DoubleVector.broadcast(species, -packedB[bp])
            c00 = a0.fma(coefficient, c00)
            c10 = a1.fma(coefficient, c10)
            if (validColumns > 1) {
                coefficient = DoubleVector.broadcast(species, -packedB[bp + 1])
                c01 = a0.fma(coefficient, c01)
                c11 = a1.fma(coefficient, c11)
            }
            if (validColumns > 2) {
                coefficient = DoubleVector.broadcast(species, -packedB[bp + 2])
                c02 = a0.fma(coefficient, c02)
                c12 = a1.fma(coefficient, c12)
            }
            if (validColumns > 3) {
                coefficient = DoubleVector.broadcast(species, -packedB[bp + 3])
                c03 = a0.fma(coefficient, c03)
                c13 = a1.fma(coefficient, c13)
            }
            ap += rows
            bp += COLUMNS
        }
        c00.intoArray(c, cOff, lowMask)
        c10.intoArray(c, cOff + lanes, highMask)
        if (validColumns > 1) {
            c01.intoArray(c, cOff + rows, lowMask)
            c11.intoArray(c, cOff + rows + lanes, highMask)
        }
        if (validColumns > 2) {
            c02.intoArray(c, cOff + 2 * rows, lowMask)
            c12.intoArray(c, cOff + 2 * rows + lanes, highMask)
        }
        if (validColumns > 3) {
            c03.intoArray(c, cOff + 3 * rows, lowMask)
            c13.intoArray(c, cOff + 3 * rows + lanes, highMask)
        }
    }

    // The vectors must stay in the caller so a small-depth tile does not materialize them on the heap.
    @Suppress("NOTHING_TO_INLINE")
    private inline fun addColumn(
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
        column: Int,
        low: DoubleVector,
        high: DoubleVector,
    ) {
        val base = cOff + column * ldc
        DoubleVector.fromArray(species, c, base).add(low).intoArray(c, base)
        DoubleVector.fromArray(species, c, base + lanes).add(high).intoArray(c, base + lanes)
    }

    // The vectors must stay in the caller so a small-depth fused tile does not materialize them on the heap.
    @Suppress("NOTHING_TO_INLINE")
    private inline fun storeColumn(c: DoubleArray, cOff: Int, column: Int, low: DoubleVector, high: DoubleVector) {
        val base = cOff + column * rows
        low.intoArray(c, base)
        high.intoArray(c, base + lanes)
    }
}
