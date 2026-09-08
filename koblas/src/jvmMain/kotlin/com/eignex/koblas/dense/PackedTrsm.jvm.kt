package com.eignex.koblas.dense

import jdk.incubator.vector.DoubleVector

private val PACKED_TRSM_SPECIES = DoubleVector.SPECIES_PREFERRED
private val PACKED_TRSM_LANES = PACKED_TRSM_SPECIES.length()

/** JVM SIMD implementation of [portableTrsmTile], retaining the four RHS columns in registers. */
@Suppress("LongMethod", "LongParameterList") // the written-out four-column register tile is intentional
internal fun simdTrsmTile(
    validRows: Int,
    order: Int,
    packedTriangle: DoubleArray,
    triangleOff: Int,
    lower: Boolean,
    unitDiag: Boolean,
    x: DoubleArray,
    xOff: Int,
) {
    if (validRows == 0 || order == 0) return
    val tileRows = 2 * PACKED_TRSM_LANES
    val lowMask = PACKED_TRSM_SPECIES.indexInRange(0, validRows)
    val highMask = PACKED_TRSM_SPECIES.indexInRange(PACKED_TRSM_LANES, validRows)
    var x00 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff, lowMask)
    var x10 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + PACKED_TRSM_LANES, highMask)
    var x01 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + tileRows, lowMask)
    var x11 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + tileRows + PACKED_TRSM_LANES, highMask)
    var x02 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + 2 * tileRows, lowMask)
    var x12 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + 2 * tileRows + PACKED_TRSM_LANES, highMask)
    var x03 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + 3 * tileRows, lowMask)
    var x13 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + 3 * tileRows + PACKED_TRSM_LANES, highMask)

    if (lower) {
        if (order > 3) {
            if (!unitDiag) {
                val diagonal = packedTriangle[triangleOff + 3 * Simd.TILE_COLS + 3]
                x03 = x03.div(diagonal)
                x13 = x13.div(diagonal)
            }
            var coefficient = packedTriangle[triangleOff + 3 * Simd.TILE_COLS]
            if (coefficient != 0.0) {
                x00 = x03.mul(-coefficient).add(x00)
                x10 = x13.mul(-coefficient).add(x10)
            }
            coefficient = packedTriangle[triangleOff + 3 * Simd.TILE_COLS + 1]
            if (coefficient != 0.0) {
                x01 = x03.mul(-coefficient).add(x01)
                x11 = x13.mul(-coefficient).add(x11)
            }
            coefficient = packedTriangle[triangleOff + 3 * Simd.TILE_COLS + 2]
            if (coefficient != 0.0) {
                x02 = x03.mul(-coefficient).add(x02)
                x12 = x13.mul(-coefficient).add(x12)
            }
        }
        if (order > 2) {
            if (!unitDiag) {
                val diagonal = packedTriangle[triangleOff + 2 * Simd.TILE_COLS + 2]
                x02 = x02.div(diagonal)
                x12 = x12.div(diagonal)
            }
            var coefficient = packedTriangle[triangleOff + 2 * Simd.TILE_COLS]
            if (coefficient != 0.0) {
                x00 = x02.mul(-coefficient).add(x00)
                x10 = x12.mul(-coefficient).add(x10)
            }
            coefficient = packedTriangle[triangleOff + 2 * Simd.TILE_COLS + 1]
            if (coefficient != 0.0) {
                x01 = x02.mul(-coefficient).add(x01)
                x11 = x12.mul(-coefficient).add(x11)
            }
        }
        if (order > 1) {
            if (!unitDiag) {
                val diagonal = packedTriangle[triangleOff + Simd.TILE_COLS + 1]
                x01 = x01.div(diagonal)
                x11 = x11.div(diagonal)
            }
            val coefficient = packedTriangle[triangleOff + Simd.TILE_COLS]
            if (coefficient != 0.0) {
                x00 = x01.mul(-coefficient).add(x00)
                x10 = x11.mul(-coefficient).add(x10)
            }
        }
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff]
            x00 = x00.div(diagonal)
            x10 = x10.div(diagonal)
        }
    } else {
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff]
            x00 = x00.div(diagonal)
            x10 = x10.div(diagonal)
        }
        if (order > 1) {
            val coefficient = packedTriangle[triangleOff + 1]
            if (coefficient != 0.0) {
                x01 = x00.mul(-coefficient).add(x01)
                x11 = x10.mul(-coefficient).add(x11)
            }
            if (!unitDiag) {
                val diagonal = packedTriangle[triangleOff + Simd.TILE_COLS + 1]
                x01 = x01.div(diagonal)
                x11 = x11.div(diagonal)
            }
        }
        if (order > 2) {
            var coefficient = packedTriangle[triangleOff + 2]
            if (coefficient != 0.0) {
                x02 = x00.mul(-coefficient).add(x02)
                x12 = x10.mul(-coefficient).add(x12)
            }
            coefficient = packedTriangle[triangleOff + Simd.TILE_COLS + 2]
            if (coefficient != 0.0) {
                x02 = x01.mul(-coefficient).add(x02)
                x12 = x11.mul(-coefficient).add(x12)
            }
            if (!unitDiag) {
                val diagonal = packedTriangle[triangleOff + 2 * Simd.TILE_COLS + 2]
                x02 = x02.div(diagonal)
                x12 = x12.div(diagonal)
            }
        }
        if (order > 3) {
            var coefficient = packedTriangle[triangleOff + 3]
            if (coefficient != 0.0) {
                x03 = x00.mul(-coefficient).add(x03)
                x13 = x10.mul(-coefficient).add(x13)
            }
            coefficient = packedTriangle[triangleOff + Simd.TILE_COLS + 3]
            if (coefficient != 0.0) {
                x03 = x01.mul(-coefficient).add(x03)
                x13 = x11.mul(-coefficient).add(x13)
            }
            coefficient = packedTriangle[triangleOff + 2 * Simd.TILE_COLS + 3]
            if (coefficient != 0.0) {
                x03 = x02.mul(-coefficient).add(x03)
                x13 = x12.mul(-coefficient).add(x13)
            }
            if (!unitDiag) {
                val diagonal = packedTriangle[triangleOff + 3 * Simd.TILE_COLS + 3]
                x03 = x03.div(diagonal)
                x13 = x13.div(diagonal)
            }
        }
    }

    x00.intoArray(x, xOff, lowMask)
    x10.intoArray(x, xOff + PACKED_TRSM_LANES, highMask)
    if (order > 1) {
        x01.intoArray(x, xOff + tileRows, lowMask)
        x11.intoArray(x, xOff + tileRows + PACKED_TRSM_LANES, highMask)
    }
    if (order > 2) {
        x02.intoArray(x, xOff + 2 * tileRows, lowMask)
        x12.intoArray(x, xOff + 2 * tileRows + PACKED_TRSM_LANES, highMask)
    }
    if (order > 3) {
        x03.intoArray(x, xOff + 3 * tileRows, lowMask)
        x13.intoArray(x, xOff + 3 * tileRows + PACKED_TRSM_LANES, highMask)
    }
}

/**
 * SIMD composition of a packed update and solve. Full tiles keep the updated columns in registers through
 * the solve; edge tiles use the two independently masked kernels because their cost is not amortised.
 */
@Suppress("LongMethod", "LongParameterList") // the written-out four-column register tile is intentional
internal fun simdGemmTrsmTile(
    depth: Int,
    validRows: Int,
    order: Int,
    packedA: DoubleArray,
    aOff: Int,
    packedB: DoubleArray,
    bOff: Int,
    packedTriangle: DoubleArray,
    triangleOff: Int,
    lower: Boolean,
    unitDiag: Boolean,
    x: DoubleArray,
    xOff: Int,
) {
    val tileRows = 2 * PACKED_TRSM_LANES
    if (validRows != tileRows || order != Simd.TILE_COLS) {
        Simd.gemmTile(depth, packedA, aOff, packedB, bOff, x, xOff, tileRows)
        simdTrsmTile(validRows, order, packedTriangle, triangleOff, lower, unitDiag, x, xOff)
        return
    }

    var x00 = DoubleVector.zero(PACKED_TRSM_SPECIES)
    var x10 = DoubleVector.zero(PACKED_TRSM_SPECIES)
    var x01 = DoubleVector.zero(PACKED_TRSM_SPECIES)
    var x11 = DoubleVector.zero(PACKED_TRSM_SPECIES)
    var x02 = DoubleVector.zero(PACKED_TRSM_SPECIES)
    var x12 = DoubleVector.zero(PACKED_TRSM_SPECIES)
    var x03 = DoubleVector.zero(PACKED_TRSM_SPECIES)
    var x13 = DoubleVector.zero(PACKED_TRSM_SPECIES)
    var ap = aOff
    var bp = bOff
    for (p in 0 until depth) {
        val a0 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, packedA, ap)
        val a1 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, packedA, ap + PACKED_TRSM_LANES)
        var coefficient = DoubleVector.broadcast(PACKED_TRSM_SPECIES, packedB[bp])
        x00 = a0.fma(coefficient, x00)
        x10 = a1.fma(coefficient, x10)
        coefficient = DoubleVector.broadcast(PACKED_TRSM_SPECIES, packedB[bp + 1])
        x01 = a0.fma(coefficient, x01)
        x11 = a1.fma(coefficient, x11)
        coefficient = DoubleVector.broadcast(PACKED_TRSM_SPECIES, packedB[bp + 2])
        x02 = a0.fma(coefficient, x02)
        x12 = a1.fma(coefficient, x12)
        coefficient = DoubleVector.broadcast(PACKED_TRSM_SPECIES, packedB[bp + 3])
        x03 = a0.fma(coefficient, x03)
        x13 = a1.fma(coefficient, x13)
        ap += tileRows
        bp += Simd.TILE_COLS
    }
    x00 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff).add(x00)
    x10 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + PACKED_TRSM_LANES).add(x10)
    x01 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + tileRows).add(x01)
    x11 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + tileRows + PACKED_TRSM_LANES).add(x11)
    x02 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + 2 * tileRows).add(x02)
    x12 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + 2 * tileRows + PACKED_TRSM_LANES).add(x12)
    x03 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + 3 * tileRows).add(x03)
    x13 = DoubleVector.fromArray(PACKED_TRSM_SPECIES, x, xOff + 3 * tileRows + PACKED_TRSM_LANES).add(x13)
    solveAndStoreFullTile(
        packedTriangle, triangleOff, lower, unitDiag, x, xOff,
        x00, x10, x01, x11, x02, x12, x03, x13,
    )
}

/*
 * This must remain inline. Passing DoubleVector values through a call the JIT declines to inline boxes them
 * as heap objects, turning an allocation-free tile into per-operation allocation and a severe slowdown.
 */
@Suppress("NOTHING_TO_INLINE", "LongMethod", "LongParameterList")
private inline fun solveAndStoreFullTile(
    packedTriangle: DoubleArray,
    triangleOff: Int,
    lower: Boolean,
    unitDiag: Boolean,
    x: DoubleArray,
    xOff: Int,
    initial00: DoubleVector,
    initial10: DoubleVector,
    initial01: DoubleVector,
    initial11: DoubleVector,
    initial02: DoubleVector,
    initial12: DoubleVector,
    initial03: DoubleVector,
    initial13: DoubleVector,
) {
    var x00 = initial00
    var x10 = initial10
    var x01 = initial01
    var x11 = initial11
    var x02 = initial02
    var x12 = initial12
    var x03 = initial03
    var x13 = initial13
    if (lower) {
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff + 3 * Simd.TILE_COLS + 3]
            x03 = x03.div(diagonal)
            x13 = x13.div(diagonal)
        }
        var coefficient = packedTriangle[triangleOff + 3 * Simd.TILE_COLS]
        if (coefficient != 0.0) {
            x00 = x03.mul(-coefficient).add(x00)
            x10 = x13.mul(-coefficient).add(x10)
        }
        coefficient = packedTriangle[triangleOff + 3 * Simd.TILE_COLS + 1]
        if (coefficient != 0.0) {
            x01 = x03.mul(-coefficient).add(x01)
            x11 = x13.mul(-coefficient).add(x11)
        }
        coefficient = packedTriangle[triangleOff + 3 * Simd.TILE_COLS + 2]
        if (coefficient != 0.0) {
            x02 = x03.mul(-coefficient).add(x02)
            x12 = x13.mul(-coefficient).add(x12)
        }
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff + 2 * Simd.TILE_COLS + 2]
            x02 = x02.div(diagonal)
            x12 = x12.div(diagonal)
        }
        coefficient = packedTriangle[triangleOff + 2 * Simd.TILE_COLS]
        if (coefficient != 0.0) {
            x00 = x02.mul(-coefficient).add(x00)
            x10 = x12.mul(-coefficient).add(x10)
        }
        coefficient = packedTriangle[triangleOff + 2 * Simd.TILE_COLS + 1]
        if (coefficient != 0.0) {
            x01 = x02.mul(-coefficient).add(x01)
            x11 = x12.mul(-coefficient).add(x11)
        }
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff + Simd.TILE_COLS + 1]
            x01 = x01.div(diagonal)
            x11 = x11.div(diagonal)
        }
        coefficient = packedTriangle[triangleOff + Simd.TILE_COLS]
        if (coefficient != 0.0) {
            x00 = x01.mul(-coefficient).add(x00)
            x10 = x11.mul(-coefficient).add(x10)
        }
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff]
            x00 = x00.div(diagonal)
            x10 = x10.div(diagonal)
        }
    } else {
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff]
            x00 = x00.div(diagonal)
            x10 = x10.div(diagonal)
        }
        var coefficient = packedTriangle[triangleOff + 1]
        if (coefficient != 0.0) {
            x01 = x00.mul(-coefficient).add(x01)
            x11 = x10.mul(-coefficient).add(x11)
        }
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff + Simd.TILE_COLS + 1]
            x01 = x01.div(diagonal)
            x11 = x11.div(diagonal)
        }
        coefficient = packedTriangle[triangleOff + 2]
        if (coefficient != 0.0) {
            x02 = x00.mul(-coefficient).add(x02)
            x12 = x10.mul(-coefficient).add(x12)
        }
        coefficient = packedTriangle[triangleOff + Simd.TILE_COLS + 2]
        if (coefficient != 0.0) {
            x02 = x01.mul(-coefficient).add(x02)
            x12 = x11.mul(-coefficient).add(x12)
        }
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff + 2 * Simd.TILE_COLS + 2]
            x02 = x02.div(diagonal)
            x12 = x12.div(diagonal)
        }
        coefficient = packedTriangle[triangleOff + 3]
        if (coefficient != 0.0) {
            x03 = x00.mul(-coefficient).add(x03)
            x13 = x10.mul(-coefficient).add(x13)
        }
        coefficient = packedTriangle[triangleOff + Simd.TILE_COLS + 3]
        if (coefficient != 0.0) {
            x03 = x01.mul(-coefficient).add(x03)
            x13 = x11.mul(-coefficient).add(x13)
        }
        coefficient = packedTriangle[triangleOff + 2 * Simd.TILE_COLS + 3]
        if (coefficient != 0.0) {
            x03 = x02.mul(-coefficient).add(x03)
            x13 = x12.mul(-coefficient).add(x13)
        }
        if (!unitDiag) {
            val diagonal = packedTriangle[triangleOff + 3 * Simd.TILE_COLS + 3]
            x03 = x03.div(diagonal)
            x13 = x13.div(diagonal)
        }
    }
    val tileRows = 2 * PACKED_TRSM_LANES
    x00.intoArray(x, xOff)
    x10.intoArray(x, xOff + PACKED_TRSM_LANES)
    x01.intoArray(x, xOff + tileRows)
    x11.intoArray(x, xOff + tileRows + PACKED_TRSM_LANES)
    x02.intoArray(x, xOff + 2 * tileRows)
    x12.intoArray(x, xOff + 2 * tileRows + PACKED_TRSM_LANES)
    x03.intoArray(x, xOff + 3 * tileRows)
    x13.intoArray(x, xOff + 3 * tileRows + PACKED_TRSM_LANES)
}
