package com.eignex.koblas.dense

/**
 * Side of the square tile the portable kernel holds, in both rows and columns.
 *
 * Not in [DenseTuning] with the block sizes: those are scheduling knobs, while this is the shape
 * [portableGemmTile] is written out to. Changing it means writing a different kernel.
 */
internal const val PORTABLE_TILE: Int = 4

/**
 * The portable tile: four rows by four columns held in sixteen scalar accumulators.
 *
 * This is what [Kernels.gemmTile] runs where a target has nothing better. Separate locals let a compiler
 * keep the tile in registers; an accumulator array would put it back in memory.
 */
@Suppress("LongParameterList") // two packed panels, a destination tile, and the shared depth
internal fun portableGemmTile(
    depth: Int,
    packedA: DoubleArray,
    aOff: Int,
    packedB: DoubleArray,
    bOff: Int,
    c: DoubleArray,
    cOff: Int,
    ldc: Int,
) {
    var c00 = 0.0
    var c10 = 0.0
    var c20 = 0.0
    var c30 = 0.0
    var c01 = 0.0
    var c11 = 0.0
    var c21 = 0.0
    var c31 = 0.0
    var c02 = 0.0
    var c12 = 0.0
    var c22 = 0.0
    var c32 = 0.0
    var c03 = 0.0
    var c13 = 0.0
    var c23 = 0.0
    var c33 = 0.0
    var ap = aOff
    var bp = bOff
    for (p in 0 until depth) {
        val a0 = packedA[ap]
        val a1 = packedA[ap + 1]
        val a2 = packedA[ap + 2]
        val a3 = packedA[ap + 3]
        var coefficient = packedB[bp]
        c00 += a0 * coefficient
        c10 += a1 * coefficient
        c20 += a2 * coefficient
        c30 += a3 * coefficient
        coefficient = packedB[bp + 1]
        c01 += a0 * coefficient
        c11 += a1 * coefficient
        c21 += a2 * coefficient
        c31 += a3 * coefficient
        coefficient = packedB[bp + 2]
        c02 += a0 * coefficient
        c12 += a1 * coefficient
        c22 += a2 * coefficient
        c32 += a3 * coefficient
        coefficient = packedB[bp + 3]
        c03 += a0 * coefficient
        c13 += a1 * coefficient
        c23 += a2 * coefficient
        c33 += a3 * coefficient
        ap += PORTABLE_TILE
        bp += PORTABLE_TILE
    }
    c[cOff] += c00
    c[cOff + 1] += c10
    c[cOff + 2] += c20
    c[cOff + 3] += c30
    c[cOff + ldc] += c01
    c[cOff + ldc + 1] += c11
    c[cOff + ldc + 2] += c21
    c[cOff + ldc + 3] += c31
    c[cOff + 2 * ldc] += c02
    c[cOff + 2 * ldc + 1] += c12
    c[cOff + 2 * ldc + 2] += c22
    c[cOff + 2 * ldc + 3] += c32
    c[cOff + 3 * ldc] += c03
    c[cOff + 3 * ldc + 1] += c13
    c[cOff + 3 * ldc + 2] += c23
    c[cOff + 3 * ldc + 3] += c33
}
