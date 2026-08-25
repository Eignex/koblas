package com.eignex.koblas.sparse.host

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.pow

/**
 * The per-row factors the portable factorization equilibrates with: a power of two near `1/max|row|`, so
 * scaling is exact in binary floating point and a host binding that scales for itself agrees with the
 * portable one entry for entry.
 *
 * A row is left alone when its factor would be 1.0 or would not be finite, which a row below `2^-1023`
 * would need. Reading the factors from the CSC arrays rather than a matrix keeps this usable from a binding
 * that already holds them.
 */
public fun equilibrationScale(rows: Int, rowIdx: IntArray, values: DoubleArray): DoubleArray {
    val largest = DoubleArray(rows)
    for (k in values.indices) {
        val magnitude = abs(values[k])
        if (magnitude > largest[rowIdx[k]]) largest[rowIdx[k]] = magnitude
    }
    return DoubleArray(rows) { row ->
        val magnitude = largest[row]
        if (magnitude <= 0.0) 1.0 else 2.0.pow(-floor(log2(magnitude)).toInt()).takeIf { it.isFinite() } ?: 1.0
    }
}

/** [values] with every entry multiplied by its row's factor from [scale], as a fresh array. */
public fun scaledValues(rowIdx: IntArray, values: DoubleArray, scale: DoubleArray): DoubleArray =
    DoubleArray(values.size) { values[it] * scale[rowIdx[it]] }

/**
 * Undoes the scaling [equilibrationScale] applied, in place in [x]. The factors are of `E·B`, so a forward
 * solve scales its right-hand side going in and a transposed solve scales its result coming out.
 */
public fun applyEquilibration(x: DoubleArray, scale: DoubleArray) {
    for (i in x.indices) x[i] *= scale[i]
}
