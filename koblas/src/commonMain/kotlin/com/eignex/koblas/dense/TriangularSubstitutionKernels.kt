@file:Suppress("LongParameterList")

package com.eignex.koblas.dense

/** Adds only source runs whose matrix coefficients are nonzero. */
private fun axpySkippingZeroSource(
    panelKernels: DensePanelKernels,
    y: DoubleArray,
    yOff: Int,
    alpha: Double,
    x: DoubleArray,
    xOff: Int,
    len: Int,
) {
    var at = 0
    while (at < len) {
        while (at < len && x[xOff + at] == 0.0) at++
        val start = at
        while (at < len && x[xOff + at] != 0.0) at++
        if (at > start) axpyArithmetic(panelKernels, y, yOff + start, alpha, x, xOff + start, at - start)
    }
}

/** Dots only matrix runs whose coefficients are nonzero. */
private fun dotSkippingZeroMatrix(
    vectorKernels: DenseVectorKernels,
    a: DoubleArray,
    aOff: Int,
    x: DoubleArray,
    xOff: Int,
    len: Int,
): Double {
    var sum = 0.0
    var at = 0
    while (at < len) {
        while (at < len && a[aOff + at] == 0.0) at++
        val start = at
        while (at < len && a[aOff + at] != 0.0) at++
        if (at > start) sum += vectorKernels.dot(a, aOff + start, x, xOff + start, at - start)
    }
    return sum
}

/**
 * Ordered triangular multiplication over the leading `n x n` triangle of [a]. The explicit traversal keeps
 * every input live until its last use, honors unit-diagonal no-read, and optionally suppresses zero-matrix
 * products for the ordinary blocked exceptional path.
 */
internal fun triangularMultiplySubstitution(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    aOff: Int = 0,
    xOff: Int = 0,
    lda: Int = n,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    guardZeroInput: Boolean = true,
    guardZeroMatrix: Boolean = false,
) {
    if (!transpose) {
        var column = if (lower) n - 1 else 0
        while (if (lower) column >= 0 else column < n) {
            val diagonal = aOff + column + column * lda
            val coefficient = x[xOff + column]
            if (!guardZeroInput || coefficient != 0.0) {
                x[xOff + column] = if (unitDiag) coefficient else a[diagonal] * coefficient
                val destination = if (lower) xOff + column + 1 else xOff
                val source = if (lower) diagonal + 1 else aOff + column * lda
                val length = if (lower) n - column - 1 else column
                if (guardZeroMatrix) {
                    axpySkippingZeroSource(panelKernels, x, destination, coefficient, a, source, length)
                } else {
                    axpyArithmetic(panelKernels, x, destination, coefficient, a, source, length)
                }
            }
            column += if (lower) -1 else 1
        }
        return
    }
    var row = if (lower) 0 else n - 1
    while (if (lower) row < n else row >= 0) {
        val diagonal = aOff + row + row * lda
        val source = if (lower) diagonal + 1 else aOff + row * lda
        val vectorOffset = if (lower) xOff + row + 1 else xOff
        val length = if (lower) n - row - 1 else row
        val diagonalProduct = if (unitDiag) x[xOff + row] else a[diagonal] * x[xOff + row]
        val offDiagonal = if (guardZeroMatrix) {
            dotSkippingZeroMatrix(vectorKernels, a, source, x, vectorOffset, length)
        } else {
            vectorKernels.dot(a, source, x, vectorOffset, length)
        }
        x[xOff + row] = diagonalProduct + offDiagonal
        row += if (lower) 1 else -1
    }
}

/**
 * Ordered triangular solve over the leading `n x n` triangle of [a]. A singular non-unit diagonal retains
 * IEEE division. The returned mask records a nonzero pivot that became exact zero after division.
 */
internal fun triangularSolveSubstitution(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    aOff: Int = 0,
    xOff: Int = 0,
    lda: Int = n,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    guardZeroPivot: Boolean = true,
    guardZeroMatrix: Boolean = false,
): Long {
    var zeroAfterNonzero = 0L
    if (!transpose) {
        var column = if (lower) 0 else n - 1
        while (if (lower) column < n else column >= 0) {
            if (!guardZeroPivot || x[xOff + column] != 0.0) {
                val diagonal = aOff + column + column * lda
                val quotient = if (unitDiag) x[xOff + column] else x[xOff + column] / a[diagonal]
                x[xOff + column] = quotient
                if (guardZeroPivot && quotient == 0.0) zeroAfterNonzero = zeroAfterNonzero or (1L shl column)
                val destination = if (lower) xOff + column + 1 else xOff
                val source = if (lower) diagonal + 1 else aOff + column * lda
                val length = if (lower) n - column - 1 else column
                if (guardZeroMatrix) {
                    axpySkippingZeroSource(panelKernels, x, destination, -quotient, a, source, length)
                } else {
                    axpyArithmetic(panelKernels, x, destination, -quotient, a, source, length)
                }
            }
            column += if (lower) 1 else -1
        }
        return zeroAfterNonzero
    }
    var row = if (lower) n - 1 else 0
    while (if (lower) row >= 0 else row < n) {
        val diagonal = aOff + row + row * lda
        val source = if (lower) diagonal + 1 else aOff + row * lda
        val vectorOffset = if (lower) xOff + row + 1 else xOff
        val length = if (lower) n - row - 1 else row
        val product = if (guardZeroMatrix) {
            dotSkippingZeroMatrix(vectorKernels, a, source, x, vectorOffset, length)
        } else {
            vectorKernels.dot(a, source, x, vectorOffset, length)
        }
        val residual = x[xOff + row] - product
        x[xOff + row] = if (unitDiag) residual else residual / a[diagonal]
        row += if (lower) -1 else 1
    }
    return zeroAfterNonzero
}
