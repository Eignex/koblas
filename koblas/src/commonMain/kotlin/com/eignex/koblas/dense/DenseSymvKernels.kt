package com.eignex.koblas.dense

/**
 * Adds `alpha * A * x` for one selected triangle of a symmetric column-major matrix. Scalar selection keeps
 * the ordered one-column oracle; compiled panel families may group four columns above the measured crossover.
 */
internal fun symvUpdate(
    panelKernels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    y: DoubleArray,
    lower: Boolean,
) {
    if (panelKernels !== ScalarPanelKernels && n >= DenseTuning.symvFourColumnCrossover) {
        symvFourColumnUpdate(panelKernels, alpha, a, n, x, y, lower)
    } else {
        symvColumnRangeUpdate(panelKernels, alpha, a, n, x, y, lower, 0, n)
    }
}

/** Applies ordinary symmetric column updates for `[from, until)`, preserving the scalar traversal. */
private fun symvColumnRangeUpdate(
    panelKernels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    y: DoubleArray,
    lower: Boolean,
    from: Int,
    until: Int,
) {
    var column = from
    while (column < until) {
        val diagonal = column + column * n
        val coefficient = alpha * x[column]
        val runOffset = if (lower) column + 1 else 0
        val length = if (lower) n - column - 1 else column
        y[column] += coefficient * a[diagonal]
        y[column] +=
            alpha * panelKernels.dotAxpy(y, runOffset, coefficient, a, runOffset + column * n, x, runOffset, length)
        column++
    }
}

/** Shares the common run of four adjacent symmetric columns across dot4 and axpy4 panel leaves. */
private fun symvFourColumnUpdate(
    panelKernels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    y: DoubleArray,
    lower: Boolean,
) {
    var block = 0
    val groupedEnd = n - 3
    while (block < groupedEnd) {
        val commonOffset = if (lower) block + 4 else 0
        val commonLength = if (lower) n - commonOffset else block
        val old0 = y[block]
        val old1 = y[block + 1]
        val old2 = y[block + 2]
        val old3 = y[block + 3]
        panelKernels.dot4(a, commonOffset + block * n, n, x, commonOffset, commonLength, y, block)
        val sum0 = y[block]
        val sum1 = y[block + 1]
        val sum2 = y[block + 2]
        val sum3 = y[block + 3]
        y[block] = old0
        y[block + 1] = old1
        y[block + 2] = old2
        y[block + 3] = old3

        val c0 = alpha * x[block]
        val c1 = alpha * x[block + 1]
        val c2 = alpha * x[block + 2]
        val c3 = alpha * x[block + 3]
        var dot0 = sum0
        var dot1 = sum1
        var dot2 = sum2
        var dot3 = sum3
        var reversed0 = 0.0
        var reversed1 = 0.0
        var reversed2 = 0.0
        var reversed3 = 0.0
        if (lower) {
            val t01 = a[block + 1 + block * n] * x[block + 1]
            val t02 = a[block + 2 + block * n] * x[block + 2]
            val t03 = a[block + 3 + block * n] * x[block + 3]
            val t12 = a[block + 2 + (block + 1) * n] * x[block + 2]
            val t13 = a[block + 3 + (block + 1) * n] * x[block + 3]
            val t23 = a[block + 3 + (block + 2) * n] * x[block + 3]
            dot0 += t01
            dot0 += t02
            dot0 += t03
            dot1 += t12
            dot1 += t13
            dot2 += t23
            reversed0 += t01
            reversed0 += t02
            reversed0 += t03
            reversed1 += t12
            reversed1 += t13
            reversed2 += t23
        } else {
            val t01 = a[block + (block + 1) * n] * x[block]
            val t02 = a[block + (block + 2) * n] * x[block]
            val t12 = a[block + 1 + (block + 2) * n] * x[block + 1]
            val t03 = a[block + (block + 3) * n] * x[block]
            val t13 = a[block + 1 + (block + 3) * n] * x[block + 1]
            val t23 = a[block + 2 + (block + 3) * n] * x[block + 2]
            dot1 += t01
            dot2 += t02
            dot2 += t12
            dot3 += t03
            dot3 += t13
            dot3 += t23
            reversed1 += t01
            reversed2 += t02
            reversed2 += t12
            reversed3 += t03
            reversed3 += t13
            reversed3 += t23
        }
        reversed0 += sum0
        reversed1 += sum1
        reversed2 += sum2
        reversed3 += sum3
        val unsafePartialDot =
            !finiteSymvPartial(alpha, dot0) || !finiteSymvPartial(alpha, dot1) ||
                !finiteSymvPartial(alpha, dot2) || !finiteSymvPartial(alpha, dot3) ||
                !finiteSymvPartial(alpha, reversed0) || !finiteSymvPartial(alpha, reversed1) ||
                !finiteSymvPartial(alpha, reversed2) || !finiteSymvPartial(alpha, reversed3)
        if (unsafePartialDot) {
            symvColumnRangeUpdate(panelKernels, alpha, a, n, x, y, lower, block, block + 4)
            block += 4
            continue
        }

        panelKernels.axpy4(y, commonOffset, a, commonOffset + block * n, n, c0, c1, c2, c3, commonLength)
        if (lower) {
            var column = block
            while (column < block + 4) {
                val coefficient = alpha * x[column]
                y[column] += coefficient * a[column + column * n]
                var row = column + 1
                while (row < block + 4) {
                    y[row] += coefficient * a[row + column * n]
                    row++
                }
                y[column] += alpha * when (column - block) {
                    0 -> dot0
                    1 -> dot1
                    2 -> dot2
                    else -> dot3
                }
                column++
            }
        } else {
            var column = block
            while (column < block + 4) {
                val coefficient = alpha * x[column]
                var row = block
                while (row < column) {
                    y[row] += coefficient * a[row + column * n]
                    row++
                }
                y[column] += coefficient * a[column + column * n]
                y[column] += alpha * when (column - block) {
                    0 -> dot0
                    1 -> dot1
                    2 -> dot2
                    else -> dot3
                }
                column++
            }
        }
        block += 4
    }
    symvColumnRangeUpdate(panelKernels, alpha, a, n, x, y, lower, block, n)
}

/** Both the regrouped partial and its final scaling must remain finite. */
private fun finiteSymvPartial(alpha: Double, partial: Double): Boolean =
    partial.isFinite() && (alpha * partial).isFinite()
