@file:Suppress("LongParameterList") // a window includes independent vector offsets

package com.eignex.koblas.dense

import com.eignex.koblas.*
import kotlin.test.assertEquals

/** The scalar and vector compositions and the platform default, each once. */
internal fun withKoblasEngine(body: (KoblasEngine) -> Unit) {
    listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd, koblas).distinct().forEach(body)
}

internal fun referenceWindow(
    a: DenseMatrix,
    x: DoubleArray,
    y: DoubleArray,
    rows: Int,
    columns: Int,
    xOffset: Int = 1,
    yOffset: Int = 4,
): DenseMatrix {
    val expected = DenseMatrix.wrap(a.rows, a.cols, a.values.copyOf())
    val rectangle = DenseMatrix.wrap(
        rows,
        columns,
        DoubleArray(rows * columns) { index -> a[2 + index % rows, 3 + index / rows] },
    )
    ReferenceBlas.ger(
        -0.875,
        x.copyOfRange(xOffset, xOffset + rows),
        y.copyOfRange(yOffset, yOffset + columns),
        rectangle,
    )
    for (j in 0 until columns) for (i in 0 until rows) expected[2 + i, 3 + j] = rectangle[i, j]
    return expected
}

internal fun assertWindowAgreesWithReference(expected: DenseMatrix, actual: DenseMatrix, rows: Int, columns: Int) {
    assertClose(expected, actual, "windowed GER")
    for (j in 0 until actual.cols) {
        for (i in 0 until actual.rows) {
            if (i !in 2 until 2 + rows || j !in 3 until 3 + columns) {
                assertEquals(expected[i, j], actual[i, j], "outside rectangle at $i $j")
            }
        }
    }
}
