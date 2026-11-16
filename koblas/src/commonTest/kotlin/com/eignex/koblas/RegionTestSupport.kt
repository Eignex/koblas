package com.eignex.koblas

import kotlin.test.assertEquals

/** The logical entries of [a] in contiguous storage of their own, which is what the oracle is handed. */
internal fun contiguousCopy(a: Matrix): DenseMatrix = DenseMatrix.wrap(
    a.rows,
    a.cols,
    DoubleArray(a.rows * a.cols) { a[it % a.rows, it / a.rows] },
)

/**
 * That [actual] holds [expected], and that every entry of its backing array outside the rectangle still
 * holds what [before] recorded, which is what catches a write into the padding.
 */
internal fun assertRegionAgreesWithReference(
    expected: DenseMatrix,
    actual: DenseMatrix,
    before: DoubleArray,
    context: String = "region",
) {
    val full = before.copyOf()
    val addressed = BooleanArray(full.size)
    for (j in 0 until actual.cols) {
        for (i in 0 until actual.rows) {
            val index = actual.offset + i + j * actual.leadingDimension
            full[index] = expected[i, j]
            addressed[index] = true
        }
    }
    for (index in full.indices) {
        if (!addressed[index] || full[index].isNaN()) {
            assertEquals(full[index], actual.values[index], "$context backing index $index")
        } else {
            assertClose(full[index], actual.values[index], "$context backing index $index")
        }
    }
}
