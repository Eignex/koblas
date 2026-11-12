package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.test.assertEquals

/**
 * A [rows] by [cols] view two levels into a parent whose other entries hold a guard value, so a call that
 * addresses it as contiguous storage reads or writes the guard. The diagonal dominates, so a triangular
 * solve over the view is well conditioned.
 */
internal fun paddedRegion(rows: Int, cols: Int, seed: Int = 1): DenseMatrix {
    val parent = DenseMatrix.wrap(rows + 4, cols + 3, DoubleArray((rows + 4) * (cols + 3)) { -999.0 })
    val view = parent.view(1, 1, rows + 2, cols + 1).view(1, 1, rows, cols)
    for (j in 0 until cols) {
        for (i in 0 until rows) {
            view[i, j] = if (i == j) rows + seed.toDouble() else ((i * 3 + j * 7 + seed) % 11 - 5) / 32.0
        }
    }
    return view
}

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

/**
 * Runs [body] on each built-in composition and on an installed library admitted at every size, so each
 * fixture also reaches the library, in place where its binding takes regions.
 */
internal fun withRegionEngines(body: (KoblasEngine) -> Unit) {
    withEngines(body)
    installedHost?.let(body)
}

private val installedHost: KoblasEngine? by lazy {
    Vendor.entries.firstNotNullOfOrNull { KoblasEngine.exactVendor(it) }?.let {
        engineWithHost(engineWithKernels(ScalarEngine, ScalarEngine), it)
    }
}
