@file:kotlin.jvm.JvmName("DenseBlas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.gerWindow
import kotlin.jvm.JvmOverloads

/**
 * `A(rowOffset + i, columnOffset + j) += alpha · x(xOffset + i) · y(yOffset + j)` for
 * `0 <= i < rows` and `0 <= j < columns`. Entries outside this rectangle are untouched.
 *
 * The destination retains [a]'s column-major leading dimension. Built-in panel kernels update it
 * directly, without copying the rectangle or allocating vector views. All matrix and vector ranges
 * are validated before writing, including when [alpha] is zero or either extent is empty.
 *
 * A vector sharing [a]'s buffer is snapshotted over only its selected range. A zero [alpha] or an empty
 * rectangle reads no entries and takes no scratch.
 */
@Suppress("LongParameterList") // the rank update and its three independently addressed windows
@JvmOverloads
public fun KoblasEngine.ger(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DenseMatrixRegion,
    rowOffset: Int,
    columnOffset: Int,
    rows: Int,
    columns: Int,
    xOffset: Int = 0,
    yOffset: Int = 0,
    scratch: Scratch? = null,
) {
    gerWindow(alpha, x, y, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset, scratch)
}
