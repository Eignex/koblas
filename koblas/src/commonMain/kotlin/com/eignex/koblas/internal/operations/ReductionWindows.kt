package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.KernelProvider
import kotlin.math.abs

/* Built-in reductions already own primitive windows. Only custom engines need a typed view
 * to reach their public overrides; creating that view on the built-in path depends on escape analysis.
 */

internal fun KoblasEngine.sumWindow(values: DoubleArray, offset: Int, size: Int, stride: Int = 1): Double =
    if (this is KernelProvider) {
        sum(values, offset, size, stride)
    } else {
        sum(DenseVector.wrap(values, offset, size, stride))
    }

internal fun KoblasEngine.absoluteSumWindow(values: DoubleArray, offset: Int, size: Int, stride: Int = 1): Double =
    if (this is KernelProvider) {
        asum(values, offset, size, stride)
    } else {
        asum(DenseVector.wrap(values, offset, size, stride))
    }

internal fun KoblasEngine.euclideanNormWindow(values: DoubleArray, offset: Int, size: Int, stride: Int = 1): Double =
    if (this is KernelProvider) {
        nrm2(values, offset, size, stride)
    } else {
        nrm2(DenseVector.wrap(values, offset, size, stride))
    }

internal fun KoblasEngine.maximumMagnitude(values: DoubleArray, offset: Int, size: Int, stride: Int): Double {
    if (size == 0) return 0.0
    val index = if (this is KernelProvider) {
        iamax(values, offset, size, stride)
    } else {
        iamax(DenseVector.wrap(values, offset, size, stride))
    }
    return abs(values[offset + index * stride])
}
