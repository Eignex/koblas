@file:Suppress("MatchingDeclarationName") // public adapter and raw leaf share one implementation home

package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.ScalarKernels
import com.eignex.koblas.internal.configuration.ImplementationNames
import com.eignex.koblas.requireShape

/** Scalar sparse-vector kernels used as the semantic oracle and universal fallback. */
internal object ScalarIndexedSparseKernels : IndexedSparseKernels {
    override fun dotDense(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        dense: DoubleArray,
    ): Double {
        var sum = 0.0
        for (k in fromIndex until toIndex) sum += values[k] * dense[indices[k]]
        return sum
    }

    override fun dotSparse(
        xIndices: IntArray,
        xValues: DoubleArray,
        xFromIndex: Int,
        xToIndex: Int,
        yIndices: IntArray,
        yValues: DoubleArray,
        yFromIndex: Int,
        yToIndex: Int,
    ): Double {
        var sum = 0.0
        var a = xFromIndex
        var b = yFromIndex
        while (a < xToIndex && b < yToIndex) {
            val ia = xIndices[a]
            val ib = yIndices[b]
            when {
                ia < ib -> a++

                ia > ib -> b++

                else -> {
                    sum += xValues[a] * yValues[b]
                    a++
                    b++
                }
            }
        }
        return sum
    }

    override fun axpy(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        for (k in fromIndex until toIndex) destination[indices[k]] += alpha * values[k]
    }

    override fun scatter(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        destination: DoubleArray,
    ) {
        for (k in fromIndex until toIndex) destination[indices[k]] = values[k]
    }

    override fun gather(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, source: DoubleArray) {
        for (k in fromIndex until toIndex) values[k] = source[indices[k]]
    }

    override fun gatherZero(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        source: DoubleArray,
    ) {
        for (k in fromIndex until toIndex) {
            val i = indices[k]
            values[k] = source[i]
            source[i] = 0.0
        }
    }
}

internal object ScalarSparseKernels : SparseKernels {
    override val name: String get() = ImplementationNames.SCALAR

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return ScalarIndexedSparseKernels.dotDense(x.indices, x.values, 0, x.values.size, y)
    }

    override fun dot(x: SparseVector, y: SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return ScalarIndexedSparseKernels.dotSparse(
            x.indices,
            x.values,
            0,
            x.values.size,
            y.indices,
            y.values,
            0,
            y.values.size,
        )
    }

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0) return
        ScalarIndexedSparseKernels.axpy(x.indices, x.values, 0, x.values.size, alpha, y)
    }

    override fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        ScalarIndexedSparseKernels.scatter(x.indices, x.values, 0, x.values.size, out)
    }

    override fun gather(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
        ScalarIndexedSparseKernels.gather(x.indices, x.values, 0, x.values.size, from)
    }

    override fun gatherZero(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
        ScalarIndexedSparseKernels.gatherZero(x.indices, x.values, 0, x.values.size, from)
    }

    override fun nrm2(x: SparseVector): Double = ScalarKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: SparseVector): Double = ScalarKernels.asum(x.values, 0, x.values.size)
}

internal val scalarSparseKernelFamilies: SparseKernelFamilies = sparseKernelFamilies(
    ScalarSparseKernels,
    ScalarIndexedSparseKernels,
    ScalarKernels,
    com.eignex.koblas.dense.ScalarPanelKernels,
)
