package com.eignex.koblas.sparse

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
