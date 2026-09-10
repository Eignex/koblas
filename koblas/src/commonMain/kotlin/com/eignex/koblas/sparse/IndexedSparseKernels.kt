package com.eignex.koblas.sparse

/** Numerical leaves over trusted sorted indexed slices. Callers own shape and slice validation. */
internal interface IndexedSparseKernels {
    fun dotDense(indices: IntArray, values: DoubleArray, dense: DoubleArray): Double =
        dotDense(indices, values, 0, values.size, dense)

    fun dotDense(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, dense: DoubleArray): Double

    fun dotSparse(xIndices: IntArray, xValues: DoubleArray, yIndices: IntArray, yValues: DoubleArray): Double =
        dotSparse(xIndices, xValues, 0, xValues.size, yIndices, yValues, 0, yValues.size)

    @Suppress("LongParameterList")
    fun dotSparse(
        xIndices: IntArray,
        xValues: DoubleArray,
        xFromIndex: Int,
        xToIndex: Int,
        yIndices: IntArray,
        yValues: DoubleArray,
        yFromIndex: Int,
        yToIndex: Int,
    ): Double

    fun axpy(indices: IntArray, values: DoubleArray, alpha: Double, destination: DoubleArray) =
        axpy(indices, values, 0, values.size, alpha, destination)

    fun axpy(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        alpha: Double,
        destination: DoubleArray,
    )

    fun scatter(indices: IntArray, values: DoubleArray, destination: DoubleArray) =
        scatter(indices, values, 0, values.size, destination)

    fun scatter(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, destination: DoubleArray)

    fun gather(indices: IntArray, values: DoubleArray, source: DoubleArray) =
        gather(indices, values, 0, values.size, source)

    fun gather(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, source: DoubleArray)

    fun gatherZero(indices: IntArray, values: DoubleArray, source: DoubleArray) =
        gatherZero(indices, values, 0, values.size, source)

    fun gatherZero(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, source: DoubleArray)
}
