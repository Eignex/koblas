package com.eignex.koblas.sparse

/** Numerical leaves over trusted sorted indexed slices. Callers own shape and slice validation. */
internal interface IndexedSparseKernels {
    fun dotDense(indices: IntArray, values: DoubleArray, dense: DoubleArray): Double =
        dotDense(indices, 0, values, 0, values.size, dense)

    @Suppress("LongParameterList")
    fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double

    fun dotSparse(xIndices: IntArray, xValues: DoubleArray, yIndices: IntArray, yValues: DoubleArray): Double =
        dotSparse(xIndices, 0, xValues, 0, xValues.size, yIndices, 0, yValues, 0, yValues.size)

    @Suppress("LongParameterList")
    fun dotSparse(
        xIndices: IntArray,
        xIndexOffset: Int,
        xValues: DoubleArray,
        xValueOffset: Int,
        xCount: Int,
        yIndices: IntArray,
        yIndexOffset: Int,
        yValues: DoubleArray,
        yValueOffset: Int,
        yCount: Int,
    ): Double

    fun axpy(indices: IntArray, values: DoubleArray, alpha: Double, destination: DoubleArray) =
        axpy(indices, 0, values, 0, values.size, alpha, destination)

    @Suppress("LongParameterList")
    fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    )

    fun scatter(indices: IntArray, values: DoubleArray, destination: DoubleArray) =
        scatter(indices, 0, values, 0, values.size, destination)

    fun gatherZero(indices: IntArray, values: DoubleArray, source: DoubleArray) =
        gatherZero(indices, 0, values, 0, values.size, source)

    @Suppress("LongParameterList")
    fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    )

    @Suppress("LongParameterList")
    fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    )

    @Suppress("LongParameterList")
    fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    )

    fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double
}

// An interface default on a delegated implementation calls the scalar delegate instead of its override.
internal fun IndexedSparseKernels.gather(indices: IntArray, values: DoubleArray, source: DoubleArray) =
    gather(indices, 0, values, 0, values.size, source)
