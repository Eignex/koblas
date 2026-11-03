@file:Suppress("TooManyFunctions", "LongParameterList")

package com.eignex.koblas.internal.portable

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.engine.PanelWork
import com.eignex.koblas.internal.portable.dense.*
import com.eignex.koblas.internal.portable.dense.packsProductByWork
import com.eignex.koblas.internal.portable.sparse.*

/** Complete portable arithmetic, independent of every accelerated provider. */
internal object PortableEngine : KoblasEngine {
    override val name: String = "scalar"
    override val id: String = "portable"
    override val implementations = EngineImplementation.entries.toSet()

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = scalarDot(a, aOff, aStride, b, bOff, bStride, len)

    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ): Unit = scalarAxpy(y, yOff, yStride, alpha, x, xOff, xStride, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int): Unit =
        scalarScale(v, vOff, vStride, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = euclideanNorm(v, vOff, vStride, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = absoluteSum(v, vOff, vStride, len)

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int = scalarIamax(v, vOff, vStride, len)

    override fun rot(
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        len: Int,
        c: Double,
        s: Double,
        xStride: Int,
        yStride: Int,
    ) = scalarRot(x, xOff, y, yOff, len, c, s, xStride, yStride)

    override fun swap(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Unit = scalarSwap(a, aOff, aStride, b, bOff, bStride, len)

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        scalarCopy(a, aOff, b, bOff, len, aStride, bStride)

    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = scalarSum(v, vOff, vStride, len)

    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = indexedDotDense(indices, indexOffset, values, valueOffset, count, dense)

    override fun dotSparse(
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
    ): Double = indexedDotSparse(
        xIndices, xIndexOffset, xValues, xValueOffset, xCount, yIndices, yIndexOffset, yValues,
        yValueOffset, yCount,
    )

    override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ) = indexedAxpy(indices, indexOffset, values, valueOffset, count, alpha, destination)

    override fun computeIndexedScatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) = indexedScatter(indices, indexOffset, values, valueOffset, count, destination)

    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) = indexedGather(indices, indexOffset, values, valueOffset, count, source)

    override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) = indexedGatherZero(indices, indexOffset, values, valueOffset, count, source)

    override fun computeIndexedNrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        indexedNrm2(indices, indexOffset, count, values)

    /** Share each vector load across four dot/update columns, or two coupled/rank columns. */
    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int {
        val preferred = when {
            work == PanelWork.CoupledDotUpdate || work == PanelWork.RankUpdate -> PANEL_NARROW_GROUP
            work == PanelWork.SparseRightHandSides -> SPARSE_RHS_GROUP
            work == PanelWork.SparseRightHandSideReduction && !contiguous -> SPARSE_REDUCTION_GROUP
            work == PanelWork.SparseRightHandSideReduction -> SPARSE_RHS_GROUP
            else -> PANEL_WIDE_GROUP
        }
        return if (columns <= 0) 1 else minOf(preferred, columns)
    }

    /** Scalar access gains nothing from staging a strided window. */
    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean = false

    override fun multiDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) = scalarMultiDot(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, beta, y, yOffset, yStride)

    override fun columnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) = scalarColumnUpdate(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset, yStride)

    override fun coupledUpdateDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
        xStride: Int,
        yStride: Int,
        coefficientStride: Int,
        sumStride: Int,
    ) = scalarCoupledUpdateDot(
        alpha, a, aOffset, lda, x, xOffset, rows, columns, y, yOffset, coefficients, coefficientOffset,
        sums, sumOffset, xStride, yStride, coefficientStride, sumStride,
    )

    override fun rankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        coefficientStride: Int,
    ) = scalarRankUpdate(
        alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, coefficients, coefficientOffset,
        coefficientStride,
    )

    override fun rankTwoUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        rows: Int,
        columns: Int,
        u: DoubleArray,
        uOffset: Int,
        uStride: Int,
        v: DoubleArray,
        vOffset: Int,
        vStride: Int,
    ) = scalarRankTwoUpdate(
        alpha, a, aOffset, lda, x, xOffset, xStride, y, yOffset, yStride, rows, columns, u, uOffset,
        uStride, v, vOffset, vStride,
    )

    override fun indexedColumnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
    ) = sparseColumnUpdate(
        alpha, a, aOffset, rowStride, indexStride, indices, values, fromIndex, columns, rows, y, yOffset,
    )

    override fun indexedRankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        x: DoubleArray,
        xOffset: Int,
    ) = sparseRankUpdate(
        alpha, a, aOffset, rowStride, indexStride, indices, values, fromIndex, columns, rows, x, xOffset,
    )

    override fun indexedCoupledUpdate(
        alpha: Double,
        a: DoubleArray,
        b: DoubleArray,
        offset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        pivot: Int,
        sums: DoubleArray,
        sumOffset: Int,
        excluded: Int,
        destinationOffset: Int,
        destinationRowStride: Int,
        destinationIndexStride: Int,
    ) = sparseCoupledUpdate(
        alpha, a, b, offset, rowStride, indexStride, indices, values, fromIndex, columns, rows, pivot, sums,
        sumOffset, excluded, destinationOffset, destinationRowStride, destinationIndexStride,
    )

    override val tileRows: Int get() = PORTABLE_PRODUCT_TILE

    override val tileColumns: Int get() = PORTABLE_PRODUCT_TILE

    override fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean =
        packsProductByWork(rows, columns, depth, tileRows, tileColumns)

    override fun productBlock(
        alpha: Double,
        packedA: DoubleArray,
        aOffset: Int,
        aGroupStride: Int,
        packedB: DoubleArray,
        bOffset: Int,
        bGroupStride: Int,
        rows: Int,
        columns: Int,
        depth: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
    ) = scalarProductBlock(
        alpha, packedA, aOffset, aGroupStride, packedB, bOffset, bGroupStride, rows, columns, depth, beta,
        c, cOffset, ldc,
    )

    override fun rightHandSideGroup(order: Int, sides: Int): Int = if (sides > 0) sides else 1

    override fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean = false

    override fun diagonalSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int,
    ) = scalarDiagonalSolve(
        t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
    )

    override fun diagonalMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int,
    ) = scalarDiagonalMultiply(
        t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
    )

    override fun computeGemv(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ): Unit = portableGemv(alpha, a, x.values, x.offset, x.stride, beta, y.values, y.offset, y.stride, transposeA)

    override fun computeSymv(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ): Unit =
        portableSymv(alpha, a, x.values, x.offset, x.stride, beta, y.values, y.offset, y.stride, structure.isLower)

    override fun computeGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion): Unit =
        portableGer(alpha, x.values, x.offset, x.stride, y.values, y.offset, y.stride, a, a.offset, a.rows, a.cols)

    override fun computeSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure): Unit =
        portableSyr(alpha, x, a, structure.isLower)

    override fun computeSyr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
    ): Unit = portableSyr2(alpha, x, y, a, structure.isLower)

    override fun computeTrsv(
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
    ): Unit = portableTrsv(a, x.values, x.offset, x.stride, structure.isLower, transposeA, structure.isUnitDiagonal)

    override fun computeTrmv(
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
    ): Unit = portableTrmv(a, x.values, x.offset, x.stride, structure.isLower, transposeA, structure.isUnitDiagonal)

    override fun computeGemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
    ): Unit = portableGemm(alpha, a, transposeA, b, transposeB, beta, c, if (transposeA) a.rows else a.cols, null)

    override fun computeSymm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        rightSide: Boolean,
    ): Unit = portableSymm(alpha, a, b, beta, c, structure.isLower, rightSide, null)

    override fun computeSyrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ): Unit = portableSyrk(alpha, a, transposeA, beta, c, structure.isLower, null)

    override fun computeSyr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ): Unit = portableSyr2k(alpha, a, b, transposeA, beta, c, structure.isLower, null)

    override fun computeTrmm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ): Unit = portableTriangularMatrix(alpha, a, structure, transposeA, b, rightSide, false, null)

    override fun computeTrsm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ): Unit = portableTriangularMatrix(alpha, a, structure, transposeA, b, rightSide, true, null)

    override fun computeGemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ): Unit = portableGemmt(alpha, a, transposeA, b, transposeB, beta, c, structure.isLower, null)
}
