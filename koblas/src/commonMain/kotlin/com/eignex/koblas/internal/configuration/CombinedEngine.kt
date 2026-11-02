@file:Suppress("LongParameterList") // Forward raw primitive windows without allocating call descriptions.

package com.eignex.koblas.internal.configuration

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.dense.*
import com.eignex.koblas.vendor.Vendor

/** Backends are bound once; arithmetic never resolves identifiers or capabilities. */
internal class CompiledRouting(profile: EngineProfile, engines: List<KoblasEngine>) {
    val backends = engines.flatMap { it.providers }.filterNot { it === PortableEngine }.distinct() + PortableEngine
    init {
        val known = backends.map { it.id } + "jvm-simd" + Vendor.entries.map { it.vendorName.lowercase() }
        require(profile.rules.all { it.source in known }) { "Unknown engine source in ${profile.name}" }
    }
    private val routes = Array(EngineImplementation.entries.size) { index ->
        val operation = EngineImplementation.entries[index]
        profile.rules.mapNotNull { rule ->
            backends.firstOrNull { it.id == rule.source && operation in it.implementations }
                ?.takeIf { operation in rule.implementations }
                ?.let { rule to it }
        }.toTypedArray()
    }

    private val lengths = Array(EngineImplementation.entries.size) { index ->
        EngineImplementation.entries[index].takeIf { it.vector }?.let(::compileLength)
    }

    /** Fold all vector-invariant shape/work constraints before arithmetic reaches a length check. */
    private fun compileLength(operation: EngineImplementation): LengthDispatch {
        var result = LengthDispatch(PortableEngine, 0, Int.MAX_VALUE, 3, null)
        for ((rule, backend) in routes[operation.ordinal].reversed()) {
            val minimum = backend.minimumLength(operation)
            val lower = maxOf(rule.rows.first, minimum, rule.work.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
            val strided = !backend.requiresUnitStride(operation)
            val mask = (if (rule.matches(lower, 1, 1, true, false)) 1 else 0) or
                (if (strided && rule.matches(lower, 1, 1, false, false)) 2 else 0)
            if (mask != 0) {
                result = LengthDispatch(
                    backend,
                    lower,
                    rule.rows.last,
                    mask,
                    result,
                )
            }
        }
        return result
    }

    fun select(
        operation: EngineImplementation,
        rows: Int = 0,
        columns: Int = 1,
        depth: Int = 1,
        contiguous: Boolean = true,
        borrowed: Boolean = false,
    ): KoblasEngine {
        lengths[operation.ordinal]?.let { return it.select(rows, contiguous) }
        for ((rule, backend) in routes[operation.ordinal]) {
            if (rule.matches(rows, columns, depth, contiguous, borrowed)) return backend
        }
        return PortableEngine
    }
}

/** Primitive length and layout checks over the backends bound to one operation. */
internal class LengthDispatch(
    private val backend: KoblasEngine,
    private val minimum: Int,
    private val maximum: Int,
    private val layouts: Int,
    private val fallback: LengthDispatch?,
) {
    fun select(length: Int, contiguous: Boolean = true): KoblasEngine {
        if (fallback == null) return backend
        return if (length >= minimum && length <= maximum && layouts and (if (contiguous) 1 else 2) != 0) {
            backend
        } else {
            fallback.select(length, contiguous)
        }
    }
}

/** One primitive-window interface serves leaves and the immutable routed composition. */
internal class CombinedEngine(private val routing: CompiledRouting, override val name: String) : KoblasEngine {
    internal val components get() = routing.backends
    override val implementations = routing.backends.flatMapTo(mutableSetOf()) { it.implementations }
    internal fun wholeCallFor(
        operation: EngineImplementation,
        rows: Int,
        columns: Int,
        depth: Int,
        borrowed: Boolean,
    ): KoblasEngine? = routing.select(
        operation,
        rows,
        columns,
        depth,
        borrowed = borrowed,
    ).wholeCallFor(operation, rows, columns, depth, borrowed)
    internal val panelImplementation = routing.select(EngineImplementation.Panels)
    internal val productImplementation = routing.select(EngineImplementation.Products)
    internal val triangularImplementation = routing.select(EngineImplementation.Triangular)
    internal fun vectors(operation: EngineImplementation, length: Int, contiguous: Boolean = true): KoblasEngine =
        routing.select(operation, length, contiguous = contiguous)

    internal fun implementationFor(operation: EngineImplementation, length: Int, contiguous: Boolean): String? =
        routing.select(operation, length, contiguous = contiguous).implementationFor(operation, length, contiguous)

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = vectors(EngineImplementation.Dot, len, aStride == 1 && bStride == 1)
        .dot(a, aOff, b, bOff, len, aStride, bStride)
    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        vectors(EngineImplementation.Sum, len, vStride == 1).sum(v, vOff, len, vStride)
    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        vectors(EngineImplementation.Nrm2, len, vStride == 1).nrm2(v, vOff, len, vStride)
    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        vectors(EngineImplementation.Asum, len, vStride == 1).asum(v, vOff, len, vStride)
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int =
        vectors(EngineImplementation.Iamax, len, vStride == 1).iamax(v, vOff, len, vStride)
    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) = vectors(EngineImplementation.Axpy, len, yStride == 1 && xStride == 1)
        .axpy(y, yOff, alpha, x, xOff, len, yStride, xStride)
    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) =
        vectors(EngineImplementation.Scale, len, vStride == 1).scale(v, vOff, alpha, len, vStride)
    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        vectors(EngineImplementation.Swap, len, aStride == 1 && bStride == 1)
            .swap(a, aOff, b, bOff, len, aStride, bStride)
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
    ) {
        val kernels = vectors(EngineImplementation.Rot, len, xStride == 1 && yStride == 1)
        kernels.rot(x, xOff, y, yOff, len, c, s, xStride, yStride)
    }
    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = (routing.select(EngineImplementation.IndexedDot, count))
        .dotDense(indices, indexOffset, values, valueOffset, count, dense)

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
    ): Double = routing.select(
        EngineImplementation.IndexedSparseDot,
        maxOf(xCount, yCount),
    ).dotSparse(
        xIndices,
        xIndexOffset,
        xValues,
        xValueOffset,
        xCount,
        yIndices,
        yIndexOffset,
        yValues,
        yValueOffset,
        yCount,
    )

    override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ): Unit = routing.select(
        EngineImplementation.IndexedAxpy,
        count,
    ).axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)

    override fun computeIndexedScatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = routing.select(
        EngineImplementation.IndexedScatter,
        count,
    ).computeIndexedScatter(indices, indexOffset, values, valueOffset, count, destination)

    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = routing.select(
        EngineImplementation.IndexedGather,
        count,
    ).gather(indices, indexOffset, values, valueOffset, count, source)

    override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = routing.select(
        EngineImplementation.IndexedGatherZero,
        count,
    ).gatherZero(indices, indexOffset, values, valueOffset, count, source)

    override fun computeIndexedNrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        routing.select(EngineImplementation.IndexedNrm2, count).computeIndexedNrm2(indices, indexOffset, count, values)

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        routing.select(EngineImplementation.Copy, len, contiguous = aStride == 1 && bStride == 1)
            .copy(a, aOff, b, bOff, len, aStride, bStride)

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int =
        panelImplementation.executionGroup(work, rows, columns, contiguous)

    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean =
        panelImplementation.prefersContiguous(work, rows, columns)

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
    ): Unit = panelImplementation.multiDot(
        alpha,
        a,
        aOffset,
        lda,
        x,
        xOffset,
        xStride,
        rows,
        columns,
        beta,
        y,
        yOffset,
        yStride,
    )

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
    ): Unit = panelImplementation.columnUpdate(
        alpha,
        a,
        aOffset,
        lda,
        x,
        xOffset,
        xStride,
        rows,
        columns,
        y,
        yOffset,
        yStride,
    )

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
    ): Unit = panelImplementation.coupledUpdateDot(
        alpha,
        a,
        aOffset,
        lda,
        x,
        xOffset,
        rows,
        columns,
        y,
        yOffset,
        coefficients,
        coefficientOffset,
        sums,
        sumOffset,
        xStride,
        yStride,
        coefficientStride,
        sumStride,
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
    ): Unit = panelImplementation.rankUpdate(
        alpha,
        a,
        aOffset,
        lda,
        x,
        xOffset,
        xStride,
        rows,
        columns,
        coefficients,
        coefficientOffset,
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
    ): Unit = panelImplementation.rankTwoUpdate(
        alpha,
        a,
        aOffset,
        lda,
        x,
        xOffset,
        xStride,
        y,
        yOffset,
        yStride,
        rows,
        columns,
        u,
        uOffset,
        uStride,
        v,
        vOffset,
        vStride,
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
    ): Unit = panelImplementation.indexedColumnUpdate(
        alpha,
        a,
        aOffset,
        rowStride,
        indexStride,
        indices,
        values,
        fromIndex,
        columns,
        rows,
        y,
        yOffset,
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
    ): Unit = panelImplementation.indexedRankUpdate(
        alpha,
        a,
        aOffset,
        rowStride,
        indexStride,
        indices,
        values,
        fromIndex,
        columns,
        rows,
        x,
        xOffset,
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
    ): Unit = panelImplementation.indexedCoupledUpdate(
        alpha,
        a,
        b,
        offset,
        rowStride,
        indexStride,
        indices,
        values,
        fromIndex,
        columns,
        rows,
        pivot,
        sums,
        sumOffset,
        excluded,
        destinationOffset,
        destinationRowStride,
        destinationIndexStride,
    )

    override val tileRows: Int get() = productImplementation.tileRows

    override val tileColumns: Int get() = productImplementation.tileColumns

    override fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean =
        productImplementation.packsProduct(rows, columns, depth)

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
    ): Unit = productImplementation.productBlock(
        alpha,
        packedA,
        aOffset,
        aGroupStride,
        packedB,
        bOffset,
        bGroupStride,
        rows,
        columns,
        depth,
        beta,
        c,
        cOffset,
        ldc,
    )

    override fun rightHandSideGroup(order: Int, sides: Int): Int =
        triangularImplementation.rightHandSideGroup(order, sides)

    override fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean =
        triangularImplementation.gathersRightHandSides(rhsStride, order, sides)

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
    ): Unit = triangularImplementation.diagonalSolve(
        t,
        n,
        start,
        size,
        transposed,
        lower,
        unitDiag,
        x,
        xOffset,
        orderStride,
        rhsStride,
        sides,
        tOffset,
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
    ): Unit = triangularImplementation.diagonalMultiply(
        t,
        n,
        start,
        size,
        transposed,
        lower,
        unitDiag,
        x,
        xOffset,
        orderStride,
        rhsStride,
        sides,
        tOffset,
    )
}
