@file:Suppress("LongParameterList") // Forward raw primitive windows without allocating call descriptions.

package com.eignex.koblas.engine

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.sparse.*

/** Backends are bound once; arithmetic never resolves identifiers or capabilities. */
internal class CompiledRouting(profile: EngineProfile, engines: List<EngineKernels>) {
    private val portable = PortableEngine
    val backends: List<EngineKernels> = engines.flatMap {
        if (it is CombinedEngine) {
            it.routing.backends
        } else {
            listOf(
                it,
            )
        }
    }
        .filterNot { it === portable }.distinct() + portable
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

    /** Fold all vector-invariant shape/work constraints before arithmetic reaches a length check. */
    fun compileLength(operation: EngineImplementation): LengthDispatch {
        val windows = routes[operation.ordinal].mapNotNull { (rule, backend) ->
            val lower = maxOf(
                rule.rows.first,
                backend.minimumLength(operation),
                rule.work.coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
            )
            val strided = !backend.requiresUnitStride(operation)
            val mask = (if (rule.matches(lower, 1, 1, true, false)) 1 else 0) or
                (if (strided && rule.matches(lower, 1, 1, false, false)) 2 else 0)
            if (mask == 0) null else backend to intArrayOf(lower, rule.rows.last, mask)
        }
        return LengthDispatch(
            windows.map { it.first }.toTypedArray(),
            windows.flatMap { it.second.asList() }.toIntArray(),
        )
    }

    fun select(
        operation: EngineImplementation,
        rows: Int = 0,
        columns: Int = 1,
        depth: Int = 1,
        contiguous: Boolean = true,
        borrowed: Boolean = false,
    ): EngineKernels {
        val vector = operation.vector
        val width = if (vector) 1 else columns
        val shared = if (vector) 1 else depth
        for ((rule, backend) in routes[operation.ordinal]) {
            if (vector && rows < backend.minimumLength(operation)) continue
            if (vector && !contiguous && backend.requiresUnitStride(operation)) continue
            if (rule.matches(rows, width, shared, contiguous, borrowed && !vector)) return backend
        }
        return portable
    }
}

/** Primitive length and layout bounds, in profile order, beside their bound providers. */
internal class LengthDispatch(val backends: Array<EngineKernels>, val windows: IntArray) {
    // A final receiver lets Native inline the fallback without a per-call singleton lookup.
    val portable = PortableEngine
    val minimum = windows.indices.step(3).minOfOrNull { windows[it] } ?: Int.MAX_VALUE
}

/** Inline both calls so the portable fallback has a statically known receiver. */
private inline fun <R> LengthDispatch.withBackend(
    length: Int,
    contiguous: Boolean = true,
    action: (EngineKernels) -> R,
): R {
    if (length < minimum) return action(portable)
    val layout = if (contiguous) 1 else 2
    var index = 0
    while (index < windows.size) {
        if (length >= windows[index] && length <= windows[index + 1] && windows[index + 2] and layout != 0) {
            return action(backends[index / 3])
        }
        index += 3
    }
    return action(portable)
}

/** One primitive-window interface serves leaves and the immutable routed composition. */
internal class CombinedEngine(val routing: CompiledRouting, override val name: String) : EngineKernels {
    override val implementations = PortableEngine.implementations.filterTo(mutableSetOf()) { it.vector }
    private val dotRoute = routing.compileLength(EngineImplementation.Dot)
    private val sumRoute = routing.compileLength(EngineImplementation.Sum)
    private val nrm2Route = routing.compileLength(EngineImplementation.Nrm2)
    private val asumRoute = routing.compileLength(EngineImplementation.Asum)
    private val iamaxRoute = routing.compileLength(EngineImplementation.Iamax)
    private val axpyRoute = routing.compileLength(EngineImplementation.Axpy)
    private val scaleRoute = routing.compileLength(EngineImplementation.Scale)
    private val swapRoute = routing.compileLength(EngineImplementation.Swap)
    private val rotRoute = routing.compileLength(EngineImplementation.Rot)
    private val copyRoute = routing.compileLength(EngineImplementation.Copy)
    private val indexedDotRoute = routing.compileLength(EngineImplementation.IndexedDot)
    private val indexedSparseDotRoute = routing.compileLength(EngineImplementation.IndexedSparseDot)
    private val indexedAxpyRoute = routing.compileLength(EngineImplementation.IndexedAxpy)
    private val indexedScatterRoute = routing.compileLength(EngineImplementation.IndexedScatter)
    private val indexedGatherRoute = routing.compileLength(EngineImplementation.IndexedGather)
    private val indexedGatherZeroRoute = routing.compileLength(EngineImplementation.IndexedGatherZero)
    private val indexedNrm2Route = routing.compileLength(EngineImplementation.IndexedNrm2)
    internal val panelImplementation = routing.select(EngineImplementation.Panels)
    internal val productImplementation = routing.select(EngineImplementation.Products)
    internal val triangularImplementation = routing.select(EngineImplementation.Triangular)
    internal fun copyProvider(length: Int, contiguous: Boolean): EngineKernels =
        copyRoute.withBackend(length, contiguous) { it }
    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = dotRoute.withBackend(len, aStride == 1 && bStride == 1) {
        it.dot(a, aOff, b, bOff, len, aStride, bStride)
    }
    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        sumRoute.withBackend(len, vStride == 1) { it.sum(v, vOff, len, vStride) }
    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        nrm2Route.withBackend(len, vStride == 1) { it.nrm2(v, vOff, len, vStride) }
    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        asumRoute.withBackend(len, vStride == 1) { it.asum(v, vOff, len, vStride) }
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int =
        iamaxRoute.withBackend(len, vStride == 1) { it.iamax(v, vOff, len, vStride) }
    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) = axpyRoute.withBackend(len, yStride == 1 && xStride == 1) {
        it.axpy(y, yOff, alpha, x, xOff, len, yStride, xStride)
    }
    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) =
        scaleRoute.withBackend(len, vStride == 1) { it.scale(v, vOff, alpha, len, vStride) }
    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        swapRoute.withBackend(len, aStride == 1 && bStride == 1) {
            it.swap(a, aOff, b, bOff, len, aStride, bStride)
        }
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
    ) = rotRoute.withBackend(len, xStride == 1 && yStride == 1) {
        it.rot(x, xOff, y, yOff, len, c, s, xStride, yStride)
    }
    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = indexedDotRoute.withBackend(count) {
        it.dotDense(indices, indexOffset, values, valueOffset, count, dense)
    }

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
    ): Double = indexedSparseDotRoute.withBackend(maxOf(xCount, yCount)) {
        it.dotSparse(
            xIndices, xIndexOffset, xValues, xValueOffset, xCount,
            yIndices, yIndexOffset, yValues, yValueOffset, yCount,
        )
    }

    override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ): Unit = indexedAxpyRoute.withBackend(count) {
        it.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
    }

    override fun computeIndexedScatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = indexedScatterRoute.withBackend(count) {
        it.computeIndexedScatter(indices, indexOffset, values, valueOffset, count, destination)
    }

    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = indexedGatherRoute.withBackend(count) {
        it.gather(indices, indexOffset, values, valueOffset, count, source)
    }

    override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = indexedGatherZeroRoute.withBackend(count) {
        it.gatherZero(indices, indexOffset, values, valueOffset, count, source)
    }

    override fun computeIndexedNrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        indexedNrm2Route.withBackend(count) { it.computeIndexedNrm2(indices, indexOffset, count, values) }

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        copyRoute.withBackend(len, aStride == 1 && bStride == 1) {
            it.copy(a, aOff, b, bOff, len, aStride, bStride)
        }

    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        transpose: Boolean,
        scratch: Scratch?,
    ) {
        val operation = if (transpose) EngineImplementation.GemvTransposed else EngineImplementation.Gemv
        val provider = wholeCallFor(operation, a.rows, a.cols, 1, a !is DenseMatrix)
        if (provider != null) {
            provider.gemv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch)
        } else {
            super.gemv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch)
        }
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        val provider = wholeCallFor(EngineImplementation.Symv, a.rows, a.cols, 1, a !is DenseMatrix)
        if (provider != null) {
            provider.symv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch)
        } else {
            super.symv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch)
        }
    }

    override fun ger(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        scratch: Scratch?,
    ) {
        val provider = wholeCallFor(EngineImplementation.Ger, a.rows, a.cols, 1, a !is DenseMatrix)
        if (provider != null) {
            provider.ger(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch)
        } else {
            super.ger(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch)
        }
    }

    override fun syr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        val provider = wholeCallFor(EngineImplementation.Syr, a.rows, a.rows, 1, a !is DenseMatrix)
        if (provider != null) {
            provider.syr(alpha, x, xOffset, xStride, a, lower, scratch)
        } else {
            super.syr(alpha, x, xOffset, xStride, a, lower, scratch)
        }
    }

    override fun syr2(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        val provider = wholeCallFor(EngineImplementation.Syr2, a.rows, a.rows, 1, a !is DenseMatrix)
        if (provider != null) {
            provider.syr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, lower, scratch)
        } else {
            super.syr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, lower, scratch)
        }
    }

    override fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) {
        val operation = if (transpose) EngineImplementation.TrsvTransposed else EngineImplementation.Trsv
        val provider = wholeCallFor(operation, a.rows, a.rows, 1, a !is DenseMatrix)
        if (provider != null) {
            provider.trsv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
        } else {
            super.trsv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
        }
    }

    override fun trmv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) {
        val operation = if (transpose) EngineImplementation.TrmvTransposed else EngineImplementation.Trmv
        val provider = wholeCallFor(operation, a.rows, a.rows, 1, a !is DenseMatrix)
        if (provider != null) {
            provider.trmv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
        } else {
            super.trmv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
        }
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        scratch: Scratch?,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val provider = wholeCallFor(EngineImplementation.Gemm, c.rows, c.cols, depth, borrowed)
        if (provider != null) {
            provider.gemm(alpha, a, transposeA, b, transposeB, beta, c, scratch)
        } else {
            super.gemm(alpha, a, transposeA, b, transposeB, beta, c, scratch)
        }
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val provider = wholeCallFor(EngineImplementation.Gemmt, c.rows, c.cols, depth, borrowed)
        if (provider != null) {
            provider.gemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, scratch)
        } else {
            super.gemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, scratch)
        }
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        right: Boolean,
        scratch: Scratch?,
    ) {
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val provider = wholeCallFor(EngineImplementation.Symm, c.rows, c.cols, a.rows, borrowed)
        if (provider != null) {
            provider.symm(alpha, a, b, beta, c, lower, right, scratch)
        } else {
            super.symm(alpha, a, b, beta, c, lower, right, scratch)
        }
    }

    override fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        val depth = if (transpose) a.rows else a.cols
        val provider = wholeCallFor(
            EngineImplementation.Syrk,
            c.rows,
            c.cols,
            depth,
            a !is DenseMatrix || c !is DenseMatrix,
        )
        if (provider != null) {
            provider.syrk(alpha, a, transpose, beta, c, lower, scratch)
        } else {
            super.syrk(alpha, a, transpose, beta, c, lower, scratch)
        }
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        val depth = if (transpose) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val provider = wholeCallFor(EngineImplementation.Syr2k, c.rows, c.cols, depth, borrowed)
        if (provider != null) {
            provider.syr2k(alpha, a, b, transpose, beta, c, lower, scratch)
        } else {
            super.syr2k(alpha, a, b, transpose, beta, c, lower, scratch)
        }
    }

    override fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        scratch: Scratch?,
    ) {
        val provider = wholeCallFor(
            EngineImplementation.Trsm,
            b.rows,
            b.cols,
            a.rows,
            a !is DenseMatrix || b !is DenseMatrix,
        )
        if (provider != null) {
            provider.trsm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
        } else {
            super.trsm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
        }
    }

    override fun trmm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        scratch: Scratch?,
    ) {
        val provider = wholeCallFor(
            EngineImplementation.Trmm,
            b.rows,
            b.cols,
            a.rows,
            a !is DenseMatrix || b !is DenseMatrix,
        )
        if (provider != null) {
            provider.trmm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
        } else {
            super.trmm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
        }
    }
}
