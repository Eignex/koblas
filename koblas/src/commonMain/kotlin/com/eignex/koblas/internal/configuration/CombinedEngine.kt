@file:Suppress("LongParameterList") // Forward raw primitive windows without allocating call descriptions.

package com.eignex.koblas.internal.configuration

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.Vendor
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.sparse.*

/** Backends are bound once; arithmetic never resolves identifiers or capabilities. */
internal class CompiledRouting(profile: EngineProfile, engines: List<KoblasEngine>) {
    private val portable = PortableEngine
    val backends = engines.flatMap { it.providers }.filterNot { it === portable }.distinct() + portable
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
    ): KoblasEngine {
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
internal class LengthDispatch(val backends: Array<KoblasEngine>, val windows: IntArray) {
    // A final receiver lets Native inline the fallback without a per-call singleton lookup.
    val portable = PortableEngine
    val minimum = windows.indices.step(3).minOfOrNull { windows[it] } ?: Int.MAX_VALUE
}

/** Inline both calls so the portable fallback has a statically known receiver. */
private inline fun <R> LengthDispatch.withBackend(
    length: Int,
    contiguous: Boolean = true,
    action: (KoblasEngine) -> R,
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
internal class CombinedEngine(val routing: CompiledRouting, override val name: String) : KoblasEngine {
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
    internal fun copyProvider(length: Int, contiguous: Boolean): KoblasEngine =
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
}
