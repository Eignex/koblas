@file:Suppress("LongParameterList") // Forward raw primitive windows without allocating call descriptions.

package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.kernels.PanelWork
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/** Backends are bound once; arithmetic never resolves identifiers or capabilities. */
internal class CompiledRouting(profile: EngineProfile, engines: List<EngineKernels>) {
    private val portable = ScalarEngine
    val backends: List<EngineKernels> = engines.flatMap { (it as? CombinedEngine)?.routing?.backends ?: listOf(it) }
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

    /** The leaf a whole matrix call or a scheduled family binds; a vector run dispatches on [compileLength]. */
    fun select(
        operation: EngineImplementation,
        rows: Int = 0,
        columns: Int = 1,
        depth: Int = 1,
        borrowed: Boolean = false,
    ): EngineKernels {
        for ((rule, backend) in routes[operation.ordinal]) {
            if (rule.matches(rows, columns, depth, true, borrowed)) return backend
        }
        return portable
    }
}

/** Primitive length and layout bounds, in profile order, beside their bound providers. */
internal class LengthDispatch(val backends: Array<EngineKernels>, val windows: IntArray) {
    // A final receiver lets Native inline the fallback without a per-call singleton lookup.
    val portable = ScalarEngine
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
    override val implementations = ScalarEngine.implementations.filterTo(mutableSetOf()) { it.vector }

    override val providers: List<KoblasEngine> get() = routing.backends.toList()

    // Indexed by ordinal; only vector entries are read. Each kernel below keeps its own final field.
    private val lengthRoutes = EngineImplementation.entries.map { routing.compileLength(it) }
    private val dotRoute = lengthRoutes[EngineImplementation.Dot.ordinal]
    private val sumRoute = lengthRoutes[EngineImplementation.Sum.ordinal]
    private val nrm2Route = lengthRoutes[EngineImplementation.Nrm2.ordinal]
    private val asumRoute = lengthRoutes[EngineImplementation.Asum.ordinal]
    private val iamaxRoute = lengthRoutes[EngineImplementation.Iamax.ordinal]
    private val axpyRoute = lengthRoutes[EngineImplementation.Axpy.ordinal]
    private val scaleRoute = lengthRoutes[EngineImplementation.Scale.ordinal]
    private val swapRoute = lengthRoutes[EngineImplementation.Swap.ordinal]
    private val rotRoute = lengthRoutes[EngineImplementation.Rot.ordinal]
    private val copyRoute = lengthRoutes[EngineImplementation.Copy.ordinal]
    private val indexedDotRoute = lengthRoutes[EngineImplementation.IndexedDot.ordinal]
    private val indexedSparseDotRoute = lengthRoutes[EngineImplementation.IndexedSparseDot.ordinal]
    private val indexedAxpyRoute = lengthRoutes[EngineImplementation.IndexedAxpy.ordinal]
    private val indexedScatterRoute = lengthRoutes[EngineImplementation.IndexedScatter.ordinal]
    private val indexedGatherRoute = lengthRoutes[EngineImplementation.IndexedGather.ordinal]
    private val indexedGatherZeroRoute = lengthRoutes[EngineImplementation.IndexedGatherZero.ordinal]
    internal val panelImplementation = routing.select(EngineImplementation.Panels)
    internal val productImplementation = routing.select(EngineImplementation.Products)
    internal val triangularImplementation = routing.select(EngineImplementation.Triangular)

    /** The leaf a vector [operation] over [length] entries dispatches to, read from the same windows it runs. */
    internal fun vectorProvider(operation: EngineImplementation, length: Int, contiguous: Boolean): EngineKernels =
        lengthRoutes[operation.ordinal].withBackend(length, contiguous) { it }

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String =
        panelImplementation.implementationFor(work, rows, columns, contiguous)

    override fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> =
        productImplementation.implementationsFor(rows, columns, depth)

    override fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> =
        triangularImplementation.implementationsFor(order, sides, contiguous)

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
    override fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = indexedDotRoute.withBackend(count) {
        it.dot(indices, indexOffset, values, valueOffset, count, dense)
    }

    override fun dot(
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
        it.dot(
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

    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = indexedScatterRoute.withBackend(count) {
        it.scatter(indices, indexOffset, values, valueOffset, count, destination)
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

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        copyRoute.withBackend(len, aStride == 1 && bStride == 1) {
            it.copy(a, aOff, b, bOff, len, aStride, bStride)
        }
}
