@file:Suppress("LongParameterList") // Forward raw primitive windows without allocating call descriptions.

package com.eignex.koblas.internal.configuration

import com.eignex.koblas.*
import com.eignex.koblas.internal.dense.*

/** Backends are bound once; arithmetic never resolves identifiers or capabilities. */
internal class CompiledRouting(profile: EngineProfile, engines: List<KoblasEngine>) {
    val backends = engines.filterNot { it === PortableEngine } + PortableEngine
    private val routes = Array(EngineImplementation.entries.size) { index ->
        val operation = EngineImplementation.entries[index]
        profile.rules.mapNotNull { rule ->
            backends.firstOrNull { it.id == rule.source && operation in it.implementations }
                ?.takeIf { operation in rule.implementations }
                ?.let { rule to it }
        }.toTypedArray()
    }

    private val lengths = Array(EngineImplementation.Panels.ordinal) {
        compileLength(EngineImplementation.entries[it])
    }

    /** Fold all vector-invariant shape/work constraints before arithmetic reaches a length check. */
    private fun compileLength(operation: EngineImplementation): LengthDispatch {
        var result = LengthDispatch(PortableEngine, 0, Int.MAX_VALUE, 3, null)
        for ((rule, backend) in routes[operation.ordinal].reversed()) {
            val minimum = if (operation.ordinal >= EngineImplementation.IndexedDot.ordinal) {
                backend.minimumIndexedLength
            } else {
                backend.minimumVectorLength
            }
            val lower = maxOf(rule.rows.first, minimum, rule.work.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
            val strided = (operation.ordinal >= EngineImplementation.IndexedDot.ordinal || !backend.contiguousVectors)
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
        if (operation.ordinal < EngineImplementation.Panels.ordinal) {
            return lengths[operation.ordinal].select(rows, contiguous)
        }
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
internal class CombinedEngine(private val routing: CompiledRouting, override val name: String) : KoblasEngine() {
    override val components get() = routing.backends
    private val host = routing.backends.firstOrNull { it.vendor != null }
    override val vendor get() = host?.vendor
    override val libraryPath get() = host?.libraryPath.orEmpty()
    override val version get() = host?.version.orEmpty()
    override val threadEvidence get() = host?.threadEvidence ?: super.threadEvidence
    override val implementations = routing.backends.flatMapTo(mutableSetOf()) { it.implementations }
    override fun wholeCallFor(
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
    ).takeUnless { it === PortableEngine }
    override val panelImplementation = family(EngineImplementation.Panels, PortablePanelKernels)
    override val productImplementation = family(EngineImplementation.Products, PortableProductKernels)
    override val triangularImplementation = family(EngineImplementation.Triangular, PortableTriangularKernels)
    private fun family(operation: EngineImplementation, portable: KoblasEngine): KoblasEngine =
        routing.select(operation).takeUnless { it === PortableEngine } ?: portable
    private fun vectors(operation: EngineImplementation, length: Int, contiguous: Boolean = true): KoblasEngine =
        routing.select(operation, length, contiguous = contiguous)

    override fun implementationFor(operation: EngineImplementation, length: Int, contiguous: Boolean): String? {
        val backend = routing.select(operation, length, contiguous = contiguous)
        return if ((operation == EngineImplementation.Nrm2 && backend.dynamicNorm) ||
            (operation == EngineImplementation.Rot && backend.vendor != null)
        ) {
            null
        } else {
            backend.name
        }
    }

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
        // CBLAS may store x before loading y; the owned contract loads both before either write.
        val kernels = if (x === y && xOff == yOff && xStride == yStride) {
            PortableEngine
        } else {
            vectors(
                EngineImplementation.Rot,
                len,
                xStride == 1 && yStride == 1,
            )
        }
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
}
