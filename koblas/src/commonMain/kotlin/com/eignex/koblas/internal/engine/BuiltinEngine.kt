@file:Suppress("LongParameterList") // Primitive BLAS windows are forwarded without allocating call descriptions.

package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/** Fixed arithmetic roles, with scalar coverage for every omitted role. */
internal class KernelSet(
    val vectors: EngineKernels = ScalarEngine,
    val indexed: EngineKernels = ScalarEngine,
    val panels: EngineKernels = ScalarEngine,
    val products: EngineKernels = ScalarEngine,
    val triangular: EngineKernels = ScalarEngine,
)

/** Owned arithmetic roles and one optional host library with measured crossovers. */
internal class BuiltinEngine(
    override val name: String,
    val roles: KernelSet = KernelSet(),
    val host: EngineKernels? = null,
    private val tuning: EngineTuning? = null,
) : EngineKernels {
    override val implementations = ScalarEngine.implementations.filterTo(mutableSetOf()) { it.vector }
    override val providers: List<KoblasEngine> = listOfNotNull(
        host,
        roles.vectors,
        roles.indexed,
        roles.panels,
        roles.products,
        roles.triangular,
        ScalarEngine,
    ).distinct()

    private val owned = EngineImplementation.entries.map { operation ->
        val leaf = if (operation.name.startsWith("Indexed")) roles.indexed else roles.vectors
        if (operation in leaf.implementations) leaf else ScalarEngine
    }
    private val minimums = IntArray(owned.size) { i ->
        maxOf(
            owned[i].minimumLength(EngineImplementation.entries[i]),
            tuning?.simdMinimum(EngineImplementation.entries[i]) ?: 0,
        )
    }
    private val hostAdjacent = IntArray(owned.size) { hostMinimum(EngineImplementation.entries[it], true) }
    private val hostStrided = IntArray(owned.size) { hostMinimum(EngineImplementation.entries[it], false) }

    private fun hostMinimum(operation: EngineImplementation, contiguous: Boolean): Int =
        if (host == null || operation !in host.implementations) {
            -1
        } else {
            tuning?.hostMinimum(operation, contiguous) ?: 0
        }

    internal fun vectorProvider(operation: EngineImplementation, length: Int, contiguous: Boolean): EngineKernels {
        val i = operation.ordinal
        val limit = (if (contiguous) hostAdjacent else hostStrided)[i]
        if (limit >= 0 && length >= limit) return requireNotNull(host)
        val leaf = owned[i]
        return if (length >= minimums[i] && (contiguous || !leaf.requiresUnitStride(operation))) leaf else ScalarEngine
    }

    internal fun hostFor(
        operation: EngineImplementation,
        rows: Int,
        columns: Int,
        depth: Int,
        borrowed: Boolean,
    ): EngineKernels? = host?.takeIf {
        operation in it.implementations &&
            (tuning?.hostEligible(operation, rows, columns, depth, borrowed) != false)
    }

    /** A constant scalar receiver keeps short portable calls inlineable. */
    private inline fun <R> withVector(
        operation: EngineImplementation,
        length: Int,
        contiguous: Boolean = true,
        action: (EngineKernels) -> R,
    ): R {
        val provider = vectorProvider(operation, length, contiguous)
        return if (provider === ScalarEngine) action(ScalarEngine) else action(provider)
    }

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String =
        roles.panels.implementationFor(work, rows, columns, contiguous)
    override fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> =
        roles.products.implementationsFor(rows, columns, depth)
    override fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> =
        roles.triangular.implementationsFor(order, sides, contiguous)

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = withVector(Dot, len, aStride == 1 && bStride == 1) {
        it.dot(a, aOff, b, bOff, len, aStride, bStride)
    }
    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        withVector(Sum, len, vStride == 1) { it.sum(v, vOff, len, vStride) }
    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        withVector(Nrm2, len, vStride == 1) { it.nrm2(v, vOff, len, vStride) }
    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        withVector(Asum, len, vStride == 1) { it.asum(v, vOff, len, vStride) }
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int =
        withVector(Iamax, len, vStride == 1) { it.iamax(v, vOff, len, vStride) }
    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) = withVector(Axpy, len, yStride == 1 && xStride == 1) {
        it.axpy(y, yOff, alpha, x, xOff, len, yStride, xStride)
    }
    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) =
        withVector(Scale, len, vStride == 1) { it.scale(v, vOff, alpha, len, vStride) }
    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        withVector(Swap, len, aStride == 1 && bStride == 1) {
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
    ) = withVector(Rot, len, xStride == 1 && yStride == 1) {
        it.rot(x, xOff, y, yOff, len, c, s, xStride, yStride)
    }
    override fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = withVector(IndexedDot, count) {
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
    ): Double = withVector(IndexedSparseDot, maxOf(xCount, yCount)) {
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
    ): Unit = withVector(IndexedAxpy, count) {
        it.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
    }

    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = withVector(IndexedScatter, count) {
        it.scatter(indices, indexOffset, values, valueOffset, count, destination)
    }

    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = withVector(IndexedGather, count) {
        it.gather(indices, indexOffset, values, valueOffset, count, source)
    }

    override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = withVector(IndexedGatherZero, count) {
        it.gatherZero(indices, indexOffset, values, valueOffset, count, source)
    }

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        val contiguous = aStride == 1 && bStride == 1
        val provider = vectorProvider(Copy, len, contiguous)
        if (provider === ScalarEngine && contiguous) {
            a.copyInto(b, bOff, aOff, aOff + len)
        } else {
            provider.copy(a, aOff, b, bOff, len, aStride, bStride)
        }
    }
}
