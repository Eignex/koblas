@file:Suppress("LongParameterList") // Forward raw primitive windows without allocating call descriptions.

package com.eignex.koblas.internal.configuration

import com.eignex.koblas.*
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.dense.ScalarVectorKernels
import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseOperation

/** Backends are bound once; arithmetic never resolves identifiers or capabilities. */
internal class CompiledRouting(profile: EngineProfile, backends: List<EngineBackend>) {
    private val routes = Array(EngineImplementation.entries.size) { index ->
        val operation = EngineImplementation.entries[index]
        profile.rules.mapNotNull { rule ->
            backends.firstOrNull { it.name == rule.backend && operation in it.implementations }
                ?.takeIf { operation in rule.implementations }
                ?.let { rule to it }
        }.toTypedArray()
    }

    fun hasPreference(operation: EngineImplementation): Boolean =
        routes[operation.ordinal].any { it.second !== portableBackend }

    private val lengths = Array(EngineImplementation.IndexedDot.ordinal + 1) {
        compileLength(EngineImplementation.entries[it])
    }

    /** Fold all vector-invariant shape/work constraints before arithmetic reaches a length check. */
    private fun compileLength(operation: EngineImplementation): LengthDispatch {
        var result = LengthDispatch(portableBackend, 0, Int.MAX_VALUE, 3, null)
        for ((rule, backend) in routes[operation.ordinal].reversed()) {
            val minimum = if (operation == EngineImplementation.IndexedDot) {
                backend.minimumIndexedLength
            } else {
                backend.minimumVectorLength
            }
            val lower = maxOf(rule.rows.first, minimum, rule.work.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
            val strided = operation != EngineImplementation.Rot &&
                (operation == EngineImplementation.IndexedDot || !backend.contiguousVectors)
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
    ): EngineBackend {
        if (operation.ordinal <= EngineImplementation.IndexedDot.ordinal) {
            return lengths[operation.ordinal].select(rows, contiguous)
        }
        for ((rule, backend) in routes[operation.ordinal]) {
            if (rule.matches(rows, columns, depth, contiguous, borrowed)) return backend
        }
        return portableBackend
    }
}

/** Primitive length and layout checks over the backends bound to one operation. */
internal class LengthDispatch(
    private val backend: EngineBackend,
    private val minimum: Int,
    private val maximum: Int,
    private val layouts: Int,
    private val fallback: LengthDispatch?,
) {
    fun select(length: Int, contiguous: Boolean = true): EngineBackend {
        if (fallback == null) return backend
        return if (length >= minimum && length <= maximum && layouts and (if (contiguous) 1 else 2) != 0) {
            backend
        } else {
            fallback.select(length, contiguous)
        }
    }
}

/** One primitive-window interface serves leaves and the immutable routed composition. */
internal class CombinedKernels(private val routing: CompiledRouting, override val name: String) :
    DenseVectorKernels,
    IndexedSparseKernels {
    private fun vectors(operation: EngineImplementation, length: Int, contiguous: Boolean = true): DenseVectorKernels =
        routing.select(operation, length, contiguous = contiguous).vectors

    fun implementationFor(operation: EngineImplementation, length: Int, contiguous: Boolean): String? {
        val backend = routing.select(operation, length, contiguous = contiguous)
        return if ((operation == EngineImplementation.Nrm2 && backend.dynamicNorm) ||
            (operation == EngineImplementation.Rot && backend.binding != null)
        ) {
            null
        } else {
            backend.vectorName
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
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        // CBLAS may store x before loading y; the owned contract loads both before either write.
        val kernels = if (x === y && xOff == yOff) ScalarVectorKernels else vectors(EngineImplementation.Rot, len)
        kernels.rot(x, xOff, y, yOff, len, c, s)
    }
    override fun implementationFor(operation: SparseOperation, count: Int): String? =
        if (operation == SparseOperation.DotDense) {
            routing.select(EngineImplementation.IndexedDot, count).vectorName
        } else {
            ScalarIndexedSparseKernels.name
        }

    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = (routing.select(EngineImplementation.IndexedDot, count).indexedDot ?: ScalarIndexedSparseKernels)
        .dotDense(indices, indexOffset, values, valueOffset, count, dense)
}
