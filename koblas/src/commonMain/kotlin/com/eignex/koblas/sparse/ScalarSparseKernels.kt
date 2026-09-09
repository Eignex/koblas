package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.internal.configuration.ImplementationNames
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.requireShape
import kotlin.math.abs

/** Scalar sparse-vector kernels used as the semantic oracle and universal fallback. */
internal object ScalarSparseKernels : SparseKernels {
    override val name: String get() = ImplementationNames.SCALAR

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var sum = 0.0
        for (k in x.indices.indices) sum += x.values[k] * y[x.indices[k]]
        return sum
    }

    override fun dot(x: SparseVector, y: SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var sum = 0.0
        var a = 0
        var b = 0
        while (a < x.indices.size && b < y.indices.size) {
            val ia = x.indices[a]
            val ib = y.indices[b]
            when {
                ia < ib -> a++

                ia > ib -> b++

                else -> {
                    sum += x.values[a] * y.values[b]
                    a++
                    b++
                }
            }
        }
        return sum
    }

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0) return
        for (k in x.indices.indices) y[x.indices[k]] += alpha * x.values[k]
    }

    override fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        for (k in x.indices.indices) out[x.indices[k]] = x.values[k]
    }

    override fun gather(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
        for (k in x.indices.indices) x.values[k] = from[x.indices[k]]
    }

    override fun gatherZero(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
        for (k in x.indices.indices) {
            val i = x.indices[k]
            x.values[k] = from[i]
            from[i] = 0.0
        }
    }

    override fun nrm2(x: SparseVector): Double = euclideanNorm(x.values, 0, x.values.size)

    override fun asum(x: SparseVector): Double {
        var sum = 0.0
        for (value in x.values) sum += abs(value)
        return sum
    }
}
