package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import kotlinx.benchmark.*

/** Tall, wide, and remainder-heavy coverage for both orientations of dense GEMV. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class GemvShapeBenchmark {
    @Param("64x2048", "2048x64", "2047x257", "257x2047", "8192x64")
    lateinit var shape: String

    @Param(BUILTIN_BACKEND, OPENBLAS_BACKEND, ONEMKL_BACKEND)
    lateinit var denseArm: String

    private lateinit var arm: DenseBenchmarkArm
    private lateinit var a: DenseMatrix
    private lateinit var x: DoubleArray
    private lateinit var transposedX: DoubleArray
    private lateinit var y: DoubleArray
    private lateinit var transposedY: DoubleArray

    @Setup
    fun setup() {
        arm = DenseBenchmarkArm.resolve(denseArm)
        val parts = shape.split('x')
        val rows = parts[0].toInt()
        val cols = parts[1].toInt()
        val rng = benchRng()
        a = randomMatrix(rows, cols, rng)
        x = randomVector(cols, rng)
        transposedX = randomVector(rows, rng)
        y = DoubleArray(rows)
        transposedY = DoubleArray(cols)
    }

    @Benchmark
    fun gemv() {
        arm.external?.gemv(1.0, a, x, 0.0, y, false) ?: arm.context!!.gemv(1.0, a, x, 0.0, y)
    }

    @Benchmark
    fun gemvTransposed() {
        arm.external?.gemv(1.0, a, transposedX, 0.0, transposedY, true)
            ?: arm.context!!.gemv(1.0, a, transposedX, 0.0, transposedY, transpose = true)
    }
}
