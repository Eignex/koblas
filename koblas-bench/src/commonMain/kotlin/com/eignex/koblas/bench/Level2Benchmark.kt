package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class Level2Benchmark {
    @Param("16", "64", "256", "1024", "2048")
    var n: Int = 0

    @Param(BUILTIN_BACKEND, OPENBLAS_BACKEND, ONEMKL_BACKEND)
    var denseArm: String = BUILTIN_BACKEND

    private lateinit var arm: DenseBenchmarkArm

    private lateinit var a: DenseMatrix
    private lateinit var sym: DenseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray

    private lateinit var y2: DoubleArray
    private lateinit var target: DenseMatrix
    private lateinit var xv: DenseVector
    private lateinit var yv: DenseVector

    private lateinit var triangular: DenseMatrix
    private lateinit var rhs: DoubleArray

    @Setup
    fun setup() {
        arm = DenseBenchmarkArm.resolve(denseArm)
        val rng = benchRng()
        a = randomMatrix(n, n, rng)
        sym = lowerSymmetricMatrix(n, rng)
        x = randomVector(n, rng)
        y = DoubleArray(n)
        y2 = randomVector(n, rng)
        target = randomMatrix(n, n, rng)
        xv = DenseVector.of(x)
        yv = DenseVector.of(y2)
        triangular = dominantMatrix(n, rng)
        rhs = DoubleArray(n)
    }

    @Benchmark
    fun gemv() {
        arm.external?.gemv(1.0, a, x, 0.0, y, false) ?: arm.context!!.gemv(1.0, a, x, 0.0, y)
    }

    @Benchmark
    fun gemvTransposed() {
        arm.external?.gemv(1.0, a, x, 0.0, y, true) ?: arm.context!!.gemv(1.0, a, x, 0.0, y, transpose = true)
    }

    @Benchmark
    fun symv() {
        arm.external?.symv(1.0, sym, x, 0.0, y, true) ?: arm.context!!.symv(1.0, sym, x, 0.0, y)
    }

    @Benchmark
    fun ger() {
        arm.external?.ger(NEAR_UNIT_SCALE, x, y2, target) ?: arm.context!!.ger(NEAR_UNIT_SCALE, x, y2, target)
    }

    @Benchmark
    fun syr() {
        arm.external?.syr(NEAR_UNIT_SCALE, x, target, true) ?: arm.context!!.syr(NEAR_UNIT_SCALE, xv, target)
    }

    @Benchmark
    fun syr2() {
        arm.external?.syr2(NEAR_UNIT_SCALE, x, y2, target, true) ?: arm.context!!.syr2(NEAR_UNIT_SCALE, xv, yv, target)
    }

    @Benchmark
    fun trsv() {
        x.copyInto(rhs)
        arm.external?.trsv(triangular, rhs, true, false, false) ?: arm.context!!.trsv(triangular, rhs, lower = true)
    }

    @Benchmark
    fun trmv() {
        x.copyInto(rhs)
        arm.external?.trmv(triangular, rhs, true, false, false) ?: arm.context!!.trmv(triangular, rhs, lower = true)
    }
}
