package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.koblas
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class Level2Benchmark {
    @Param("16", "64", "256", "1024", "2048")
    var n: Int = 0

    @Param(AUTO_BACKEND, REFERENCE_BACKEND, FORCED_BACKEND)
    var backend: String = AUTO_BACKEND

    private lateinit var a: F64DenseMatrix
    private lateinit var sym: F64DenseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray

    private lateinit var y2: DoubleArray
    private lateinit var target: F64DenseMatrix
    private lateinit var xv: F64DenseVector
    private lateinit var yv: F64DenseVector

    private lateinit var triangular: F64DenseMatrix
    private lateinit var rhs: DoubleArray

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = benchRng()
        a = randomMatrix(n, n, rng)
        sym = lowerSymmetricMatrix(n, rng)
        x = randomVector(n, rng)
        y = DoubleArray(n)
        y2 = randomVector(n, rng)
        target = randomMatrix(n, n, rng)
        xv = F64DenseVector.of(x)
        yv = F64DenseVector.of(y2)
        triangular = dominantMatrix(n, rng)
        rhs = DoubleArray(n)
    }

    @Benchmark
    fun gemv() {
        koblas.gemv(1.0, a, x, 0.0, y)
    }

    @Benchmark
    fun gemvTransposed() {
        koblas.gemv(1.0, a, x, 0.0, y, transpose = true)
    }

    @Benchmark
    fun symv() {
        koblas.symv(1.0, sym, x, 0.0, y)
    }

    @Benchmark
    fun ger() {
        koblas.ger(NEAR_UNIT_SCALE, x, y2, target)
    }

    @Benchmark
    fun syr() {
        koblas.syr(NEAR_UNIT_SCALE, xv, target)
    }

    @Benchmark
    fun syr2() {
        koblas.syr2(NEAR_UNIT_SCALE, xv, yv, target)
    }

    @Benchmark
    fun trsv() {
        x.copyInto(rhs)
        koblas.trsv(triangular, rhs, lower = true)
    }

    @Benchmark
    fun trmv() {
        x.copyInto(rhs)
        koblas.trmv(triangular, rhs, lower = true)
    }
}
