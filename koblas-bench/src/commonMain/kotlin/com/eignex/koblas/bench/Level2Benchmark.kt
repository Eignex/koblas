package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.koblas
import kotlin.random.Random
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class Level2Benchmark {
    @Param("16", "64", "256", "1024", "2048")
    var n: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var a: DenseMatrix
    private lateinit var sym: DenseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray

    /** A second operand for the rank-1 and rank-2 updates, and the matrix they write into. */
    private lateinit var y2: DoubleArray
    private lateinit var target: DenseMatrix
    private lateinit var xv: DenseVector
    private lateinit var yv: DenseVector

    /** Unit-diagonal-free lower triangle, so the solve has a real diagonal to divide by. */
    private lateinit var triangular: DenseMatrix
    private lateinit var rhs: DoubleArray

    @Setup
    fun setup() {
        installBackend(backend)
        val rng = Random(BENCH_SEED)
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

    /** Rank-1 update, the one level-2 routine that writes the whole matrix rather than a vector. */
    @Benchmark
    fun ger() {
        koblas.ger(1.000001, x, y2, target)
    }

    @Benchmark
    fun syr() {
        koblas.syr(1.000001, xv, target)
    }

    @Benchmark
    fun syr2() {
        koblas.syr2(1.000001, xv, yv, target)
    }

    /** Triangular solve, sequential down the columns and the hardest level-2 routine to accelerate. */
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
