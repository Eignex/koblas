package com.eignex.koblas.bench

import com.eignex.koblas.DenseVector
import com.eignex.koblas.axpy
import com.eignex.koblas.dot
import com.eignex.koblas.mathBackend
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State
import kotlin.random.Random

/**
 * Level-1 kernel cost across vector lengths, which locates the SIMD-versus-scalar crossover: the vector
 * path pays a fixed setup (lane broadcast, horizontal reduce) and does no vector work at all below one
 * lane width.
 *
 * These kernels sit below the [com.eignex.koblas.LinearAlgebra] seam, so the `backend` parameter used by
 * the other suites does not reach them. Two switches do. On the JVM, run the sweep twice, once as-is and
 * once with `-Pkoblas.noSimd=true` to withhold the incubator module, to compare SIMD against scalar. On
 * the other targets the `kernels` parameter installs or clears host level-1 kernels, which is how the
 * length threshold that decides between a foreign call and the built-in loop was measured.
 *
 * [mathBackend] is printed to confirm which kernels each run actually measured.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class Level1Benchmark {
    @Param("2", "4", "8", "16", "32", "64", "128", "256", "1024", "4096")
    var len: Int = 0

    /** `host` installs the platform's level-1 kernels when it has any; `builtin` clears them. */
    @Param("builtin", "host")
    var kernels: String = "builtin"

    private lateinit var x: DenseVector
    private lateinit var y: DenseVector

    @Setup
    fun setup() {
        useHostLevel1(kernels == "host")
        println("resolved: primitives=$mathBackend")
        val rng = Random(BENCH_SEED)
        x = DenseVector.of(randomVector(len, rng))
        y = DenseVector.of(randomVector(len, rng))
    }

    @Benchmark
    fun dot(): Double = x dot y

    @Benchmark
    fun axpyBench() {
        axpy(y, 1.000001, x)
    }
}
