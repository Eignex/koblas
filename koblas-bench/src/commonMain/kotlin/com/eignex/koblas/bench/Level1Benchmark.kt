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
 * Level-1 kernel cost across vector lengths, for locating the SIMD-versus-scalar crossover: the
 * vector path pays a fixed setup (lane broadcast, horizontal reduce) and does no vector work at all
 * below one lane width. Run the same sweep twice, once with the incubator module on the fork and
 * once without (`-Pkoblas.noSimd=true`), and compare; [mathBackend] is printed to confirm which
 * kernel each run measured.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class Level1Benchmark {
    @Param("2", "3", "4", "8", "16", "64", "256")
    var len: Int = 0

    private lateinit var x: DenseVector
    private lateinit var y: DenseVector

    @Setup
    fun setup() {
        println("resolved: primitives=$mathBackend")
        val rng = Random(20260728)
        x = DenseVector.of(DoubleArray(len) { rng.nextDouble(-1.0, 1.0) })
        y = DenseVector.of(DoubleArray(len) { rng.nextDouble(-1.0, 1.0) })
    }

    @Benchmark
    fun dot(): Double = x dot y

    @Benchmark
    fun axpyBench() {
        axpy(y, 1.000001, x)
    }
}
