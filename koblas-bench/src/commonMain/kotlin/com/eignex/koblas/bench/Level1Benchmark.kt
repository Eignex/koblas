package com.eignex.koblas.bench

import com.eignex.koblas.DenseVector
import com.eignex.koblas.axpy
import com.eignex.koblas.dot
import com.eignex.koblas.mathBackend
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

/**
 * The vector path pays a fixed setup, lane broadcast and horizontal reduce, and does no vector work below
 * one lane width. On the JVM, re-run with `-Pkoblas.noSimd=true` to compare SIMD against scalar.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class Level1Benchmark {
    @Param("2", "4", "8", "16", "32", "64", "128", "256", "1024", "4096")
    var len: Int = 0

    /** host installs the platform's level-1 kernels when it has any, builtin clears them. */
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
