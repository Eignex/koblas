package com.eignex.koblas.bench

import com.eignex.koblas.DenseVector
import com.eignex.koblas.asum
import com.eignex.koblas.axpy
import com.eignex.koblas.dot
import com.eignex.koblas.iamax
import com.eignex.koblas.koblas
import com.eignex.koblas.mathBackend
import com.eignex.koblas.norm2
import com.eignex.koblas.scale
import kotlinx.benchmark.*

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

    /** [HOST_KERNELS] installs the platform's level-1 kernels when it has any, [BUILTIN_KERNELS] clears them. */
    @Param(BUILTIN_KERNELS, HOST_KERNELS)
    var kernels: String = BUILTIN_KERNELS

    private lateinit var x: DenseVector
    private lateinit var y: DenseVector

    /** Four contiguous runs of [len], the layout `dot4` expects. */
    private lateinit var quad: DoubleArray
    private val quadOut = DoubleArray(4)

    @Setup
    fun setup() {
        useHostLevel1(kernels == HOST_KERNELS)
        println("resolved: primitives=$mathBackend")
        val rng = benchRng()
        x = DenseVector.of(randomVector(len, rng))
        y = DenseVector.of(randomVector(len, rng))
        quad = randomVector(4 * len, rng)
    }

    @Benchmark
    fun dot(): Double = x dot y

    @Benchmark
    fun axpyBench() {
        axpy(y, NEAR_UNIT_SCALE, x)
    }

    @Benchmark
    fun scaleBench() {
        scale(y, NEAR_UNIT_SCALE)
    }

    /** Rescales rather than summing squares, so it costs more than a dot of the same length. */
    @Benchmark
    fun nrm2(): Double = norm2(x)

    @Benchmark
    fun asumBench(): Double = asum(x)

    @Benchmark
    fun iamaxBench(): Int = iamax(x)

    /** Four dots against one shared operand, which loads that operand once rather than four times. */
    @Benchmark
    fun dot4(): Double {
        koblas.vectorKernels.dot4(quad, 0, len, x.data, 0, len, quadOut, 0)
        return quadOut[0]
    }
}
