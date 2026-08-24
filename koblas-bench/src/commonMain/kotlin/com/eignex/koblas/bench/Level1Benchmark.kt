package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.asum
import com.eignex.koblas.axpy
import com.eignex.koblas.dot
import com.eignex.koblas.iamax
import com.eignex.koblas.koblas
import com.eignex.koblas.mathBackend
import com.eignex.koblas.norm2
import com.eignex.koblas.scale
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class Level1Benchmark {
    @Param("2", "4", "8", "16", "32", "64", "128", "256", "1024", "4096")
    var len: Int = 0

    // reference is the compiled-in kernels and forced is the host ones past their gate. No auto arm: a
    // level-1 gate routes either everything to one side or everything to the other.
    @Param(REFERENCE_BACKEND, FORCED_BACKEND)
    var backend: String = REFERENCE_BACKEND

    private lateinit var x: F64DenseVector
    private lateinit var y: F64DenseVector

    private lateinit var quad: DoubleArray
    private val quadOut = DoubleArray(4)

    @Setup
    fun setup() {
        installBackend(backend)
        println("resolved: primitives=$mathBackend")
        val rng = benchRng()
        x = F64DenseVector.of(randomVector(len, rng))
        y = F64DenseVector.of(randomVector(len, rng))
        quad = randomVector(4 * len, rng)
    }

    @Benchmark
    fun dot(): Double = x dot y

    @Benchmark
    fun axpyBench() {
        y.axpy(NEAR_UNIT_SCALE, x)
    }

    @Benchmark
    fun scaleBench() {
        y.scale(NEAR_UNIT_SCALE)
    }

    @Benchmark
    fun nrm2(): Double = x.norm2()

    @Benchmark
    fun asumBench(): Double = x.asum()

    @Benchmark
    fun iamaxBench(): Int = x.iamax()

    @Benchmark
    fun dot4(): Double {
        koblas.kernels.dot4(quad, 0, len, x.data, 0, len, quadOut, 0)
        return quadOut[0]
    }
}
