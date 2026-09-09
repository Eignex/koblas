package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.DenseVector
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class Level1Benchmark {
    // 63, 64, and 65 make the host-router crossover explicit in scalar/host comparisons.
    @Param("2", "4", "8", "16", "32", "63", "64", "65", "128", "256", "1024", "4096")
    var len: Int = 0

    @Param(BUILTIN_KERNELS, SCALAR_KERNELS, C_KERNELS)
    var kernels: String = BUILTIN_KERNELS

    private lateinit var x: DenseVector
    private lateinit var y: DenseVector
    private lateinit var modifiedRotation: ModifiedGivens
    private lateinit var rotation: Givens
    private lateinit var engine: KoblasContext

    private lateinit var quad: DoubleArray
    private val quadOut = DoubleArray(4)

    @Setup
    fun setup() {
        engine = kernelEngine(kernels)
        println("resolved: primitives=${engine.kernels.name}")
        val rng = benchRng()
        x = DenseVector.of(randomVector(len, rng))
        y = DenseVector.of(randomVector(len, rng))
        // Near-identity, like NEAR_UNIT_SCALE: rotmBench applies this every invocation with no per-call
        // reset, so a transformation with eigenvalues away from unit magnitude would blow x/y up to
        // Infinity/NaN partway through a trial.
        modifiedRotation = engine.kernels.rotmg(1.0, 1.0, 1.0, NEAR_UNIT_SCALE - 1.0)
        // Any rotation is safe to reapply without a per-call reset, unlike the modified one above: a plane
        // rotation is orthogonal, so repeated application preserves the magnitudes it started with.
        rotation = rotg(3.0, 4.0)
        quad = randomVector(4 * len, rng)
        verifyNearZeroManagedAllocation("level1/$kernels/axpy4") {
            engine.kernels.axpy4(
                y.data, 0, quad, 0, len,
                NEAR_UNIT_SCALE, -NEAR_UNIT_SCALE, NEAR_UNIT_SCALE, -NEAR_UNIT_SCALE, len,
            )
        }
        verifyNearZeroManagedAllocation("level1/$kernels/dotAxpy") {
            engine.kernels.dotAxpy(y.data, 0, NEAR_UNIT_SCALE, x.data, 0, quad, 0, len)
        }
        verifyNearZeroManagedAllocation("level1/$kernels/dot") { engine.kernels.dot(x.data, 0, y.data, 0, len) }
        verifyNearZeroManagedAllocation("level1/$kernels/ssqd") { engine.kernels.ssqd(x.data, 0, y.data, 0, len) }
        verifyNearZeroManagedAllocation("level1/$kernels/dot4") {
            engine.kernels.dot4(quad, 0, len, x.data, 0, len, quadOut, 0)
        }
    }

    @Benchmark
    fun dot(): Double = engine.kernels.dot(x.data, 0, y.data, 0, len)

    @Benchmark
    fun ssqd(): Double = engine.kernels.ssqd(x.data, 0, y.data, 0, len)

    @Benchmark
    fun axpyBench() {
        engine.kernels.axpy(y.data, 0, NEAR_UNIT_SCALE, x.data, 0, len)
    }

    @Benchmark
    fun scaleBench() {
        engine.kernels.scale(y.data, 0, NEAR_UNIT_SCALE, len)
    }

    @Benchmark
    fun nrm2(): Double = engine.kernels.nrm2(x.data, 0, len)

    @Benchmark
    fun sumBench(): Double = engine.kernels.sum(x.data, 0, len)

    @Benchmark
    fun compensatedSumBench(): Double = x.compensatedSum()

    @Benchmark
    fun asumBench(): Double = engine.kernels.asum(x.data, 0, len)

    @Benchmark
    fun iamaxBench(): Int = x.iamax()

    // Two loads and two stores an element against one load for the reductions, so this is the level-1
    // routine most likely to be bandwidth-bound rather than issue-bound, and the one where the host has
    // least room to win.
    @Benchmark
    fun swapBench() {
        swap(x, y)
    }

    @Benchmark
    fun rotmgBench(): ModifiedGivens = engine.kernels.rotmg(1.0, 1.0, 2.0, 1.0)

    @Benchmark
    fun rotmBench() {
        engine.kernels.rotm(x.data, 0, 1, y.data, 0, 1, len, modifiedRotation)
    }

    @Benchmark
    fun rotBench() {
        engine.kernels.rot(x.data, 0, y.data, 0, len, rotation.c, rotation.s)
    }

    @Benchmark
    fun dot4(): Double {
        engine.kernels.dot4(quad, 0, len, x.data, 0, len, quadOut, 0)
        return quadOut[0]
    }

    @Benchmark
    fun axpy4(): Double {
        engine.kernels.axpy4(
            y.data, 0, quad, 0, len,
            NEAR_UNIT_SCALE, -NEAR_UNIT_SCALE, NEAR_UNIT_SCALE, -NEAR_UNIT_SCALE, len,
        )
        return y.data[0]
    }

    @Benchmark
    fun axpy4AsFourAxpy(): Double {
        engine.kernels.axpy(y.data, 0, NEAR_UNIT_SCALE, quad, 0, len)
        engine.kernels.axpy(y.data, 0, -NEAR_UNIT_SCALE, quad, len, len)
        engine.kernels.axpy(y.data, 0, NEAR_UNIT_SCALE, quad, 2 * len, len)
        engine.kernels.axpy(y.data, 0, -NEAR_UNIT_SCALE, quad, 3 * len, len)
        return y.data[0]
    }

    @Benchmark
    fun dotAxpy(): Double = engine.kernels.dotAxpy(y.data, 0, NEAR_UNIT_SCALE, x.data, 0, quad, 0, len)

    @Benchmark
    fun dotAxpySeparate(): Double {
        val result = engine.kernels.dot(x.data, 0, quad, 0, len)
        engine.kernels.axpy(y.data, 0, NEAR_UNIT_SCALE, x.data, 0, len)
        return result
    }
}
