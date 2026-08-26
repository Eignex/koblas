package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseVector
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseLevel1Benchmark {
    @Param("256", "4096", "65536")
    var len: Int = 0

    @Param("0.001", "0.01", "0.1")
    var density: Double = 0.0

    private lateinit var sparse: F64SparseVector
    private lateinit var other: F64SparseVector
    private lateinit var dense: F64DenseVector

    @Setup
    fun setup() {
        val rng = benchRng()
        sparse = randomSparseVector(len, density, rng)
        other = randomSparseVector(len, density, rng)
        dense = F64DenseVector.of(randomVector(len, rng))
    }

    @Benchmark
    fun sparseDotSparse(): Double = sparse dot other

    @Benchmark
    fun sparseDotDense(): Double = sparse dot dense

    @Benchmark
    fun sparseAxpy() {
        dense.axpy(NEAR_UNIT_SCALE, sparse)
    }

    @Benchmark
    fun sparseNrm2(): Double = sparse.norm2()

    @Benchmark
    fun sparseAsum(): Double = sparse.asum()

    @Benchmark
    fun sparseScatter() {
        koblas.sparseKernels.scatter(sparse, dense.data)
    }

    @Benchmark
    fun sparseGather() {
        koblas.sparseKernels.gather(sparse, dense.data)
    }

    @Benchmark
    fun sparseGatherZero() {
        koblas.sparseKernels.gatherZero(sparse, dense.data)
    }
}
