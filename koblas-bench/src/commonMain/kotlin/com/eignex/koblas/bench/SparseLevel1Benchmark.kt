package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseVector
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseLevel1Benchmark {
    @Param("256", "4096", "65536")
    var len: Int = 0

    @Param("0.001", "0.01", "0.1")
    var density: Double = 0.0

    @Param(BUILTIN_KERNELS, SCALAR_KERNELS, C_KERNELS)
    var kernels: String = BUILTIN_KERNELS

    private lateinit var sparse: SparseVector
    private lateinit var other: SparseVector
    private lateinit var dense: DenseVector
    private lateinit var engine: KoblasContext

    @Setup
    fun setup() {
        engine = kernelEngine(kernels)
        val rng = benchRng()
        sparse = randomSparseVector(len, density, rng)
        other = randomSparseVector(len, density, rng)
        dense = DenseVector.of(randomVector(len, rng))
    }

    @Benchmark
    fun sparseDotSparse(): Double = engine.sparseKernels.dot(sparse, other)

    fun sparseDotDense(): Double = engine.sparseKernels.dot(sparse, dense.data)

    fun sparseAxpy() {
        engine.sparseKernels.axpy(dense.data, NEAR_UNIT_SCALE, sparse)
    }

    @Benchmark
    fun sparseNrm2(): Double = engine.sparseKernels.nrm2(sparse)

    @Benchmark
    fun sparseAsum(): Double = engine.sparseKernels.asum(sparse)

    fun sparseScatter() {
        engine.sparseKernels.scatter(sparse, dense.data)
    }

    fun sparseGather() {
        engine.sparseKernels.gather(sparse, dense.data)
    }

    fun sparseGatherZero() {
        engine.sparseKernels.gatherZero(sparse, dense.data)
    }
}

/** Direct oneMKL legacy sparse-BLAS counterparts and the same explicitly selected built-in kernels. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseLevel1ComparisonBenchmark {
    @Param("256", "4096", "65536")
    var len: Int = 0

    @Param("0.001", "0.01", "0.1")
    var density: Double = 0.0

    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND)
    var sparseArm: String = BUILTIN_BACKEND

    private lateinit var sparse: SparseVector
    private lateinit var dense: DenseVector
    private lateinit var gathered: DoubleArray
    private lateinit var arm: SparseBenchmarkArm

    @Setup
    fun setup() {
        val rng = benchRng()
        sparse = randomSparseVector(len, density, rng)
        dense = DenseVector.of(randomVector(len, rng))
        gathered = DoubleArray(sparse.values.size)
        arm = SparseBenchmarkArm.resolve(sparseArm)
        if (sparseArm == BUILTIN_BACKEND) {
            verifyNearZeroManagedAllocation("sparse-level1/$sparseArm/dot") {
                arm.dot(sparse, dense.data)
            }
        } else {
            reportAllocatingWorkload(
                "sparse-level1/$sparseArm/ffi",
                "benchmark foreign-function boundary wrappers",
            )
        }
    }

    @Benchmark
    fun sparseDotDense(): Double = arm.dot(sparse, dense.data)

    @Benchmark
    fun sparseAxpy() {
        arm.axpy(NEAR_UNIT_SCALE, sparse, dense.data)
    }

    @Benchmark
    fun sparseScatter() {
        arm.scatter(sparse, dense.data)
    }

    @Benchmark
    fun sparseGather() {
        arm.gather(sparse, dense.data, gathered)
    }

    @Benchmark
    fun sparseGatherZero() {
        arm.gatherZero(sparse, dense.data, gathered)
    }
}
