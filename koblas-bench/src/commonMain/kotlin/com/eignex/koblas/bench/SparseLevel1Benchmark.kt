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

    @Param(AUTOMATIC_KERNELS, SCALAR_KERNELS, C_KERNELS)
    var kernels: String = AUTOMATIC_KERNELS

    private lateinit var sparse: SparseVector
    private lateinit var other: SparseVector
    private lateinit var dense: DenseVector

    @Setup
    fun setup() {
        installKernelProvider(kernels)
        println("resolved: sparseKernels=${koblas.sparseKernels.name}")
        val rng = benchRng()
        sparse = randomSparseVector(len, density, rng)
        other = randomSparseVector(len, density, rng)
        dense = DenseVector.of(randomVector(len, rng))
    }

    @Benchmark
    fun sparseDotSparse(): Double = sparse dot other

    fun sparseDotDense(): Double = sparse dot dense

    fun sparseAxpy() {
        dense.axpy(NEAR_UNIT_SCALE, sparse)
    }

    @Benchmark
    fun sparseNrm2(): Double = sparse.norm2()

    @Benchmark
    fun sparseAsum(): Double = sparse.asum()

    fun sparseScatter() {
        koblas.sparseKernels.scatter(sparse, dense.data)
    }

    fun sparseGather() {
        koblas.sparseKernels.gather(sparse, dense.data)
    }

    fun sparseGatherZero() {
        koblas.sparseKernels.gatherZero(sparse, dense.data)
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
    private var builtIn: com.eignex.koblas.sparse.SparseKernels? = null
    private var oneMkl: SparseComparator? = null

    @Setup
    fun setup() {
        val rng = benchRng()
        sparse = randomSparseVector(len, density, rng)
        dense = DenseVector.of(randomVector(len, rng))
        gathered = DoubleArray(sparse.values.size)
        if (sparseArm == BUILTIN_BACKEND) builtIn = explicitBuiltInContext().sparseKernels else {
            oneMkl = checkNotNull(oneMklSparseComparator()) { "the benchmark-only oneMKL sparse comparator is unavailable" }
        }
        val identity = oneMkl?.identity ?: "built-in/${builtIn!!.name}"
        check(identity.startsWith(sparseArm)) { "sparse level-1 arm $sparseArm resolved $identity" }
        println("resolved: arm=$sparseArm sparseLevel1=$identity threading=${oneMkl?.threading ?: "single calling thread"}")
        if (oneMkl == null) {
            verifyNearZeroManagedAllocation("sparse-level1/$sparseArm/dot") {
                builtIn!!.dot(sparse, dense.data)
            }
        } else {
            reportAllocatingWorkload(
                "sparse-level1/$sparseArm/ffi",
                "benchmark foreign-function boundary wrappers",
            )
        }
    }

    @Benchmark
    fun sparseDotDense(): Double = oneMkl?.dot(sparse, dense.data) ?: builtIn!!.dot(sparse, dense.data)

    @Benchmark
    fun sparseAxpy() {
        oneMkl?.axpy(NEAR_UNIT_SCALE, sparse, dense.data) ?: builtIn!!.axpy(dense.data, NEAR_UNIT_SCALE, sparse)
    }

    @Benchmark
    fun sparseScatter() {
        oneMkl?.scatter(sparse, dense.data) ?: builtIn!!.scatter(sparse, dense.data)
    }

    @Benchmark
    fun sparseGather() {
        oneMkl?.gather(sparse, dense.data, gathered) ?: builtIn!!.gather(sparse, dense.data)
    }

    @Benchmark
    fun sparseGatherZero() {
        oneMkl?.gatherZero(sparse, dense.data, gathered) ?: builtIn!!.gatherZero(sparse, dense.data)
    }
}
