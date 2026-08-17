package com.eignex.koblas.bench

import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseVector
import com.eignex.koblas.asum
import com.eignex.koblas.axpy
import com.eignex.koblas.dot
import com.eignex.koblas.norm2
import kotlinx.benchmark.*

/**
 * The sparse vector primitives, which cost by stored entry rather than by length. [density] is what decides
 * whether a sparse operand beats the dense one, so it is the parameter that matters here.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseLevel1Benchmark {
    @Param("256", "4096", "65536")
    var len: Int = 0

    @Param("0.001", "0.01", "0.1")
    var density: Double = 0.0

    private lateinit var sparse: SparseVector
    private lateinit var other: SparseVector
    private lateinit var dense: DenseVector

    @Setup
    fun setup() {
        val rng = benchRng()
        sparse = randomSparseVector(len, density, rng)
        other = randomSparseVector(len, density, rng)
        dense = DenseVector.of(randomVector(len, rng))
    }

    /** Both operands sparse, which walks the two index lists in step. */
    @Benchmark
    fun sparseDotSparse(): Double = sparse dot other

    /** One sparse operand against a dense one, which indexes straight into the dense side. */
    @Benchmark
    fun sparseDotDense(): Double = sparse dot dense

    @Benchmark
    fun sparseAxpy() {
        axpy(dense, NEAR_UNIT_SCALE, sparse)
    }

    @Benchmark
    fun sparseNrm2(): Double = norm2(sparse)

    @Benchmark
    fun sparseAsum(): Double = asum(sparse)
}
