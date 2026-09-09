package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.gemvInto
import kotlinx.benchmark.*

/**
 * The destination-passing matvec against a sparse right-hand side, where the per-entry run is `rows` long.
 * Small row counts are the interesting ones: a classifier's weight matrix has one row per class, so the run
 * each stored entry drives is far shorter than the length a kernel call earns back.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class GemvIntoBenchmark {
    @Param("2", "8", "32", "128")
    var rows: Int = 0

    @Param("64")
    var cols: Int = 0

    private lateinit var a: DenseMatrix
    private lateinit var sparseX: SparseVector
    private lateinit var out: DoubleArray

    @Setup
    fun setup() {
        val rng = benchRng()
        a = randomMatrix(rows, cols, rng)
        sparseX = randomSparseVector(cols, density = 0.25, rng = rng)
        out = DoubleArray(rows)
    }

    @Benchmark
    fun gemvIntoSparseX(): DoubleArray {
        a.gemvInto(sparseX, out)
        return out
    }
}
