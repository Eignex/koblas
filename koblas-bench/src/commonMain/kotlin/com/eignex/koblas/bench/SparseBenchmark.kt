package com.eignex.koblas.bench

import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.koblas
import com.eignex.koblas.syr
import com.eignex.koblas.syr2
import com.eignex.koblas.sparse.*
import com.eignex.koblas.transpose
import kotlinx.benchmark.*

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    private lateinit var a: SparseMatrix
    private lateinit var rhs: DoubleArray

    private lateinit var x: DoubleArray
    private lateinit var multiplied: DoubleArray
    private lateinit var rankX: SparseVector
    private lateinit var rankY: DenseVector

    @Setup
    fun setup() {
        val rng = benchRng()
        a = sparseDominantMatrix(n, rng)
        rhs = randomVector(n, rng)
        x = DoubleArray(n)
        multiplied = DoubleArray(n)
        val rankNnz = (n + 3) / 4
        rankX = SparseVector.of(
            n,
            IntArray(rankNnz) { it * 4 },
            DoubleArray(rankNnz) { rng.nextDouble(-1.0, 1.0) },
        )
        rankY = DenseVector.of(randomVector(n, rng))
    }

    @Benchmark
    fun sparseGemv(): DoubleArray = koblas.gemv(a, rhs)

    @Benchmark
    fun sparseGemvTransposed(): DoubleArray = koblas.gemv(a, rhs, transpose = true)

    @Benchmark
    fun sparseTrsv(): DoubleArray {
        rhs.copyInto(x)
        a.trsv(x, lower = true)
        return x
    }

    @Benchmark
    fun sparseTrmv(): DoubleArray {
        rhs.copyInto(multiplied)
        a.trmv(multiplied, lower = true)
        return multiplied
    }

    @Benchmark
    fun sparseTranspose(): SparseMatrix = a.transpose()

    @Benchmark
    fun sparseSyr(): SparseMatrix = a.syr(NEAR_UNIT_SCALE, rankX)

    @Benchmark
    fun sparseSyr2(): SparseMatrix = a.syr2(NEAR_UNIT_SCALE, rankX, rankY)
}
