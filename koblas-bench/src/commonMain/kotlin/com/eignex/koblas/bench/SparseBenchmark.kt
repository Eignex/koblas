package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
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

    private lateinit var a: F64SparseMatrix
    private lateinit var rhs: DoubleArray

    private lateinit var x: DoubleArray
    private lateinit var rankX: F64SparseVector
    private lateinit var rankY: F64DenseVector

    private lateinit var luFactored: F64SparseFactorization

    /** The portable factorization set to scale rows, which is where equilibration lives now. */
    private val equilibrating = F64ReferenceSparseDecompositions(equilibrate = true)

    @Setup
    fun setup() {
        val rng = benchRng()
        a = sparseDominantMatrix(n, rng)
        rhs = randomVector(n, rng)
        luFactored = a.lu()
        x = DoubleArray(n)
        val rankNnz = (n + 3) / 4
        rankX = F64SparseVector.of(
            n,
            IntArray(rankNnz) { it * 4 },
            DoubleArray(rankNnz) { rng.nextDouble(-1.0, 1.0) },
        )
        rankY = F64DenseVector.of(randomVector(n, rng))
    }

    @Benchmark
    fun sparseLuSolve(): DoubleArray = equilibrating.factor(a).solve(rhs)

    @Benchmark
    fun sparseLuFtran(): DoubleArray = luFactored.solveInto(rhs, x)

    @Benchmark
    fun sparseLuBtran(): DoubleArray = luFactored.solveInto(rhs, x, transpose = true)

    @Benchmark
    fun sparseGemv(): DoubleArray = koblas.gemv(a, rhs)

    @Benchmark
    fun sparseGemvTransposed(): DoubleArray = koblas.gemv(a, rhs, transpose = true)

    @Benchmark
    fun sparseLuFactor(): F64SparseFactorization = a.lu()

    @Benchmark
    fun sparseLuFactorEquilibrated(): F64SparseFactorization =
        equilibrating.factor(a)

    @Benchmark
    fun sparseTrsv(): DoubleArray {
        rhs.copyInto(x)
        a.trsv(x, lower = true)
        return x
    }

    @Benchmark
    fun sparseTranspose(): F64SparseMatrix = a.transpose()

    @Benchmark
    fun sparseSyr(): F64SparseMatrix = a.syr(NEAR_UNIT_SCALE, rankX)

    @Benchmark
    fun sparseSyr2(): F64SparseMatrix = a.syr2(NEAR_UNIT_SCALE, rankX, rankY)
}
