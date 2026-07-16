package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseLu
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.lu
import com.eignex.koblas.matMul
import com.eignex.koblas.solve
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State
import kotlin.random.Random

/**
 * Dense linear-algebra throughput: LINPACK-style `A x = b` solve (LU factor + solve), matrix–matrix and
 * matrix–vector products, over the active [koblas] backend at a few sizes. Reports average time per op.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class DenseBenchmark {
    @Param("16", "64", "256")
    var n: Int = 0

    private lateinit var a: DenseMatrix
    private lateinit var b: DenseMatrix
    private lateinit var rhs: DoubleArray

    @Setup
    fun setup() {
        val rng = Random(20260716)
        a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        for (i in 0 until n) a[i, i] = a[i, i] + n // diagonally dominant ⇒ well-conditioned
        b = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        rhs = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    }

    /** LINPACK-style dense solve: LU factorize then back/forward solve. */
    @Benchmark
    fun luSolve(): DoubleArray = a.lu().solve(rhs)

    @Benchmark
    fun gemm(): DenseMatrix = a.matMul(b)

    @Benchmark
    fun gemv(): DoubleArray = koblas.gemv(a, rhs)
}

/**
 * Sparse linear-algebra throughput: factorize a sparse diagonally-dominant matrix and solve, at a few
 * sizes and a fixed off-diagonal density.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    private lateinit var a: SparseMatrix
    private lateinit var rhs: DoubleArray

    @Setup
    fun setup() {
        val rng = Random(20260716)
        val columns = List(n) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            entries.add(j to (rng.nextDouble(-1.0, 1.0) + n)) // dominant diagonal
            for (i in 0 until n) if (i != j && rng.nextDouble() < 0.01) entries.add(i to rng.nextDouble(-1.0, 1.0))
            entries
        }
        a = SparseMatrix.ofColumns(n, n, columns)
        rhs = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    }

    /** Sparse LU factorize + FTRAN solve. */
    @Benchmark
    fun sparseLuSolve(): DoubleArray {
        val lu = SparseLu.factorize(a, equilibrate = true) ?: return rhs
        return lu.ftran(rhs)
    }

    @Benchmark
    fun sparseGemv(): DoubleArray = a.gemv(rhs)
}
