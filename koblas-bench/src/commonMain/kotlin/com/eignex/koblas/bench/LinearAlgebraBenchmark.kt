package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LuDecomposition
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.SparseLu
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Uplo
import com.eignex.koblas.installLinearAlgebra
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

    /** `auto` is whatever the platform resolves (OpenBLAS on the benchmark classpath). */
    @Param("auto", "reference")
    var backend: String = "auto"

    private val nrhs = 8

    private lateinit var a: DenseMatrix
    private lateinit var b: DenseMatrix
    private lateinit var rhs: DoubleArray
    private lateinit var block: DenseMatrix
    private lateinit var w: DenseMatrix
    private lateinit var factored: LuDecomposition

    @Setup
    fun setup() {
        installLinearAlgebra(if (backend == "reference") ReferenceLinearAlgebra else null)
        val rng = Random(20260716)
        a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        for (i in 0 until n) a[i, i] = a[i, i] + n // diagonally dominant ⇒ well-conditioned
        b = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        rhs = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        block = DenseMatrix.wrap(n, nrhs, DoubleArray(n * nrhs) { rng.nextDouble(-1.0, 1.0) })
        w = DenseMatrix(n, n)
        factored = a.lu()
    }

    /** LINPACK-style dense solve: LU factorize then back/forward solve. */
    @Benchmark
    fun luSolve(): DoubleArray = a.lu().solve(rhs)

    /** All right-hand sides in one block solve against a prefactored basis. */
    @Benchmark
    fun luSolveBlock(): DenseMatrix = koblas.solve(factored, block)

    /** The pre-block alternative: one vector solve per column, extraction included. */
    @Benchmark
    fun luSolveColumns(): DoubleArray {
        var last = rhs
        for (c in 0 until nrhs) {
            val col = DoubleArray(n) { block[it, c] }
            last = koblas.solve(factored, col)
        }
        return last
    }

    @Benchmark
    fun gemm(): DenseMatrix = a.matMul(b)

    @Benchmark
    fun gemv(): DoubleArray = koblas.gemv(a, rhs)

    /** The full-symmetric extension: both triangles written, exactly symmetric. */
    @Benchmark
    fun syrkFull(): DenseMatrix {
        koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = w)
        return w
    }

    /** Strict BLAS dsyrk: one triangle only, no scratch mirror in the native backends. */
    @Benchmark
    fun syrkLower(): DenseMatrix {
        koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = w, uplo = Uplo.LOWER)
        return w
    }
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
