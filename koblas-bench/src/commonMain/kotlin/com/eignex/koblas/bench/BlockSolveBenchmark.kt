package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LuDecomposition
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.installLinearAlgebra
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo
import com.eignex.koblas.lu
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
 * Block solve against a loop of vector solves as the right-hand-side count grows, which is where the
 * two strategies trade places: a block solve reads the factor once but updates rows in `nrhs`-length
 * runs, while per-column solves re-read the factor once per column and get full-length contiguous
 * runs. Locates the crossover the block-solve dispatch should use.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class BlockSolveBenchmark {
    @Param("64", "256")
    var n: Int = 0

    @Param("1", "2", "4", "8", "16", "32", "64")
    var nrhs: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var factored: LuDecomposition
    private lateinit var block: DenseMatrix

    @Setup
    fun setup() {
        installLinearAlgebra(if (backend == "reference") ReferenceLinearAlgebra else null)
        println("resolved: $koblasInfo")
        val rng = Random(20260729)
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        for (i in 0 until n) a[i, i] = a[i, i] + n
        factored = a.lu()
        block = DenseMatrix.wrap(n, nrhs, DoubleArray(n * nrhs) { rng.nextDouble(-1.0, 1.0) })
    }

    @Benchmark
    fun blockSolve(): DenseMatrix = koblas.solve(factored, block)

    @Benchmark
    fun columnSolves(): DoubleArray {
        var last = DoubleArray(0)
        for (c in 0 until nrhs) {
            val col = DoubleArray(n) { block[it, c] }
            last = koblas.solve(factored, col)
        }
        return last
    }
}
