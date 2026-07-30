package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LdlDecomposition
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.installLinearAlgebra
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo
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
 * The symmetric indefinite vector solve, against a prefactored matrix.
 *
 * This is the one routine left where a native backend still handles `O(n²)` work over `O(n²)` data — the
 * shape that made gemv, symv and the LU vector solve lose to the portable kernels once measured, because
 * per-call marshalling cannot be amortized. Whether `dsytrs` behaves the same way was never measured.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class LdlSolveBenchmark {
    @Param("64", "256", "1024")
    var n: Int = 0

    @Param("auto", "reference")
    var backend: String = "auto"

    private lateinit var factorization: LdlDecomposition
    private lateinit var b: DoubleArray
    private lateinit var x: DoubleArray

    @Setup
    fun setup() {
        installLinearAlgebra(if (backend == "reference") ReferenceLinearAlgebra else null)
        println("resolved: $koblasInfo")
        val rng = Random(20260730)
        // Symmetric indefinite: mixed-sign diagonal shifts force the 2x2 Bunch-Kaufman pivots.
        val a = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var v = rng.nextDouble(-1.0, 1.0)
                if (i == j) v += if (i % 2 == 0) n.toDouble() else -n.toDouble()
                a[i, j] = v
                a[j, i] = v
            }
        }
        factorization = koblas.ldl(a)
        b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        x = DoubleArray(n)
    }

    @Benchmark
    fun ldlSolve(): DoubleArray = koblas.solveInto(factorization, b, x)
}
