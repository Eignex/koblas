package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import kotlinx.benchmark.*

/** Retained sparse matrix coverage, separating one-shot conversion from reusable prepared products. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseCompletionBenchmark {
    @Param("32", "129", "256") var n: Int = 32
    @Param("0.01", "0.1") var density: Double = 0.01
    @Param("true", "false") var lower: Boolean = true
    @Param("left") var side: String = "left"
    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND) var sparseArm: String = BUILTIN_BACKEND

    private lateinit var arm: SparseBenchmarkArm
    private lateinit var resources: BenchmarkResources
    private lateinit var symmetric: SparseMatrix
    private lateinit var a: SparseMatrix
    private lateinit var b: SparseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray
    private lateinit var rhs: DenseMatrix
    private lateinit var c: DenseMatrix
    private lateinit var preparedSymmetric: PreparedSparseBenchmarkArm
    private val right: Boolean get() = side == "right"

    @Setup
    fun setup() {
        val rng = benchRng()
        arm = SparseBenchmarkArm.resolve(sparseArm)
        resources = BenchmarkResources()
        a = sparseComparisonMatrix(n, n / 2 + 1, density, "regular", rng)
        b = sparseComparisonMatrix(n / 2 + 1, n, density, "regular", rng)
        symmetric = selectedSymmetricMatrix(n, density, lower)
        x = randomVector(n, rng)
        y = DoubleArray(n)
        check(side == "left" || side == "right") { "unknown symmetric side: $side" }
        rhs = if (right) randomMatrix(8, n, rng) else randomMatrix(n, 8, rng)
        c = DenseMatrix.zero(n)
        preparedSymmetric = resources.acquire {
            arm.prepare(symmetric, SparseDescriptor(symmetric = true, lower = lower))
        }
        reportAllocatingWorkload("sparse-completion/$sparseArm/$n/$density", "one-shot conversion and fresh sparse results remain timed")
    }

    @TearDown
    fun tearDown() {
        resources.close()
    }

    @Benchmark
    fun symv(): DoubleArray {
        arm.symv(1.0, symmetric, x, 0.0, y, lower)
        return y
    }

    @Benchmark
    fun preparedSymv(): DoubleArray {
        preparedSymmetric.symv(1.0, x, 0.0, y)
        return y
    }

    @Benchmark
    fun symm(): DenseMatrix {
        val out = DenseMatrix.zero(rhs.rows, rhs.cols)
        arm.symm(1.0, symmetric, rhs, 0.0, out, lower, right)
        return out
    }

    @Benchmark
    fun preparedSymm(): DenseMatrix {
        val out = DenseMatrix.zero(rhs.rows, rhs.cols)
        preparedSymmetric.symm(1.0, rhs, 0.0, out, right)
        return out
    }

    @Benchmark
    fun sparseProductScaledTransposed(): SparseMatrix = arm.scaledTransposedProduct(-0.75, a)

    @Benchmark
    fun denseProduct(): DenseMatrix {
        arm.denseProduct(1.0, a, b, 0.0, c)
        return c
    }

    @Benchmark
    fun syrkDense(): DenseMatrix {
        arm.syrkDense(1.0, a, 0.0, c)
        return c
    }

    @Benchmark
    fun syrkSparse(): SparseMatrix = arm.syrkSparse(a)

    @Benchmark
    fun addScaled(): SparseMatrix = arm.addScaled(-0.75, a)
}

private fun selectedSymmetricMatrix(n: Int, density: Double, lower: Boolean): SparseMatrix {
    val columns = List(n) { j ->
        buildList {
            add(j to (j + 2.0))
            for (i in 0 until n) {
                if (i == j || (if (lower) i < j else i > j)) continue
                if (((i * 31 + j * 17) and 1023) < density * 1024) add(i to ((i + j) % 7 - 3.0))
            }
        }
    }
    return SparseMatrix.ofColumns(n, n, columns)
}
