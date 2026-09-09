package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.PreparedSparseMatrix
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

    private var builtIn: com.eignex.koblas.sparse.SparseBlas? = null
    private var external: SparseComparator? = null
    private lateinit var symmetric: SparseMatrix
    private lateinit var a: SparseMatrix
    private lateinit var b: SparseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray
    private lateinit var rhs: DenseMatrix
    private lateinit var c: DenseMatrix
    private lateinit var prepared: PreparedSparseMatrix
    private var externalSymmetric: PreparedSparseComparator? = null
    private val right: Boolean get() = side == "right"

    @Setup
    fun setup() {
        val rng = benchRng()
        if (sparseArm == BUILTIN_BACKEND) {
            builtIn = explicitBuiltInContext().sparseBlas
            println("resolved: arm=$sparseArm sparse=${builtIn!!.name} threading=single calling thread")
        } else {
            external = checkNotNull(oneMklSparseComparator()) { "the benchmark-only oneMKL sparse comparator is unavailable" }
            println("resolved: arm=$sparseArm sparse=${external!!.identity} threading=${external!!.threading}")
        }
        a = sparseComparisonMatrix(n, n / 2 + 1, density, "regular", rng)
        b = sparseComparisonMatrix(n / 2 + 1, n, density, "regular", rng)
        symmetric = selectedSymmetricMatrix(n, density, lower)
        x = randomVector(n, rng)
        y = DoubleArray(n)
        check(side == "left" || side == "right") { "unknown symmetric side: $side" }
        rhs = if (right) randomMatrix(8, n, rng) else randomMatrix(n, 8, rng)
        c = DenseMatrix.zero(n)
        if (builtIn != null) prepared = builtIn!!.prepare(symmetric)
        else externalSymmetric = external!!.prepare(symmetric, symmetric = true, lower = lower)
        reportAllocatingWorkload("sparse-completion/$sparseArm/$n/$density", "one-shot conversion and fresh sparse results remain timed")
    }

    @TearDown
    fun tearDown() {
        if (::prepared.isInitialized) prepared.close()
        externalSymmetric?.close()
    }

    @Benchmark
    fun symv(): DoubleArray {
        external?.prepare(symmetric, symmetric = true, lower = lower)?.use { it.symv(1.0, x, 0.0, y) }
            ?: builtIn!!.symv(1.0, symmetric, x, 0.0, y, lower)
        return y
    }

    @Benchmark
    fun preparedSymv(): DoubleArray {
        externalSymmetric?.symv(1.0, x, 0.0, y) ?: prepared.symv(1.0, x, 0.0, y, lower)
        return y
    }

    @Benchmark
    fun symm(): DenseMatrix {
        val out = DenseMatrix.zero(rhs.rows, rhs.cols)
        external?.prepare(symmetric, symmetric = true, lower = lower)?.use { externalSymm(it, out) }
            ?: builtIn!!.symm(1.0, symmetric, rhs, 0.0, out, lower, right)
        return out
    }

    @Benchmark
    fun preparedSymm(): DenseMatrix {
        val out = DenseMatrix.zero(rhs.rows, rhs.cols)
        externalSymmetric?.let { externalSymm(it, out) } ?: prepared.symm(1.0, rhs, 0.0, out, lower, right)
        return out
    }

    private fun externalSymm(prepared: PreparedSparseComparator, out: DenseMatrix) {
        if (!right) {
            prepared.symm(1.0, rhs, 0.0, out)
            return
        }
        // oneMKL places the sparse operand on the left: B*A = transpose(A*transpose(B)).
        val transposedInput = DenseMatrix.zero(n, rhs.rows)
        for (j in 0 until rhs.cols) for (i in 0 until rhs.rows) transposedInput[j, i] = rhs[i, j]
        val transposedOut = DenseMatrix.zero(n, rhs.rows)
        prepared.symm(1.0, transposedInput, 0.0, transposedOut)
        for (j in 0 until out.cols) for (i in 0 until out.rows) out[i, j] = transposedOut[j, i]
    }

    @Benchmark
    fun sparseProductScaledTransposed(): SparseMatrix {
        if (external == null) return builtIn!!.gemm(-0.75, a, true, a, false)
        val transposed = transposeCscForComparison(a)
        val result = external!!.sparseProduct(transposed, a)
        for (i in result.values.indices) result.values[i] *= -0.75
        return result
    }

    @Benchmark
    fun denseProduct(): DenseMatrix {
        external?.denseProduct(1.0, a, false, b, false, 0.0, c)
            ?: builtIn!!.gemm(1.0, a, false, b, false, 0.0, c)
        return c
    }

    @Benchmark
    fun syrkDense(): DenseMatrix {
        external?.syrkd(1.0, a, false, 0.0, c) ?: builtIn!!.syrk(1.0, a, false, 0.0, c, lower = false)
        return c
    }

    @Benchmark
    fun syrkSparse(): SparseMatrix = external?.syrk(a, false) ?: builtIn!!.syrk(a, lower = false)

    @Benchmark
    fun addScaled(): SparseMatrix = external?.addScaled(-0.75, a, false, a)
        ?: builtIn!!.addScaled(-0.75, a, false, a)
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

/** Benchmark-owned structural transpose; comparator compositions never call koblas arithmetic. */
private fun transposeCscForComparison(a: SparseMatrix): SparseMatrix {
    val rows = IntArray(a.nnz)
    val columns = IntArray(a.nnz)
    val values = DoubleArray(a.nnz)
    var at = 0
    for (j in 0 until a.cols) a.forEachInColumn(j) { i, value ->
        rows[at] = j
        columns[at] = i
        values[at] = value
        at++
    }
    return SparseMatrix.ofTriplets(a.cols, a.rows, rows, columns, values)
}
