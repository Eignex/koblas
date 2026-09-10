package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.*
import kotlinx.benchmark.*

/**
 * The sparse matrix products against the portable ones.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseProductHostBenchmark {
    @Param("31", "32", "33", "63", "64", "65", "255", "256", "257", "1024")
    var n: Int = 0

    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND)
    var sparseArm: String = BUILTIN_BACKEND

    @Param("0.001", "0.01", "0.1")
    var density: Double = 0.0

    @Param("regular", "banded", "skewed")
    var productShape: String = "regular"

    var triangleVariant: String = "upper-nontrans-nonunit"

    private var builtIn: SparseBlas? = null
    private var external: SparseComparator? = null

    private lateinit var a: SparseMatrix
    private lateinit var square: SparseMatrix
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray
    private lateinit var dense: DenseMatrix
    private lateinit var product: DenseMatrix
    private lateinit var denseSingle: DenseMatrix
    private lateinit var productSingle: DenseMatrix
    private lateinit var triangle: SparseMatrix
    private lateinit var triangularVector: DoubleArray
    private lateinit var triangularDense: DenseMatrix
    private lateinit var triangularDenseRight: DenseMatrix
    private lateinit var scratch: DoubleArray
    private lateinit var triangularProduct: DenseMatrix
    private lateinit var triangularSolve: DenseMatrix
    private lateinit var triangularProductRight: DenseMatrix
    private lateinit var prepared: PreparedSparseMatrix
    private var externalPrepared: PreparedSparseComparator? = null
    private var externalPreparedSquare: PreparedSparseComparator? = null
    private var externalTriangle: PreparedSparseComparator? = null
    private var triangleLower: Boolean = false
    private var triangleTranspose: Boolean = false
    private var triangleUnitDiag: Boolean = false

    @Setup
    fun setup() {
        val rng = benchRng()
        if (sparseArm == BUILTIN_BACKEND) {
            val context = explicitBuiltInContext()
            builtIn = context.sparseBlas
            check(context.sparseBlas.name == BUILTIN_BACKEND) { "built-in sparse arm resolved ${context.sparseBlas.name}" }
            println("resolved: arm=$sparseArm sparse=${context.sparseBlas.name}/${context.sparseKernels.name} threading=single calling thread")
        } else {
            external = checkNotNull(oneMklSparseComparator()) { "the benchmark-only oneMKL sparse comparator is unavailable" }
            check(external!!.identity == "onemkl/sparse-blas" && external!!.threading == "1 thread")
            println("resolved: arm=$sparseArm sparse=${external!!.identity} threading=${external!!.threading}")
        }
        a = sparseComparisonMatrix(n + 1, n - 1, density, productShape, rng)
        square = sparseComparisonMatrix(n - 1, n + 2, density, productShape, rng)
        x = randomVector(n - 1, rng)
        y = randomVector(n + 1, rng)
        dense = randomMatrix(n - 1, RIGHT_HAND_SIDES, rng)
        product = randomMatrix(n + 1, RIGHT_HAND_SIDES, rng)
        denseSingle = randomMatrix(n - 1, 1, rng)
        productSingle = randomMatrix(n + 1, 1, rng)
        val triangleParts = triangleVariant.split('-')
        check(triangleParts.size == 3) { "unknown triangular variant: $triangleVariant" }
        triangleLower = when (triangleParts[0]) {
            "lower" -> true
            "upper" -> false
            else -> error("unknown triangle: ${triangleParts[0]}")
        }
        triangleTranspose = when (triangleParts[1]) {
            "trans" -> true
            "nontrans" -> false
            else -> error("unknown transpose: ${triangleParts[1]}")
        }
        triangleUnitDiag = when (triangleParts[2]) {
            "unit" -> true
            "nonunit" -> false
            else -> error("unknown diagonal: ${triangleParts[2]}")
        }
        triangle = bandTriangle(n, triangleLower)
        triangularVector = randomVector(n, rng)
        triangularDense = randomMatrix(n, RIGHT_HAND_SIDES, rng)
        triangularDenseRight = randomMatrix(RIGHT_HAND_SIDES, n, rng)
        scratch = DoubleArray(n)
        triangularProduct = DenseMatrix.zero(n, RIGHT_HAND_SIDES)
        triangularSolve = DenseMatrix.zero(n, RIGHT_HAND_SIDES)
        triangularProductRight = DenseMatrix.zero(RIGHT_HAND_SIDES, n)
        if (builtIn != null) {
            prepared = builtIn!!.prepare(a)
        } else {
            externalPrepared = external!!.prepare(a)
            externalPreparedSquare = external!!.prepare(square)
        }
        externalTriangle = external?.prepare(
            triangle,
            triangular = true,
            lower = triangleLower,
            unitDiag = triangleUnitDiag,
        )
        println("workload: n=$n density=$density shape=$productShape nnz(A)=${a.nnz}")
        reportAllocatingWorkload(
            "sparse/$sparseArm/prepared-product",
            "reusable operand preparation is outside the measured row; fresh result construction remains timed",
        )
        reportAllocatingWorkload(
            "sparse/$sparseArm/one-shot-product",
            "operand preparation or conversion plus fresh sparse result construction",
        )
    }

    @TearDown
    fun tearDown() {
        if (::prepared.isInitialized) prepared.close()
        externalPrepared?.close()
        externalPreparedSquare?.close()
        externalTriangle?.close()
    }

    @Benchmark
    fun gemm(): DenseMatrix {
        if (external != null) {
            external!!.prepare(a).use { it.gemm(1.0, dense, 0.0, product) }
        } else {
            builtIn!!.gemm(1.0, a, false, dense, false, 0.0, product)
        }
        return product
    }

    /** The same native operation over one dense column, to expose whether its fixed marshalling cost pays. */
    @Benchmark
    fun gemmSingle(): DenseMatrix {
        if (external != null) {
            external!!.prepare(a).use { it.gemm(1.0, denseSingle, 0.0, productSingle) }
        } else {
            builtIn!!.gemm(1.0, a, false, denseSingle, false, 0.0, productSingle)
        }
        return productSingle
    }

    @Benchmark
    fun preparedGemm(): DenseMatrix {
        externalPrepared?.gemm(1.0, dense, 0.0, product) ?: prepared.gemm(1.0, false, dense, 0.0, product)
        return product
    }

    @Benchmark
    fun preparedGemv(): DoubleArray {
        externalPrepared?.gemv(1.0, x, 0.0, y) ?: prepared.gemv(1.0, x, 0.0, y)
        return y
    }

    /** Reuses every operand handle the selected implementation can prepare; result export remains timed. */
    @Benchmark
    fun preparedSparseProduct(): SparseMatrix =
        externalPrepared?.sparseProduct(checkNotNull(externalPreparedSquare)) ?: prepared.gemm(square)

    @Benchmark
    fun gemv(): DoubleArray {
        if (external != null) {
            external!!.prepare(a).use { it.gemv(1.0, x, 0.0, y) }
        } else {
            builtIn!!.gemv(1.0, a, x, 0.0, y)
        }
        return y
    }

    /** Includes operand preparation and representation conversion as a separately labelled one-shot row. */
    @Benchmark
    fun sparseProduct(): SparseMatrix = external?.sparseProduct(a, square) ?: builtIn!!.gemm(a, square)

    @Benchmark
    fun trsv(): DoubleArray {
        triangularVector.copyInto(scratch)
        val comparator = externalTriangle
        if (comparator != null) comparator.trsv(triangularVector, scratch, triangleTranspose)
        else builtIn!!.trsv(triangle, scratch, triangleLower, triangleTranspose, triangleUnitDiag)
        return scratch
    }

    @Benchmark
    fun trmv(): DoubleArray {
        triangularVector.copyInto(scratch)
        val comparator = externalTriangle
        if (comparator != null) comparator.trmv(triangularVector, scratch, triangleTranspose)
        else builtIn!!.trmv(triangle, scratch, triangleLower, triangleTranspose, triangleUnitDiag)
        return scratch
    }

    @Benchmark
    fun trmm(): DenseMatrix {
        triangularDense.data.copyInto(triangularProduct.data)
        val comparator = externalTriangle
        if (comparator != null) comparator.trmm(triangularDense, triangularProduct, triangleTranspose)
        else builtIn!!.trmm(triangle, triangularProduct, triangleLower, triangleTranspose, triangleUnitDiag)
        return triangularProduct
    }

    @Benchmark
    fun trsm(): DenseMatrix {
        triangularDense.data.copyInto(triangularSolve.data)
        val comparator = externalTriangle
        if (comparator != null) {
            comparator.trsm(triangularDense, triangularSolve, triangleTranspose)
        } else {
            builtIn!!.trsm(triangle, triangularSolve, triangleLower, triangleTranspose, triangleUnitDiag)
        }
        return triangularSolve
    }

    @Benchmark
    fun trmmRight(): DenseMatrix {
        triangularDenseRight.data.copyInto(triangularProductRight.data)
        if (external != null) {
            // oneMKL has no right-side sparse triangular multiply. This row is intentionally a composition
            // and is excluded from direct parity: B*A = transpose(A^T*transpose(B)).
            val transposedInput = DenseMatrix.wrap(n, RIGHT_HAND_SIDES, DoubleArray(n * RIGHT_HAND_SIDES))
            for (j in 0 until n) for (i in 0 until RIGHT_HAND_SIDES) transposedInput[j, i] = triangularProductRight[i, j]
            val transposedOut = DenseMatrix.zero(n, RIGHT_HAND_SIDES)
            externalTriangle!!.trmm(transposedInput, transposedOut, transpose = !triangleTranspose)
            for (j in 0 until n) for (i in 0 until RIGHT_HAND_SIDES) triangularProductRight[i, j] = transposedOut[j, i]
        } else {
            builtIn!!.trmm(
                triangle,
                triangularProductRight,
                triangleLower,
                triangleTranspose,
                triangleUnitDiag,
                right = true,
            )
        }
        return triangularProductRight
    }

}

/** Triangular storage variants kept outside the immutable contributor profile v1. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseTriangularVariantBenchmark {
    @Param("64", "257") var n: Int = 64
    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND) var sparseArm: String = BUILTIN_BACKEND
    @Param("0.01") var density: Double = 0.01
    @Param("regular") var productShape: String = "regular"
    @Param("upper-nontrans-nonunit") var triangleVariant: String = "upper-nontrans-nonunit"

    private lateinit var delegate: SparseProductHostBenchmark

    @Setup
    fun setup() {
        delegate = SparseProductHostBenchmark().also {
            it.n = n
            it.sparseArm = sparseArm
            it.density = density
            it.productShape = productShape
            it.triangleVariant = triangleVariant
            it.setup()
        }
    }

    @TearDown
    fun tearDown() {
        delegate.tearDown()
    }

    @Benchmark fun trsv(): DoubleArray = delegate.trsv()

    @Benchmark fun trmv(): DoubleArray = delegate.trmv()

    @Benchmark fun trmm(): DenseMatrix = delegate.trmm()

    @Benchmark fun trsm(): DenseMatrix = delegate.trsm()

    @Benchmark fun trmmRight(): DenseMatrix = delegate.trmmRight()
}

/** Right-side sparse triangular solve, separated from the immutable contributor profile v1. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SparseRightTriangularBenchmark {
    @Param("64", "257") var n: Int = 64
    @Param(BUILTIN_BACKEND, ONEMKL_BACKEND) var sparseArm: String = BUILTIN_BACKEND
    @Param("upper-nontrans-nonunit") var triangleVariant: String = "upper-nontrans-nonunit"

    private lateinit var triangle: SparseMatrix
    private lateinit var input: DenseMatrix
    private lateinit var output: DenseMatrix
    private var builtIn: SparseBlas? = null
    private var externalTriangle: PreparedSparseComparator? = null
    private var lower = false
    private var transpose = false
    private var unitDiag = false

    @Setup
    fun setup() {
        val parts = triangleVariant.split('-')
        check(parts.size == 3) { "unknown triangular variant: $triangleVariant" }
        lower = when (parts[0]) {
            "lower" -> true
            "upper" -> false
            else -> error("unknown triangle: ${parts[0]}")
        }
        transpose = when (parts[1]) {
            "trans" -> true
            "nontrans" -> false
            else -> error("unknown transpose: ${parts[1]}")
        }
        unitDiag = when (parts[2]) {
            "unit" -> true
            "nonunit" -> false
            else -> error("unknown diagonal: ${parts[2]}")
        }
        triangle = bandTriangle(n, lower)
        input = randomMatrix(RIGHT_HAND_SIDES, n, benchRng())
        output = DenseMatrix.zero(RIGHT_HAND_SIDES, n)
        if (sparseArm == BUILTIN_BACKEND) {
            builtIn = explicitBuiltInContext().sparseBlas
            println("resolved: arm=$sparseArm sparse=${builtIn!!.name} threading=single calling thread")
        } else {
            val external = checkNotNull(oneMklSparseComparator()) {
                "the benchmark-only oneMKL sparse comparator is unavailable"
            }
            externalTriangle = external.prepare(triangle, triangular = true, lower = lower, unitDiag = unitDiag)
            println("resolved: arm=$sparseArm sparse=${external.identity} threading=${external.threading}")
        }
        reportAllocatingWorkload(
            "sparse/$sparseArm/trsm-right-composition",
            "oneMKL requires dense transpose staging around its left-side sparse solve",
        )
    }

    @TearDown
    fun tearDown() {
        externalTriangle?.close()
    }

    @Benchmark
    fun trsmRight(): DenseMatrix {
        input.data.copyInto(output.data)
        val external = externalTriangle
        if (external == null) {
            builtIn!!.trsm(triangle, output, lower, transpose, unitDiag, right = true)
            return output
        }
        // B*op(A)^-1 = transpose(op(A)^-T*transpose(B)); oneMKL only exposes a left-side solve.
        val transposedInput = DenseMatrix.wrap(n, RIGHT_HAND_SIDES, DoubleArray(n * RIGHT_HAND_SIDES))
        for (j in 0 until n) for (i in 0 until RIGHT_HAND_SIDES) transposedInput[j, i] = output[i, j]
        val transposedOut = DenseMatrix.zero(n, RIGHT_HAND_SIDES)
        external.trsm(transposedInput, transposedOut, transpose = !transpose)
        for (j in 0 until n) for (i in 0 until RIGHT_HAND_SIDES) output[i, j] = transposedOut[j, i]
        return output
    }
}

/** Right-hand sides for the level-3 row, enough that a multi-column call is not a single-column one. */
private const val RIGHT_HAND_SIDES = 8
