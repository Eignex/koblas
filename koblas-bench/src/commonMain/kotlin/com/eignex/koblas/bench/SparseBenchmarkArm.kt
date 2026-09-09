package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.sparse.PreparedSparseMatrix
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels

/** One resolved sparse benchmark implementation, without nullable arm selection at each call site. */
internal sealed interface SparseBenchmarkArm {
    val name: String
    val identity: String
    val threading: String
    val builtIn: SparseBlas?
    val sparseKernels: SparseKernels?
    val external: SparseComparator?

    fun reportResolution() {
        println()
        println("resolved: arm=$name sparse=$identity threading=$threading")
    }

    fun prepare(a: SparseMatrix, descriptor: SparseDescriptor = SparseDescriptor()): PreparedSparseBenchmarkArm

    /** Prepares a sparse-product right operand only when the selected implementation consumes such a handle. */
    fun prepareProductRight(a: SparseMatrix): PreparedSparseBenchmarkArm = prepare(a)

    fun dot(x: SparseVector, y: DoubleArray): Double =
        external?.dot(x, y) ?: sparseKernels!!.dot(x, y)

    fun axpy(alpha: Double, x: SparseVector, y: DoubleArray) {
        external?.axpy(alpha, x, y) ?: sparseKernels!!.axpy(y, alpha, x)
    }

    fun scatter(x: SparseVector, y: DoubleArray) {
        external?.scatter(x, y) ?: sparseKernels!!.scatter(x, y)
    }

    fun gather(x: SparseVector, from: DoubleArray, out: DoubleArray) {
        external?.gather(x, from, out) ?: sparseKernels!!.gather(x, from)
    }

    fun gatherZero(x: SparseVector, from: DoubleArray, out: DoubleArray) {
        external?.gatherZero(x, from, out) ?: sparseKernels!!.gatherZero(x, from)
    }

    fun gemv(alpha: Double, a: SparseMatrix, x: DoubleArray, beta: Double, y: DoubleArray) {
        val comparator = external
        if (comparator == null) builtIn!!.gemv(alpha, a, x, beta, y)
        else comparator.prepare(a).use { it.gemv(alpha, x, beta, y) }
    }

    fun gemm(alpha: Double, a: SparseMatrix, b: DenseMatrix, beta: Double, c: DenseMatrix) {
        val comparator = external
        if (comparator == null) builtIn!!.gemm(alpha, a, false, b, false, beta, c)
        else comparator.prepare(a).use { it.gemm(alpha, b, beta, c) }
    }

    fun symv(alpha: Double, a: SparseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        val comparator = external
        if (comparator == null) builtIn!!.symv(alpha, a, x, beta, y, lower)
        else comparator.prepare(a, symmetric = true, lower = lower).use { it.symv(alpha, x, beta, y) }
    }

    fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        val comparator = external
        if (comparator == null) builtIn!!.symm(alpha, a, b, beta, c, lower, right)
        else comparator.prepare(a, symmetric = true, lower = lower).use {
            externalSymm(it, alpha, b, beta, c, right)
        }
    }

    fun sparseProduct(a: SparseMatrix, b: SparseMatrix): SparseMatrix =
        external?.sparseProduct(a, b) ?: builtIn!!.gemm(a, b)

    fun scaledTransposedProduct(alpha: Double, a: SparseMatrix): SparseMatrix {
        val comparator = external ?: return builtIn!!.gemm(alpha, a, true, a, false)
        val result = comparator.sparseProduct(transposeCscForComparison(a), a)
        for (i in result.values.indices) result.values[i] *= alpha
        return result
    }

    fun denseProduct(alpha: Double, a: SparseMatrix, b: SparseMatrix, beta: Double, c: DenseMatrix) {
        external?.denseProduct(alpha, a, false, b, false, beta, c)
            ?: builtIn!!.gemm(alpha, a, false, b, false, beta, c)
    }

    fun syrkDense(alpha: Double, a: SparseMatrix, beta: Double, c: DenseMatrix) {
        external?.syrkd(alpha, a, false, beta, c)
            ?: builtIn!!.syrk(alpha, a, false, beta, c, lower = false)
    }

    fun syrkSparse(a: SparseMatrix): SparseMatrix =
        external?.syrk(a, false) ?: builtIn!!.syrk(a, lower = false)

    fun addScaled(alpha: Double, a: SparseMatrix): SparseMatrix =
        external?.addScaled(alpha, a, false, a) ?: builtIn!!.addScaled(alpha, a, false, a)

    companion object {
        fun resolve(name: String): SparseBenchmarkArm = when (name) {
            BUILTIN_BACKEND -> BuiltInSparseBenchmarkArm(explicitBuiltInContext())
            ONEMKL_BACKEND -> ExternalSparseBenchmarkArm(
                name,
                checkNotNull(externalSparseArm(name)),
            )
            else -> error("unknown sparse benchmark arm: $name")
        }.also { arm ->
            check(arm.identity.startsWith(name)) {
                "sparse benchmark arm $name resolved ${arm.identity}"
            }
            arm.reportResolution()
        }
    }
}

/** Resolves the optional comparator once for sparse fixtures whose built-in contract is not matrix BLAS. */
internal fun externalSparseArm(name: String): SparseComparator? = when (name) {
    BUILTIN_BACKEND -> null
    ONEMKL_BACKEND -> checkNotNull(oneMklSparseComparator()) {
        "the benchmark-only oneMKL sparse comparator is unavailable"
    }.also { comparator ->
        check(comparator.identity == "$name/sparse-blas" && comparator.threading == "1 thread") {
            "$name sparse arm resolved ${comparator.identity} with ${comparator.threading}"
        }
    }
    else -> error("unknown sparse benchmark arm: $name")
}

internal data class SparseDescriptor(
    val triangular: Boolean = false,
    val symmetric: Boolean = false,
    val lower: Boolean = true,
    val unitDiag: Boolean = false,
)

internal interface PreparedSparseBenchmarkArm : AutoCloseable {
    fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false)
    fun gemm(alpha: Double, b: DenseMatrix, beta: Double, c: DenseMatrix, transpose: Boolean = false)
    fun symv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray)
    fun symm(alpha: Double, b: DenseMatrix, beta: Double, c: DenseMatrix, right: Boolean = false)
    fun trsv(x: DoubleArray, out: DoubleArray, transpose: Boolean = false)
    fun trmv(x: DoubleArray, out: DoubleArray, transpose: Boolean = false)
    fun trsm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean = false, right: Boolean = false)
    fun trmm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean = false, right: Boolean = false)
    fun sparseProduct(right: PreparedSparseBenchmarkArm): SparseMatrix
}

private class BuiltInSparseBenchmarkArm(private val context: KoblasContext) : SparseBenchmarkArm {
    override val name: String = BUILTIN_BACKEND
    override val identity: String = "$name/${context.sparseBlas.name}/${context.sparseKernels.name}"
    override val threading: String = "single calling thread"
    override val builtIn: SparseBlas = context.sparseBlas
    override val sparseKernels: SparseKernels = context.sparseKernels
    override val external: SparseComparator? = null

    init {
        check(context.sparseBlas.name == BUILTIN_BACKEND) {
            "built-in sparse arm resolved ${context.sparseBlas.name}"
        }
    }

    override fun prepare(a: SparseMatrix, descriptor: SparseDescriptor): PreparedSparseBenchmarkArm =
        BuiltInPreparedSparseBenchmarkArm(
            if (descriptor.triangular) null else context.sparseBlas.prepare(a),
            context,
            a,
            descriptor,
        )

    override fun prepareProductRight(a: SparseMatrix): PreparedSparseBenchmarkArm =
        BuiltInPreparedSparseBenchmarkArm(null, context, a, SparseDescriptor())
}

private class ExternalSparseBenchmarkArm(
    override val name: String,
    private val comparator: SparseComparator,
) : SparseBenchmarkArm {
    override val identity: String = comparator.identity
    override val threading: String = comparator.threading
    override val builtIn: SparseBlas? = null
    override val sparseKernels: SparseKernels? = null
    override val external: SparseComparator = comparator

    override fun prepare(a: SparseMatrix, descriptor: SparseDescriptor): PreparedSparseBenchmarkArm =
        ExternalPreparedSparseBenchmarkArm(
            comparator.prepare(
                a,
                triangular = descriptor.triangular,
                symmetric = descriptor.symmetric,
                lower = descriptor.lower,
                unitDiag = descriptor.unitDiag,
            ),
        )
}

private class BuiltInPreparedSparseBenchmarkArm(
    private val prepared: PreparedSparseMatrix?,
    private val context: KoblasContext,
    private val matrix: SparseMatrix,
    private val descriptor: SparseDescriptor,
) : PreparedSparseBenchmarkArm {
    override fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) =
        checkNotNull(prepared).gemv(alpha, x, beta, y, transpose)

    override fun gemm(alpha: Double, b: DenseMatrix, beta: Double, c: DenseMatrix, transpose: Boolean) =
        checkNotNull(prepared).gemm(alpha, transpose, b, beta, c)

    override fun symv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray) =
        checkNotNull(prepared).symv(alpha, x, beta, y, descriptor.lower)

    override fun symm(alpha: Double, b: DenseMatrix, beta: Double, c: DenseMatrix, right: Boolean) =
        checkNotNull(prepared).symm(alpha, b, beta, c, descriptor.lower, right)

    override fun trsv(x: DoubleArray, out: DoubleArray, transpose: Boolean) =
        context.sparseBlas.trsv(matrix, out, descriptor.lower, transpose, descriptor.unitDiag)

    override fun trmv(x: DoubleArray, out: DoubleArray, transpose: Boolean) =
        context.sparseBlas.trmv(matrix, out, descriptor.lower, transpose, descriptor.unitDiag)

    override fun trsm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean, right: Boolean) =
        context.sparseBlas.trsm(matrix, out, descriptor.lower, transpose, descriptor.unitDiag, right)

    override fun trmm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean, right: Boolean) =
        context.sparseBlas.trmm(matrix, out, descriptor.lower, transpose, descriptor.unitDiag, right)

    override fun sparseProduct(right: PreparedSparseBenchmarkArm): SparseMatrix {
        check(right is BuiltInPreparedSparseBenchmarkArm) { "sparse product arms must match" }
        return checkNotNull(prepared).gemm(right.matrix)
    }

    override fun close() {
        prepared?.close()
    }
}

private class ExternalPreparedSparseBenchmarkArm(
    private val prepared: PreparedSparseComparator,
) : PreparedSparseBenchmarkArm {
    override fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) =
        prepared.gemv(alpha, x, beta, y, transpose)

    override fun gemm(alpha: Double, b: DenseMatrix, beta: Double, c: DenseMatrix, transpose: Boolean) =
        prepared.gemm(alpha, b, beta, c, transpose)

    override fun symv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray) =
        prepared.symv(alpha, x, beta, y)

    override fun symm(alpha: Double, b: DenseMatrix, beta: Double, c: DenseMatrix, right: Boolean) {
        externalSymm(prepared, alpha, b, beta, c, right)
    }

    override fun trsv(x: DoubleArray, out: DoubleArray, transpose: Boolean) = prepared.trsv(x, out, transpose)
    override fun trmv(x: DoubleArray, out: DoubleArray, transpose: Boolean) = prepared.trmv(x, out, transpose)
    override fun trsm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean, right: Boolean) {
        if (!right) {
            prepared.trsm(b, out, transpose)
            return
        }
        transposeTriangularOperation(b, out) { input, output -> prepared.trsm(input, output, !transpose) }
    }

    override fun trmm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean, right: Boolean) {
        if (!right) {
            prepared.trmm(b, out, transpose)
            return
        }
        transposeTriangularOperation(b, out) { input, output -> prepared.trmm(input, output, !transpose) }
    }

    override fun sparseProduct(right: PreparedSparseBenchmarkArm): SparseMatrix {
        check(right is ExternalPreparedSparseBenchmarkArm) { "sparse product arms must match" }
        return prepared.sparseProduct(right.prepared)
    }

    override fun close() = prepared.close()

    private inline fun transposeTriangularOperation(
        input: DenseMatrix,
        output: DenseMatrix,
        operation: (DenseMatrix, DenseMatrix) -> Unit,
    ) {
        val transposedInput = DenseMatrix.zero(input.cols, input.rows)
        for (j in 0 until input.cols) for (i in 0 until input.rows) transposedInput[j, i] = input[i, j]
        val transposedOut = DenseMatrix.zero(output.cols, output.rows)
        operation(transposedInput, transposedOut)
        for (j in 0 until output.cols) for (i in 0 until output.rows) output[i, j] = transposedOut[j, i]
    }
}

internal fun externalSymm(
    prepared: PreparedSparseComparator,
    alpha: Double,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    right: Boolean,
) {
    if (!right) {
        prepared.symm(alpha, b, beta, c)
        return
    }
    // oneMKL places the sparse operand on the left: B*A = transpose(A*transpose(B)).
    val transposedInput = DenseMatrix.zero(b.cols, b.rows)
    for (j in 0 until b.cols) for (i in 0 until b.rows) transposedInput[j, i] = b[i, j]
    val transposedOut = DenseMatrix.zero(c.cols, c.rows)
    for (j in 0 until c.cols) for (i in 0 until c.rows) transposedOut[j, i] = c[i, j]
    prepared.symm(alpha, transposedInput, beta, transposedOut)
    for (j in 0 until c.cols) for (i in 0 until c.rows) c[i, j] = transposedOut[j, i]
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

/** Owns setup-only benchmark resources and closes partial setup in reverse acquisition order. */
internal class BenchmarkResources : AutoCloseable {
    private val resources = mutableListOf<AutoCloseable>()
    private var closed = false

    fun <T : AutoCloseable> own(resource: T): T {
        check(!closed) { "benchmark resources are closed" }
        resources += resource
        return resource
    }

    inline fun <T : AutoCloseable> acquire(factory: () -> T): T = try {
        own(factory())
    } catch (failure: Throwable) {
        close()
        throw failure
    }

    override fun close() {
        if (closed) return
        closed = true
        var failure: Throwable? = null
        for (resource in resources.asReversed()) {
            try {
                resource.close()
            } catch (closeFailure: Throwable) {
                if (failure == null) failure = closeFailure else failure.addSuppressed(closeFailure)
            }
        }
        resources.clear()
        if (failure != null) throw failure
    }
}
