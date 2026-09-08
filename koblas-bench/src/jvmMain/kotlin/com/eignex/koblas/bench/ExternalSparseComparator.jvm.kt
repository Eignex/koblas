package com.eignex.koblas.bench

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

internal actual fun oneMklSparseComparator(): SparseComparator? = JvmOneMklSparse.open()

/** oneMKL inspector-executor sparse BLAS, owned entirely by the non-published benchmark module. */
private class JvmOneMklSparse private constructor(private val library: BenchFfmLibrary) : SparseComparator {
    override val identity: String = "onemkl/sparse-blas"
    override val threading: String = "1 thread"

    private val createCsr = handle(
        "mkl_sparse_d_create_csr",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
    )
    private val destroy = handle("mkl_sparse_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS))
    private val optimize = handle("mkl_sparse_optimize", FunctionDescriptor.of(JAVA_INT, ADDRESS))
    private val mv = handle(
        "mkl_sparse_d_mv",
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, DESCRIPTOR, ADDRESS, JAVA_DOUBLE, ADDRESS),
    )
    private val mm = handle(
        "mkl_sparse_d_mm",
        FunctionDescriptor.of(
            JAVA_INT,
            JAVA_INT, JAVA_DOUBLE, ADDRESS, DESCRIPTOR, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT,
            JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val trsv = handle(
        "mkl_sparse_d_trsv",
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, DESCRIPTOR, ADDRESS, ADDRESS),
    )
    private val trsm = handle(
        "mkl_sparse_d_trsm",
        FunctionDescriptor.of(
            JAVA_INT,
            JAVA_INT, JAVA_DOUBLE, ADDRESS, DESCRIPTOR, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val spmm = handle(
        "mkl_sparse_spmm",
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS),
    )
    private val exportCsr = handle(
        "mkl_sparse_d_export_csr",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
    )
    private val ddoti = handle("cblas_ddoti", FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
    private val daxpyi = handle("cblas_daxpyi", voidOf(JAVA_INT, JAVA_DOUBLE, ADDRESS, ADDRESS, ADDRESS))
    private val dsctr = handle("cblas_dsctr", voidOf(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
    private val dgthr = handle("cblas_dgthr", voidOf(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
    private val dgthrz = handle("cblas_dgthrz", voidOf(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))

    init {
        val setThreads = checkNotNull(
            library.handleOrNull("MKL_Set_Num_Threads", voidOf(JAVA_INT), critical = false),
        ) { "oneMKL lacks MKL_Set_Num_Threads" }
        setThreads.invokeExact(1) as Unit
        library.handleOrNull("MKL_Set_Dynamic", voidOf(JAVA_INT), critical = false)?.let { it.invokeExact(0) as Unit }
    }

    override fun prepare(a: SparseMatrix, triangular: Boolean, lower: Boolean, unitDiag: Boolean): PreparedSparseComparator =
        Prepared(this, a, triangular, lower, unitDiag)

    @OptIn(UnsafeKoblasApi::class)
    override fun dot(x: SparseVector, y: DoubleArray): Double =
        if (x.values.isEmpty()) 0.0 else ddoti.invokeExact(x.values.size, seg(x.values), seg(x.indices), seg(y)) as Double

    @OptIn(UnsafeKoblasApi::class)
    override fun axpy(alpha: Double, x: SparseVector, y: DoubleArray) {
        if (x.values.isNotEmpty()) daxpyi.invokeExact(x.values.size, alpha, seg(x.values), seg(x.indices), seg(y)) as Unit
    }

    @OptIn(UnsafeKoblasApi::class)
    override fun scatter(x: SparseVector, y: DoubleArray) {
        if (x.values.isNotEmpty()) dsctr.invokeExact(x.values.size, seg(x.values), seg(x.indices), seg(y)) as Unit
    }

    @OptIn(UnsafeKoblasApi::class)
    override fun gather(x: SparseVector, from: DoubleArray, out: DoubleArray) {
        if (x.values.isNotEmpty()) dgthr.invokeExact(x.values.size, seg(from), seg(out), seg(x.indices)) as Unit
    }

    @OptIn(UnsafeKoblasApi::class)
    override fun gatherZero(x: SparseVector, from: DoubleArray, out: DoubleArray) {
        if (x.values.isNotEmpty()) dgthrz.invokeExact(x.values.size, seg(from), seg(out), seg(x.indices)) as Unit
    }

    override fun sparseProduct(a: SparseMatrix, b: SparseMatrix): SparseMatrix {
        Prepared(this, a, false, true, false).use { left ->
            Prepared(this, b, false, true, false).use { right ->
                return left.sparseProduct(right)
            }
        }
    }

    private fun multiply(left: MemorySegment, right: MemorySegment): SparseMatrix = Arena.ofConfined().use { arena ->
        val outSlot = arena.allocate(ADDRESS)
        checkStatus(spmm.invokeExact(NON_TRANSPOSE, left, right, outSlot) as Int, "mkl_sparse_spmm")
        val out = outSlot.get(ADDRESS, 0)
        try {
            export(out)
        } finally {
            checkStatus(destroy.invokeExact(out) as Int, "mkl_sparse_destroy")
        }
    }

    private fun export(matrix: MemorySegment): SparseMatrix = Arena.ofConfined().use { arena ->
        val indexing = arena.allocate(JAVA_INT)
        val rows = arena.allocate(JAVA_INT)
        val cols = arena.allocate(JAVA_INT)
        val starts = arena.allocate(ADDRESS)
        val ends = arena.allocate(ADDRESS)
        val indices = arena.allocate(ADDRESS)
        val values = arena.allocate(ADDRESS)
        checkStatus(
            exportCsr.invokeExact(matrix, indexing, rows, cols, starts, ends, indices, values) as Int,
            "mkl_sparse_d_export_csr",
        )
        check(indexing.get(JAVA_INT, 0) == INDEX_ZERO) { "oneMKL returned non-zero-based CSR" }
        val m = rows.get(JAVA_INT, 0)
        val n = cols.get(JAVA_INT, 0)
        val startPtr = starts.get(ADDRESS, 0).reinterpret(m.toLong() * Int.SIZE_BYTES)
        val endPtr = ends.get(ADDRESS, 0).reinterpret(m.toLong() * Int.SIZE_BYTES)
        val rowPtr = IntArray(m + 1)
        for (i in 0 until m) {
            rowPtr[i] = startPtr.getAtIndex(JAVA_INT, i.toLong())
            rowPtr[i + 1] = endPtr.getAtIndex(JAVA_INT, i.toLong())
        }
        val nnz = rowPtr[m]
        val columnPtr = indices.get(ADDRESS, 0).reinterpret(nnz.toLong() * Int.SIZE_BYTES)
        val valuePtr = values.get(ADDRESS, 0).reinterpret(nnz.toLong() * Double.SIZE_BYTES)
        val rowIdx = IntArray(nnz)
        for (i in 0 until m) for (p in rowPtr[i] until rowPtr[i + 1]) rowIdx[p] = i
        val colIdx = IntArray(nnz) { columnPtr.getAtIndex(JAVA_INT, it.toLong()) }
        val outValues = DoubleArray(nnz) { valuePtr.getAtIndex(JAVA_DOUBLE, it.toLong()) }
        SparseMatrix.ofTriplets(m, n, rowIdx, colIdx, outValues)
    }

    private fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle = library.handle(name, descriptor)
    private fun seg(values: DoubleArray): MemorySegment = MemorySegment.ofArray(values)
    private fun seg(values: IntArray): MemorySegment = MemorySegment.ofArray(values)

    @OptIn(UnsafeKoblasApi::class)
    private class Prepared(
        private val owner: JvmOneMklSparse,
        a: SparseMatrix,
        triangular: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
    ) : PreparedSparseComparator {
        private val arena = Arena.ofShared()
        private val rows = a.rows
        private val cols = a.cols
        private val csr = csrOf(a)
        private val starts = arena.allocateFrom(JAVA_INT, *csr.rowPtr.copyOfRange(0, rows))
        private val ends = arena.allocateFrom(JAVA_INT, *csr.rowPtr.copyOfRange(1, rows + 1))
        private val columnIdx = arena.allocateFrom(JAVA_INT, *csr.colIdx)
        private val values = arena.allocateFrom(JAVA_DOUBLE, *csr.values)
        private val matrixSlot = arena.allocate(ADDRESS)
        val matrix: MemorySegment
        private val descriptor = arena.allocate(DESCRIPTOR)

        init {
            checkStatus(
                owner.createCsr.invokeExact(
                    matrixSlot, INDEX_ZERO, rows, cols, starts, ends, columnIdx, values,
                ) as Int,
                "mkl_sparse_d_create_csr",
            )
            matrix = matrixSlot.get(ADDRESS, 0)
            descriptor.set(JAVA_INT, 0, if (triangular) TYPE_TRIANGULAR else TYPE_GENERAL)
            descriptor.set(JAVA_INT, Int.SIZE_BYTES.toLong(), if (lower) FILL_LOWER else FILL_UPPER)
            descriptor.set(JAVA_INT, (2 * Int.SIZE_BYTES).toLong(), if (unitDiag) DIAG_UNIT else DIAG_NON_UNIT)
            checkStatus(owner.optimize.invokeExact(matrix) as Int, "mkl_sparse_optimize")
        }

        override fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
            val inputSize = if (transpose) rows else cols
            val outputSize = if (transpose) cols else rows
            require(x.size == inputSize && y.size == outputSize) {
                "oneMKL sparse gemv dimensions require x=$inputSize and y=$outputSize, got ${x.size} and ${y.size}"
            }
            checkStatus(
                owner.mv.invokeExact(operation(transpose), alpha, matrix, descriptor, seg(x), beta, seg(y)) as Int,
                "mkl_sparse_d_mv",
            )
        }

        override fun gemm(alpha: Double, b: DenseMatrix, beta: Double, c: DenseMatrix, transpose: Boolean) {
            val inner = if (transpose) rows else cols
            val outputRows = if (transpose) cols else rows
            require(b.rows == inner && c.rows == outputRows && c.cols == b.cols) {
                "oneMKL sparse gemm dimensions require B=${inner}xk and C=${outputRows}xk"
            }
            checkStatus(
                owner.mm.invokeExact(
                    operation(transpose), alpha, matrix, descriptor, COLUMN_MAJOR, seg(b.data), b.cols, b.rows,
                    beta, seg(c.data), c.rows,
                ) as Int,
                "mkl_sparse_d_mm",
            )
        }

        override fun trsv(x: DoubleArray, out: DoubleArray, transpose: Boolean) {
            require(rows == cols && x.size == rows && out.size == rows) {
                "oneMKL sparse trsv requires a square matrix and vectors of length $rows"
            }
            checkStatus(
                owner.trsv.invokeExact(operation(transpose), 1.0, matrix, descriptor, seg(x), seg(out)) as Int,
                "mkl_sparse_d_trsv",
            )
        }

        override fun trmv(x: DoubleArray, out: DoubleArray, transpose: Boolean) {
            require(rows == cols && x.size == rows && out.size == rows) {
                "oneMKL sparse trmv requires a square matrix and vectors of length $rows"
            }
            checkStatus(
                owner.mv.invokeExact(operation(transpose), 1.0, matrix, descriptor, seg(x), 0.0, seg(out)) as Int,
                "mkl_sparse_d_mv triangular",
            )
        }

        override fun trsm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean) {
            require(rows == cols && b.rows == rows && out.rows == rows && out.cols == b.cols) {
                "oneMKL sparse trsm requires B and output with $rows rows and equal column counts"
            }
            checkStatus(
                owner.trsm.invokeExact(
                    operation(transpose), 1.0, matrix, descriptor, COLUMN_MAJOR,
                    seg(b.data), b.cols, b.rows, seg(out.data), out.rows,
                ) as Int,
                "mkl_sparse_d_trsm",
            )
        }

        override fun trmm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean) {
            require(rows == cols && b.rows == rows && out.rows == rows && out.cols == b.cols) {
                "oneMKL sparse trmm requires B and output with $rows rows and equal column counts"
            }
            checkStatus(
                owner.mm.invokeExact(
                    operation(transpose), 1.0, matrix, descriptor, COLUMN_MAJOR, seg(b.data), b.cols, b.rows,
                    0.0, seg(out.data), out.rows,
                ) as Int,
                "mkl_sparse_d_mm triangular",
            )
        }

        override fun sparseProduct(right: PreparedSparseComparator): SparseMatrix {
            require(right is Prepared) { "oneMKL sparse product requires two oneMKL prepared operands" }
            require(cols == right.rows) { "oneMKL sparse product inner dimensions differ: $cols and ${right.rows}" }
            return owner.multiply(matrix, right.matrix)
        }

        override fun close() {
            checkStatus(owner.destroy.invokeExact(matrix) as Int, "mkl_sparse_destroy")
            arena.close()
        }

        private fun seg(a: DoubleArray) = MemorySegment.ofArray(a)
    }

    companion object {
        private val required = listOf(
            "MKL_Set_Num_Threads", "mkl_sparse_d_create_csr", "mkl_sparse_destroy", "mkl_sparse_optimize",
            "mkl_sparse_d_mv", "mkl_sparse_d_mm", "mkl_sparse_d_trsv", "mkl_sparse_d_trsm",
            "mkl_sparse_spmm", "mkl_sparse_d_export_csr",
            "cblas_ddoti", "cblas_daxpyi", "cblas_dsctr", "cblas_dgthr", "cblas_dgthrz",
        )

        fun open(): JvmOneMklSparse? {
            val library = BenchFfmLibrary.open(
                listOf("libmkl_rt.so.2", "libmkl_rt.so", "libmkl_rt.dylib", "mkl_rt.2.dll", "mkl_rt.dll"),
                "mkl_sparse_d_create_csr",
            )
            return if (library.present && library.containsAll(required)) JvmOneMklSparse(library) else null
        }
    }
}

@OptIn(UnsafeKoblasApi::class)
private fun csrOf(a: SparseMatrix): CsrArrays {
    val rowPtr = IntArray(a.rows + 1)
    for (row in a.rowIdx) rowPtr[row + 1]++
    for (i in 0 until a.rows) rowPtr[i + 1] += rowPtr[i]
    val cursor = rowPtr.copyOf()
    val colIdx = IntArray(a.nnz)
    val values = DoubleArray(a.nnz)
    for (j in 0 until a.cols) {
        for (p in a.colPtr[j] until a.colPtr[j + 1]) {
            val target = cursor[a.rowIdx[p]]++
            colIdx[target] = j
            values[target] = a.values[p]
        }
    }
    return CsrArrays(rowPtr, colIdx, values)
}

private class CsrArrays(val rowPtr: IntArray, val colIdx: IntArray, val values: DoubleArray)

private val DESCRIPTOR: StructLayout = MemoryLayout.structLayout(
    JAVA_INT.withName("type"), JAVA_INT.withName("mode"), JAVA_INT.withName("diag"),
)
private const val INDEX_ZERO = 0
private const val NON_TRANSPOSE = 10
private const val TRANSPOSE_SPARSE = 11
private const val TYPE_GENERAL = 20
private const val TYPE_TRIANGULAR = 23
private const val FILL_LOWER = 40
private const val FILL_UPPER = 41
private const val DIAG_NON_UNIT = 50
private const val DIAG_UNIT = 51
private const val COLUMN_MAJOR = 102
private fun operation(transpose: Boolean) = if (transpose) TRANSPOSE_SPARSE else NON_TRANSPOSE
private fun checkStatus(status: Int, operation: String) = check(status == 0) { "$operation failed with oneMKL status $status" }
