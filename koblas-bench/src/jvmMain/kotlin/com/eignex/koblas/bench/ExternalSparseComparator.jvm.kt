package com.eignex.koblas.bench

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

internal actual fun oneMklSparseComparator(): SparseComparator? = JvmOneMklSparse.open()

/** oneMKL inspector-executor sparse BLAS, owned entirely by the non-published benchmark module. */
private class JvmOneMklSparse private constructor(private val library: BenchFfmLibrary) : SparseComparator {
    override val identity: String = "onemkl/sparse-blas"
    override val threading: String = "1 thread"

    private val createCsc = handle(
        "mkl_sparse_d_create_csc",
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
    private val exportCsc = handle(
        "mkl_sparse_d_export_csc",
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

    override fun prepare(a: F64SparseMatrix, triangular: Boolean, lower: Boolean, unitDiag: Boolean): PreparedSparseComparator =
        Prepared(this, a, triangular, lower, unitDiag)

    @OptIn(UnsafeKoblasApi::class)
    override fun dot(x: F64SparseVector, y: DoubleArray): Double =
        if (x.values.isEmpty()) 0.0 else ddoti.invokeExact(x.values.size, seg(x.values), seg(x.indices), seg(y)) as Double

    @OptIn(UnsafeKoblasApi::class)
    override fun axpy(alpha: Double, x: F64SparseVector, y: DoubleArray) {
        if (x.values.isNotEmpty()) daxpyi.invokeExact(x.values.size, alpha, seg(x.values), seg(x.indices), seg(y)) as Unit
    }

    @OptIn(UnsafeKoblasApi::class)
    override fun scatter(x: F64SparseVector, y: DoubleArray) {
        if (x.values.isNotEmpty()) dsctr.invokeExact(x.values.size, seg(x.values), seg(x.indices), seg(y)) as Unit
    }

    @OptIn(UnsafeKoblasApi::class)
    override fun gather(x: F64SparseVector, from: DoubleArray, out: DoubleArray) {
        if (x.values.isNotEmpty()) dgthr.invokeExact(x.values.size, seg(from), seg(out), seg(x.indices)) as Unit
    }

    @OptIn(UnsafeKoblasApi::class)
    override fun gatherZero(x: F64SparseVector, from: DoubleArray, out: DoubleArray) {
        if (x.values.isNotEmpty()) dgthrz.invokeExact(x.values.size, seg(from), seg(out), seg(x.indices)) as Unit
    }

    override fun sparseProduct(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix {
        Prepared(this, a, false, true, false).use { left ->
            Prepared(this, b, false, true, false).use { right ->
                Arena.ofConfined().use { arena ->
                    val outSlot = arena.allocate(ADDRESS)
                    checkStatus(spmm.invokeExact(NON_TRANSPOSE, left.matrix, right.matrix, outSlot) as Int, "mkl_sparse_spmm")
                    val out = outSlot.get(ADDRESS, 0)
                    try {
                        return export(out)
                    } finally {
                        checkStatus(destroy.invokeExact(out) as Int, "mkl_sparse_destroy")
                    }
                }
            }
        }
    }

    private fun export(matrix: MemorySegment): F64SparseMatrix = Arena.ofConfined().use { arena ->
        val indexing = arena.allocate(JAVA_INT)
        val rows = arena.allocate(JAVA_INT)
        val cols = arena.allocate(JAVA_INT)
        val starts = arena.allocate(ADDRESS)
        val ends = arena.allocate(ADDRESS)
        val indices = arena.allocate(ADDRESS)
        val values = arena.allocate(ADDRESS)
        checkStatus(
            exportCsc.invokeExact(matrix, indexing, rows, cols, starts, ends, indices, values) as Int,
            "mkl_sparse_d_export_csc",
        )
        check(indexing.get(JAVA_INT, 0) == INDEX_ZERO) { "oneMKL returned non-zero-based CSC" }
        val m = rows.get(JAVA_INT, 0)
        val n = cols.get(JAVA_INT, 0)
        val startPtr = starts.get(ADDRESS, 0).reinterpret(n.toLong() * Int.SIZE_BYTES)
        val endPtr = ends.get(ADDRESS, 0).reinterpret(n.toLong() * Int.SIZE_BYTES)
        val colPtr = IntArray(n + 1)
        for (j in 0 until n) {
            colPtr[j] = startPtr.getAtIndex(JAVA_INT, j.toLong())
            colPtr[j + 1] = endPtr.getAtIndex(JAVA_INT, j.toLong())
        }
        val nnz = colPtr[n]
        val indexPtr = indices.get(ADDRESS, 0).reinterpret(nnz.toLong() * Int.SIZE_BYTES)
        val valuePtr = values.get(ADDRESS, 0).reinterpret(nnz.toLong() * Double.SIZE_BYTES)
        val rowIdx = IntArray(nnz) { indexPtr.getAtIndex(JAVA_INT, it.toLong()) }
        val outValues = DoubleArray(nnz) { valuePtr.getAtIndex(JAVA_DOUBLE, it.toLong()) }
        F64SparseMatrix.wrap(m, n, colPtr, rowIdx, outValues)
    }

    private fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle = library.handle(name, descriptor)
    private fun seg(values: DoubleArray): MemorySegment = MemorySegment.ofArray(values)
    private fun seg(values: IntArray): MemorySegment = MemorySegment.ofArray(values)

    @OptIn(UnsafeKoblasApi::class)
    private class Prepared(
        private val owner: JvmOneMklSparse,
        a: F64SparseMatrix,
        triangular: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
    ) : PreparedSparseComparator {
        private val arena = Arena.ofShared()
        private val rows = a.rows
        private val cols = a.cols
        private val starts = arena.allocateFrom(JAVA_INT, *a.colPtr.copyOfRange(0, a.cols))
        private val ends = arena.allocateFrom(JAVA_INT, *a.colPtr.copyOfRange(1, a.cols + 1))
        private val rowIdx = arena.allocateFrom(JAVA_INT, *a.rowIdx)
        private val values = arena.allocateFrom(JAVA_DOUBLE, *a.values)
        private val matrixSlot = arena.allocate(ADDRESS)
        val matrix: MemorySegment
        private val descriptor = arena.allocate(DESCRIPTOR)

        init {
            checkStatus(
                owner.createCsc.invokeExact(
                    matrixSlot, INDEX_ZERO, rows, cols, starts, ends, rowIdx, values,
                ) as Int,
                "mkl_sparse_d_create_csc",
            )
            matrix = matrixSlot.get(ADDRESS, 0)
            descriptor.set(JAVA_INT, 0, if (triangular) TYPE_TRIANGULAR else TYPE_GENERAL)
            descriptor.set(JAVA_INT, Int.SIZE_BYTES.toLong(), if (lower) FILL_LOWER else FILL_UPPER)
            descriptor.set(JAVA_INT, (2 * Int.SIZE_BYTES).toLong(), if (unitDiag) DIAG_UNIT else DIAG_NON_UNIT)
            checkStatus(owner.optimize.invokeExact(matrix) as Int, "mkl_sparse_optimize")
        }

        override fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
            checkStatus(
                owner.mv.invokeExact(operation(transpose), alpha, matrix, descriptor, seg(x), beta, seg(y)) as Int,
                "mkl_sparse_d_mv",
            )
        }

        override fun gemm(alpha: Double, b: F64DenseMatrix, beta: Double, c: F64DenseMatrix, transpose: Boolean) {
            checkStatus(
                owner.mm.invokeExact(
                    operation(transpose), alpha, matrix, descriptor, COLUMN_MAJOR, seg(b.data), b.cols, b.rows,
                    beta, seg(c.data), c.rows,
                ) as Int,
                "mkl_sparse_d_mm",
            )
        }

        override fun trsv(x: DoubleArray, out: DoubleArray, transpose: Boolean) {
            checkStatus(
                owner.trsv.invokeExact(operation(transpose), 1.0, matrix, descriptor, seg(x), seg(out)) as Int,
                "mkl_sparse_d_trsv",
            )
        }

        override fun trmv(x: DoubleArray, out: DoubleArray, transpose: Boolean) {
            checkStatus(
                owner.mv.invokeExact(operation(transpose), 1.0, matrix, descriptor, seg(x), 0.0, seg(out)) as Int,
                "mkl_sparse_d_mv triangular",
            )
        }

        override fun trsm(b: F64DenseMatrix, out: F64DenseMatrix, transpose: Boolean) {
            checkStatus(
                owner.trsm.invokeExact(
                    operation(transpose), 1.0, matrix, descriptor, COLUMN_MAJOR,
                    seg(b.data), b.cols, b.rows, seg(out.data), out.rows,
                ) as Int,
                "mkl_sparse_d_trsm",
            )
        }

        override fun trmm(b: F64DenseMatrix, out: F64DenseMatrix, transpose: Boolean) {
            checkStatus(
                owner.mm.invokeExact(
                    operation(transpose), 1.0, matrix, descriptor, COLUMN_MAJOR, seg(b.data), b.cols, b.rows,
                    0.0, seg(out.data), out.rows,
                ) as Int,
                "mkl_sparse_d_mm triangular",
            )
        }

        override fun close() {
            checkStatus(owner.destroy.invokeExact(matrix) as Int, "mkl_sparse_destroy")
            arena.close()
        }

        private fun seg(a: DoubleArray) = MemorySegment.ofArray(a)
    }

    companion object {
        private val required = listOf(
            "MKL_Set_Num_Threads", "mkl_sparse_d_create_csc", "mkl_sparse_destroy", "mkl_sparse_optimize",
            "mkl_sparse_d_mv", "mkl_sparse_d_mm", "mkl_sparse_d_trsv", "mkl_sparse_d_trsm",
            "mkl_sparse_spmm", "mkl_sparse_d_export_csc",
            "cblas_ddoti", "cblas_daxpyi", "cblas_dsctr", "cblas_dgthr", "cblas_dgthrz",
        )

        fun open(): JvmOneMklSparse? {
            val library = BenchFfmLibrary.open(
                listOf("libmkl_rt.so.2", "libmkl_rt.so", "libmkl_rt.dylib", "mkl_rt.2.dll", "mkl_rt.dll"),
                "mkl_sparse_d_create_csc",
            )
            return if (library.present && library.containsAll(required)) JvmOneMklSparse(library) else null
        }
    }
}

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
