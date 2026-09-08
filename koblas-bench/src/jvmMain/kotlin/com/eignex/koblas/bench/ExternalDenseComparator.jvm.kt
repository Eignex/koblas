package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

@OptIn(ExperimentalKoblasApi::class)
internal actual fun explicitBuiltInContext(): KoblasContext {
    val provider = F64BuiltinKernels.simd ?: F64BuiltinKernels.c ?: F64BuiltinKernels.scalar
    return ContextBuilder().withBuiltinKernels(provider).resolve()
}

internal actual fun openBlasComparator(): DenseComparator? = JvmCblasComparator.openOpenBlas()
internal actual fun oneMklDenseComparator(): DenseComparator? = JvmCblasComparator.openOneMkl()

/** OpenBLAS owned by the non-published benchmark module; it uses no koblas host or registry type. */
private class JvmCblasComparator private constructor(
    private val library: BenchFfmLibrary,
    override val identity: String,
    threadSetter: String,
) : DenseComparator {
    override val threading: String = "1 thread"

    init {
        val setThreads = library.handleOrNull(threadSetter, voidOf(JAVA_INT), critical = false)
        checkNotNull(setThreads) { "$identity lacks $threadSetter; single-thread comparison cannot be guaranteed" }
        setThreads.invokeExact(1) as Unit
    }

    private fun seg(a: DoubleArray): MemorySegment = MemorySegment.ofArray(a)
    private fun h(name: String, descriptor: FunctionDescriptor): MethodHandle = library.handle(name, descriptor)

    private val ddot by lazy { h("cblas_ddot", doubleOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)) }
    private val daxpy by lazy { h("cblas_daxpy", voidOf(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)) }
    private val dscal by lazy { h("cblas_dscal", voidOf(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT)) }
    private val dnrm2 by lazy { h("cblas_dnrm2", doubleOf(JAVA_INT, ADDRESS, JAVA_INT)) }
    private val dasum by lazy { h("cblas_dasum", doubleOf(JAVA_INT, ADDRESS, JAVA_INT)) }
    private val dswap by lazy { h("cblas_dswap", voidOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)) }
    private val drotm by lazy { h("cblas_drotm", voidOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS)) }
    private val drot by lazy {
        h("cblas_drot", voidOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE))
    }
    private val dgemv by lazy {
        h(
            "cblas_dgemv",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
                ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }
    private val dsymv by lazy {
        h(
            "cblas_dsymv",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
                ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }
    private val dger by lazy {
        h(
            "cblas_dger",
            voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }
    private val dsyr by lazy {
        h("cblas_dsyr", voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
    }
    private val dsyr2 by lazy {
        h(
            "cblas_dsyr2",
            voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }
    private val dtrsv by lazy {
        h(
            "cblas_dtrsv",
            voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }
    private val dtrmv by lazy {
        h(
            "cblas_dtrmv",
            voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }
    private val dgemm by lazy {
        h(
            "cblas_dgemm",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
                ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }
    private val dsyrk by lazy {
        h(
            "cblas_dsyrk",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
                ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }
    private val dsyr2k by lazy {
        h(
            "cblas_dsyr2k",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
                ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }
    private val dsymm by lazy {
        h(
            "cblas_dsymm",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
                ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }
    private val dtrsm by lazy {
        h(
            "cblas_dtrsm",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
            ),
        )
    }
    private val dtrmm by lazy {
        h(
            "cblas_dtrmm",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
            ),
        )
    }

    override fun dot(x: DoubleArray, y: DoubleArray): Double =
        if (x.isEmpty()) 0.0 else ddot.invokeExact(x.size, seg(x), 1, seg(y), 1) as Double

    override fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray) {
        if (x.isNotEmpty()) daxpy.invokeExact(x.size, alpha, seg(x), 1, seg(y), 1) as Unit
    }

    override fun scale(alpha: Double, x: DoubleArray) {
        if (x.isNotEmpty()) dscal.invokeExact(x.size, alpha, seg(x), 1) as Unit
    }

    override fun nrm2(x: DoubleArray): Double = if (x.isEmpty()) 0.0 else dnrm2.invokeExact(x.size, seg(x), 1) as Double
    override fun asum(x: DoubleArray): Double = if (x.isEmpty()) 0.0 else dasum.invokeExact(x.size, seg(x), 1) as Double

    override fun swap(x: DoubleArray, y: DoubleArray) {
        if (x.isNotEmpty()) dswap.invokeExact(x.size, seg(x), 1, seg(y), 1) as Unit
    }

    override fun rotm(x: DoubleArray, y: DoubleArray, transformation: ModifiedGivens) {
        if (x.isEmpty() || transformation.flag == -2.0) return
        val p = when (transformation.flag) {
            -1.0 -> doubleArrayOf(-1.0, transformation.h11, transformation.h21, transformation.h12, transformation.h22)
            0.0 -> doubleArrayOf(0.0, 0.0, transformation.h21, transformation.h12, 0.0)
            else -> doubleArrayOf(1.0, transformation.h11, 0.0, 0.0, transformation.h22)
        }
        drotm.invokeExact(x.size, seg(x), 1, seg(y), 1, seg(p)) as Unit
    }

    override fun rot(x: DoubleArray, y: DoubleArray, c: Double, s: Double) {
        if (x.isNotEmpty()) drot.invokeExact(x.size, seg(x), 1, seg(y), 1, c, s) as Unit
    }

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        if (a.rows != 0 && a.cols != 0) {
            dgemv.invokeExact(COL_MAJOR, trans(transpose), a.rows, a.cols, alpha, seg(a.data), a.rows, seg(x), 1, beta, seg(y), 1) as Unit
        }
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean,
    ) {
        if (a.rows != 0) dsymv.invokeExact(COL_MAJOR, uplo(lower), a.rows, alpha, seg(a.data), a.rows, seg(x), 1, beta, seg(y), 1) as Unit
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        if (a.rows != 0 && a.cols != 0) dger.invokeExact(COL_MAJOR, a.rows, a.cols, alpha, seg(x), 1, seg(y), 1, seg(a.data), a.rows) as Unit
    }

    override fun syr(alpha: Double, x: DoubleArray, a: DenseMatrix, lower: Boolean) {
        if (a.rows != 0) dsyr.invokeExact(COL_MAJOR, uplo(lower), a.rows, alpha, seg(x), 1, seg(a.data), a.rows) as Unit
    }

    override fun syr2(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix, lower: Boolean) {
        if (a.rows != 0) dsyr2.invokeExact(COL_MAJOR, uplo(lower), a.rows, alpha, seg(x), 1, seg(y), 1, seg(a.data), a.rows) as Unit
    }

    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (a.rows != 0) dtrsv.invokeExact(COL_MAJOR, uplo(lower), trans(transpose), diag(unitDiag), a.rows, seg(a.data), a.rows, seg(x), 1) as Unit
    }

    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (a.rows != 0) dtrmv.invokeExact(COL_MAJOR, uplo(lower), trans(transpose), diag(unitDiag), a.rows, seg(a.data), a.rows, seg(x), 1) as Unit
    }

    override fun gemm(alpha: Double, a: DenseMatrix, transposeA: Boolean, b: DenseMatrix, transposeB: Boolean, beta: Double, c: DenseMatrix) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val n = if (transposeB) b.rows else b.cols
        if (m != 0 && n != 0) dgemm.invokeExact(COL_MAJOR, trans(transposeA), trans(transposeB), m, n, k, alpha, seg(a.data), a.rows, seg(b.data), b.rows, beta, seg(c.data), c.rows) as Unit
    }

    override fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean) {
        val k = if (transpose) a.rows else a.cols
        if (c.rows != 0) dsyrk.invokeExact(COL_MAJOR, uplo(lower), trans(transpose), c.rows, k, alpha, seg(a.data), a.rows, beta, seg(c.data), c.rows) as Unit
    }

    override fun syr2k(alpha: Double, a: DenseMatrix, b: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean) {
        val k = if (transpose) a.rows else a.cols
        if (c.rows != 0) dsyr2k.invokeExact(COL_MAJOR, uplo(lower), trans(transpose), c.rows, k, alpha, seg(a.data), a.rows, seg(b.data), b.rows, beta, seg(c.data), c.rows) as Unit
    }

    override fun symm(alpha: Double, a: DenseMatrix, b: DenseMatrix, beta: Double, c: DenseMatrix, lower: Boolean, right: Boolean) {
        if (c.rows != 0 && c.cols != 0) dsymm.invokeExact(COL_MAJOR, side(right), uplo(lower), c.rows, c.cols, alpha, seg(a.data), a.rows, seg(b.data), b.rows, beta, seg(c.data), c.rows) as Unit
    }

    override fun trsm(a: DenseMatrix, b: DenseMatrix, lower: Boolean, transpose: Boolean, unitDiag: Boolean, right: Boolean, alpha: Double) {
        if (b.rows != 0 && b.cols != 0) dtrsm.invokeExact(COL_MAJOR, side(right), uplo(lower), trans(transpose), diag(unitDiag), b.rows, b.cols, alpha, seg(a.data), a.rows, seg(b.data), b.rows) as Unit
    }

    override fun trmm(a: DenseMatrix, b: DenseMatrix, lower: Boolean, transpose: Boolean, unitDiag: Boolean, right: Boolean, alpha: Double) {
        if (b.rows != 0 && b.cols != 0) dtrmm.invokeExact(COL_MAJOR, side(right), uplo(lower), trans(transpose), diag(unitDiag), b.rows, b.cols, alpha, seg(a.data), a.rows, seg(b.data), b.rows) as Unit
    }

    companion object {
        private val required = listOf(
            "cblas_ddot", "cblas_daxpy", "cblas_dscal", "cblas_dnrm2", "cblas_dasum", "cblas_dswap",
            "cblas_drotm", "cblas_drot", "cblas_dgemv", "cblas_dsymv", "cblas_dger", "cblas_dsyr", "cblas_dsyr2",
            "cblas_dtrsv", "cblas_dtrmv", "cblas_dgemm", "cblas_dsyrk", "cblas_dsyr2k", "cblas_dsymm",
            "cblas_dtrsm", "cblas_dtrmm",
        )

        fun openOpenBlas(): JvmCblasComparator? {
            val library = BenchFfmLibrary.open(listOf("libopenblas.so.0", "libopenblas.so", "libopenblas.dylib"), "cblas_dgemm")
            return if (library.present && library.containsAll(required + "openblas_set_num_threads")) {
                JvmCblasComparator(library, "openblas/cblas", "openblas_set_num_threads")
            } else null
        }

        fun openOneMkl(): JvmCblasComparator? {
            val library = BenchFfmLibrary.open(
                listOf("libmkl_rt.so.2", "libmkl_rt.so", "libmkl_rt.dylib", "mkl_rt.2.dll", "mkl_rt.dll"),
                "cblas_dgemm",
            )
            return if (library.present && library.containsAll(required + "MKL_Set_Num_Threads")) {
                JvmCblasComparator(library, "onemkl/cblas", "MKL_Set_Num_Threads")
            } else null
        }
    }
}

private const val COL_MAJOR = 102
private const val NO_TRANS = 111
private const val TRANS = 112
private const val UPPER = 121
private const val LOWER = 122
private const val NON_UNIT = 131
private const val UNIT = 132
private const val LEFT = 141
private const val RIGHT = 142
private fun trans(value: Boolean) = if (value) TRANS else NO_TRANS
private fun uplo(lower: Boolean) = if (lower) LOWER else UPPER
private fun diag(unit: Boolean) = if (unit) UNIT else NON_UNIT
private fun side(right: Boolean) = if (right) RIGHT else LEFT

internal class BenchFfmLibrary private constructor(
    private val linker: Linker?,
    private val lookup: SymbolLookup?,
) {
    val present: Boolean get() = linker != null && lookup != null
    fun containsAll(names: Iterable<String>): Boolean = names.all { lookup?.find(it)?.isPresent == true }
    fun handleOrNull(name: String, descriptor: FunctionDescriptor, critical: Boolean = true): MethodHandle? {
        val address = lookup?.find(name)?.orElse(null) ?: return null
        val native = linker ?: return null
        return if (critical) native.downcallHandle(address, descriptor, Linker.Option.critical(true)) else native.downcallHandle(address, descriptor)
    }
    fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle =
        checkNotNull(handleOrNull(name, descriptor)) { "OpenBLAS is present but lacks $name" }

    companion object {
        fun open(candidates: List<String>, key: String): BenchFfmLibrary {
            val linker = try { Linker.nativeLinker() } catch (_: UnsupportedOperationException) { return BenchFfmLibrary(null, null) }
            for (candidate in candidates) {
                val lookup = try { SymbolLookup.libraryLookup(candidate, Arena.global()) } catch (_: IllegalArgumentException) { continue } catch (_: UnsatisfiedLinkError) { continue }
                if (lookup.find(key).isPresent) return BenchFfmLibrary(linker, lookup)
            }
            return BenchFfmLibrary(linker, null)
        }
    }
}

internal fun voidOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.ofVoid(*layouts)
internal fun doubleOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.of(JAVA_DOUBLE, *layouts)
