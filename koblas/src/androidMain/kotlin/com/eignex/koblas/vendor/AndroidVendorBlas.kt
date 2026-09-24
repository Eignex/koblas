@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import com.eignex.koblas.*
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.requireStructured
import com.eignex.koblas.dense.requireTriangular
import kotlin.math.abs

/**
 * The OpenBLAS this library bundles for Android, reached through [AndroidCblas].
 *
 * Android ships no BLAS and ART cannot call a C function directly, so the AAR carries its own: a static,
 * single-threaded OpenBLAS linked into one JNI library per ABI, with every OpenBLAS symbol hidden. An
 * application that loads a BLAS of its own gets that one and this one side by side, never one in place of
 * the other.
 *
 * Operands are pinned rather than copied. ART hands a pinned primitive array over in place, so a call reaches
 * the caller's own storage and the route of a directly addressed call names no adapter, as on Kotlin/Native.
 */
internal class AndroidVendorBlas private constructor() : Blas {
    override val vendor: Vendor = Vendor.OpenBlas

    override val libraryPath: String by lazy { AndroidCblas.libraryPath() ?: "lib${AndroidCblas.LIBRARY}.so" }

    override val version: String by lazy { AndroidCblas.config().trim().ifEmpty { "unreported" } }

    /** An instance exists only once the library was held to one thread and read back, so this is settled. */
    override val threadEvidence: ThreadEvidence = ThreadEvidence.Confirmed

    /** The bundled build exports every entry point, [BlasOperation.Gemmt] included. */
    override val directlyImplemented: Set<BlasOperation> = BlasOperation.entries.toCollection(LinkedHashSet())

    override fun routeOf(
        operation: BlasOperation,
        matrices: List<DenseMatrix>,
        vectors: List<DenseVector>,
    ): CallRoute = routeFor(
        operation = operation,
        vendor = vendor,
        exported = true,
        matrices = matrices,
        vectors = vectors,
        transfer = null,
    )

    // Level 1. A vector window is always expressible, so these never stage and never compose.

    override fun dot(x: DenseVector, y: DenseVector): Double {
        requireSameSize(x.size, y.size, "dot")
        if (noWork(x)) return 0.0
        return AndroidCblas.ddot(x.size, x.values, baseIndex(x), x.stride, y.values, baseIndex(y), y.stride)
    }

    override fun nrm2(x: DenseVector): Double {
        if (noWork(x)) return 0.0
        return AndroidCblas.dnrm2(x.size, x.values, baseIndex(x), abs(x.stride))
    }

    override fun asum(x: DenseVector): Double {
        if (noWork(x)) return 0.0
        return AndroidCblas.dasum(x.size, x.values, baseIndex(x), abs(x.stride))
    }

    /** The index the library returns, restored to the caller's order when the stride runs backwards. */
    override fun iamax(x: DenseVector): Int {
        if (noWork(x)) return 0
        val found = AndroidCblas.idamax(x.size, x.values, baseIndex(x), abs(x.stride))
        return if (x.stride >= 0) found else x.size - 1 - found
    }

    override fun axpy(alpha: Double, x: DenseVector, y: DenseVector) {
        requireSameSize(x.size, y.size, "axpy")
        if (noWork(x)) return
        AndroidCblas.daxpy(x.size, alpha, x.values, baseIndex(x), x.stride, y.values, baseIndex(y), y.stride)
    }

    override fun scal(alpha: Double, x: DenseVector) {
        if (noWork(x)) return
        AndroidCblas.dscal(x.size, alpha, x.values, baseIndex(x), abs(x.stride))
    }

    override fun copy(x: DenseVector, y: DenseVector) {
        requireSameSize(x.size, y.size, "copy")
        if (noWork(x)) return
        AndroidCblas.dcopy(x.size, x.values, baseIndex(x), x.stride, y.values, baseIndex(y), y.stride)
    }

    override fun swap(x: DenseVector, y: DenseVector) {
        requireSameSize(x.size, y.size, "swap")
        if (noWork(x)) return
        AndroidCblas.dswap(x.size, x.values, baseIndex(x), x.stride, y.values, baseIndex(y), y.stride)
    }

    override fun rot(x: DenseVector, y: DenseVector, c: Double, s: Double) {
        requireSameSize(x.size, y.size, "rot")
        if (noWork(x)) return
        AndroidCblas.drot(x.size, x.values, baseIndex(x), x.stride, y.values, baseIndex(y), y.stride, c, s)
    }

    // Level 2. One matrix operand, passed in place with its own row count as the leading dimension.

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        requireGemvOperands(a, transposeA, x.size, y.size)
        if (noWork(a)) return
        AndroidCblas.dgemv(
            transposeFor(transposeA), a.rows, a.cols, alpha, a.values, lda(a),
            x.values, baseIndex(x), x.stride, beta, y.values, baseIndex(y), y.stride,
        )
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        requireStructured(structure, "symv")
        requireSymvOperands(a, x.size, y.size)
        if (noWork(a)) return
        AndroidCblas.dsymv(
            uploFor(structure), a.rows, alpha, a.values, lda(a),
            x.values, baseIndex(x), x.stride, beta, y.values, baseIndex(y), y.stride,
        )
    }

    override fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix) {
        requireGerOperands(x.size, y.size, a)
        if (noWork(a)) return
        AndroidCblas.dger(
            a.rows, a.cols, alpha, x.values, baseIndex(x), x.stride,
            y.values, baseIndex(y), y.stride, a.values, lda(a),
        )
    }

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, structure: MatrixStructure) {
        requireStructured(structure, "syr")
        requireSyrOperands(a, x.size, "syr")
        if (noWork(a)) return
        AndroidCblas.dsyr(uploFor(structure), a.rows, alpha, x.values, baseIndex(x), x.stride, a.values, lda(a))
    }

    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, structure: MatrixStructure) {
        requireStructured(structure, "syr2")
        requireSyr2Operands(a, x.size, y.size, "syr2")
        if (noWork(a)) return
        AndroidCblas.dsyr2(
            uploFor(structure), a.rows, alpha, x.values, baseIndex(x), x.stride,
            y.values, baseIndex(y), y.stride, a.values, lda(a),
        )
    }

    override fun trsv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) {
        requireTriangular(structure, "trsv")
        requireTriangularVectorOperands(a, x.size, "trsv")
        if (noWork(a)) return
        AndroidCblas.dtrsv(
            uploFor(structure), transposeFor(transposeA), diagFor(structure), a.rows, a.values, lda(a),
            x.values, baseIndex(x), x.stride,
        )
    }

    override fun trmv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) {
        requireTriangular(structure, "trmv")
        requireTriangularVectorOperands(a, x.size, "trmv")
        if (noWork(a)) return
        AndroidCblas.dtrmv(
            uploFor(structure), transposeFor(transposeA), diagFor(structure), a.rows, a.values, lda(a),
            x.values, baseIndex(x), x.stride,
        )
    }

    // Level 3. Every operand is contiguous column-major, so the layout is fixed and a transpose is a flag.

    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        requireGemmOperands(a, transposeA, b, transposeB, c)
        if (noWork(c)) return
        AndroidCblas.dgemm(
            transposeFor(transposeA), transposeFor(transposeB), c.rows, c.cols, depth, alpha,
            a.values, lda(a), b.values, lda(b), beta, c.values, lda(c),
        )
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        rightSide: Boolean,
    ) {
        requireStructured(structure, "symm")
        requireSymmOperands(a, b, c, rightSide)
        if (noWork(c)) return
        AndroidCblas.dsymm(
            sideFor(rightSide), uploFor(structure), c.rows, c.cols, alpha,
            a.values, lda(a), b.values, lda(b), beta, c.values, lda(c),
        )
    }

    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) {
        requireStructured(structure, "syrk")
        requireSyrkOperands(a, transposeA, c)
        val depth = if (transposeA) a.rows else a.cols
        if (noWork(c)) return
        AndroidCblas.dsyrk(
            uploFor(structure), transposeFor(transposeA), c.rows, depth, alpha,
            a.values, lda(a), beta, c.values, lda(c),
        )
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) {
        requireStructured(structure, "syr2k")
        requireSyr2kOperands(a, b, transposeA, c)
        val depth = if (transposeA) a.rows else a.cols
        if (noWork(c)) return
        AndroidCblas.dsyr2k(
            uploFor(structure), transposeFor(transposeA), c.rows, depth, alpha,
            a.values, lda(a), b.values, lda(b), beta, c.values, lda(c),
        )
    }

    override fun trmm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
    ) {
        requireTriangular(structure, "trmm")
        requireTriangularMatrixOperands(a, b, rightSide, "trmm")
        if (noWork(b)) return
        AndroidCblas.dtrmm(
            sideFor(rightSide), uploFor(structure), transposeFor(transposeA), diagFor(structure),
            b.rows, b.cols, alpha, a.values, lda(a), b.values, lda(b),
        )
    }

    override fun trsm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
    ) {
        requireTriangular(structure, "trsm")
        requireTriangularMatrixOperands(a, b, rightSide, "trsm")
        if (noWork(b)) return
        AndroidCblas.dtrsm(
            sideFor(rightSide), uploFor(structure), transposeFor(transposeA), diagFor(structure),
            b.rows, b.cols, alpha, a.values, lda(a), b.values, lda(b),
        )
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) {
        requireStructured(structure, "gemmt")
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        if (noWork(c)) return
        AndroidCblas.dgemmt(
            uploFor(structure), transposeFor(transposeA), transposeFor(transposeB), c.rows, depth, alpha,
            a.values, lda(a), b.values, lda(b), beta, c.values, lda(c),
        )
    }

    internal companion object {
        /**
         * Whether the bundled library loaded and passed the checks every binding runs, settled once.
         *
         * A host test runs on a desktop JVM, where the AAR's library is not on the path, and reads false here
         * like any other host without a vendor.
         */
        private val usable: Boolean by lazy {
            // Ordered as the other bindings are: the thread count is fixed before any arithmetic reaches it.
            AndroidCblas.loaded && confirmSingleThread() && verifiedAbi()
        }

        /**
         * A new binding over the bundled library, or null where it is not usable.
         *
         * The library is loaded and checked once per process; each binding is its own instance over it, as an
         * opened vendor is on the other platforms.
         */
        fun open(): AndroidVendorBlas? = if (usable) AndroidVendorBlas() else null

        /**
         * Holds the library to one compute thread and says whether it agreed.
         *
         * The bundled build has no worker threads to begin with, so this is confirmation rather than
         * configuration; a library that reported more would be a build this binding does not understand.
         */
        private fun confirmSingleThread(): Boolean {
            AndroidCblas.holdToOneThread()
            return AndroidCblas.threads() == 1
        }

        /** The same known-answer check every binding runs before a caller reaches the library. */
        private fun verifiedAbi(): Boolean {
            if (declaresWideIntegers(AndroidCblas.config())) return false
            val x = AbiProbe.x
            if (AndroidCblas.ddot(x.size, x, 0, 1, AbiProbe.y, 0, 1) != AbiProbe.DOT) return false
            val b = AbiProbe.operand
            val c = DoubleArray(b.size)
            AndroidCblas.dgemm(Cblas.NO_TRANS, Cblas.NO_TRANS, 2, 2, 2, 1.0, AbiProbe.identity, 2, b, 2, 0.0, c, 2)
            return c.indices.all { c[it] == b[it] }
        }
    }
}

/** The leading dimension of contiguous column-major storage, which BLAS requires to be at least one. */
private fun lda(a: DenseMatrix): Int = maxOf(1, a.rows)

/** The lowest storage index [vector] touches, which is where BLAS starts for either sign of stride. */
private fun baseIndex(vector: DenseVector): Int =
    if (vector.stride >= 0) vector.offset else vector.offset + (vector.size - 1) * vector.stride
