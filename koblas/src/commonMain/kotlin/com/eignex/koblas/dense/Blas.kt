@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.VectorLike
import com.eignex.koblas.VectorView
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.requireShape
import com.eignex.koblas.transpose

/** Dense matrix routines as a backend half. */
public interface Blas : Backend {

    /** The vector kernels this half's inherited routines run on; the installed ones by default. */
    public val vectorKernels: VectorKernels get() = koblas.vectorKernels

    /** `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose].
     *  `beta == 0.0` overwrites [y] without reading it. */
    public fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
    )

    /** [gemv] with `alpha = 1, beta = 0`, into a fresh result. */
    public fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /** `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *  `beta == 0.0` overwrites [c] without reading it. */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    public fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    )

    /** [gemm] with `alpha = 1, beta = 0`, into a fresh matrix. `A.cols` must equal `B.rows`. */
    public fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }

    /** `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`).
     *  [Uplo.FULL] writes both triangles, unlike standard `dsyrk`; `beta == 0.0` overwrites without reading. */
    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    public fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo = Uplo.FULL,
        workspace: Workspace? = null,
    )

    /** `y = alpha · A · x + beta · y` for a symmetric [a] (BLAS `dsymv`). Only the [lower] triangle is read,
     *  diagonal included; `beta == 0.0` overwrites [y] without reading it. */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    public fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean = true)

    /** `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
     *  [lower] triangle of [a] is read; `beta == 0.0` overwrites [c] without reading it. */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    public fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
    )

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`), the dense form a backend can dispatch. The free `ger` accepts
     *  [VectorView] operands and takes a sparse fast path. */
    public fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        requireShape(a.rows == x.size && a.cols == y.size) {
            "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
        }
        if (alpha == 0.0) return
        val kernels = vectorKernels
        for (j in 0 until a.cols) {
            val scaled = alpha * y[j]
            if (scaled != 0.0) kernels.axpy(a.data, a.colOffset(j), scaled, x, 0, a.rows)
        }
    }

    /** `A += alpha · x · xᵀ` (BLAS `dsyr`), writing the triangles [uplo] selects. [syrk] is the rank-k form. */
    public fun syr(alpha: Double, x: VectorLike, a: DenseMatrix, uplo: Uplo = Uplo.FULL) {
        requireShape(a.rows == a.cols) { "syr: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "syr: x length ${x.size} != ${a.rows}" }
        if (alpha == 0.0) return
        val n = a.rows
        val ad = a.data
        val xs = x.toDoubleArray()
        for (j in 0 until n) {
            val xj = alpha * xs[j]
            if (xj == 0.0) continue
            for (i in j until n) addUplo(ad, n, i, j, xj * xs[i], uplo)
        }
    }

    /** `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing the triangles [uplo] selects. */
    public fun syr2(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix, uplo: Uplo = Uplo.FULL) {
        requireShape(a.rows == a.cols) { "syr2: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows && y.size == a.rows) {
            "syr2: operand lengths ${x.size} and ${y.size} must both be ${a.rows}"
        }
        if (alpha == 0.0) return
        val n = a.rows
        val ad = a.data
        val xs = x.toDoubleArray()
        val ys = y.toDoubleArray()
        for (j in 0 until n) {
            for (i in j until n) {
                val v = alpha * (xs[i] * ys[j] + ys[i] * xs[j])
                if (v != 0.0) addUplo(ad, n, i, j, v, uplo)
            }
        }
    }

    /** `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     *  [transpose]. Writes the triangles [uplo] selects. */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    public fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo = Uplo.FULL,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(b.rows == a.rows && b.cols == a.cols) {
            "syr2k: B is ${b.rows}x${b.cols}, expected ${a.rows}x${a.cols} to match A"
        }
        requireShape(c.rows == n && c.cols == n) { "syr2k: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        scaleUplo(vectorKernels, c.data, n, beta, uplo)
        if (alpha == 0.0 || n == 0 || k == 0) return
        val cd = c.data
        for (j in 0 until n) {
            for (i in j until n) {
                var s = 0.0
                if (transpose) {
                    for (p in 0 until k) {
                        s += a.getUnsafe(p, i) * b.getUnsafe(p, j) + b.getUnsafe(p, i) * a.getUnsafe(p, j)
                    }
                } else {
                    for (p in 0 until k) {
                        s += a.getUnsafe(i, p) * b.getUnsafe(j, p) + b.getUnsafe(i, p) * a.getUnsafe(j, p)
                    }
                }
                if (s != 0.0) addUplo(cd, n, i, j, alpha * s, uplo)
            }
        }
    }

    /** Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     *  `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out. */
    public fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ) {
        requireShape(a.rows == a.cols) { "trsv requires a square matrix; got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "trsv: x length ${x.size} != ${a.rows}" }
        trsvCore(vectorKernels, a.data, a.rows, x, lower = lower, transpose = transpose, unitDiag = unitDiag)
    }

    /** Solve `op(T) · X = B` in place, or `X · op(T) = B` when [right] (BLAS `dtrsm`). Flags follow [trsv];
     *  the right-hand sides are the columns of [b] from the left and its rows from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
    ) {
        requireShape(a.rows == a.cols) { "trsm requires a square matrix; got ${a.rows}x${a.cols}" }
        if (right) {
            requireShape(b.cols == a.rows) { "trsm right: B has ${b.cols} cols, expected ${a.rows}" }
            forEachRow(a.rows, b) { row ->
                trsvCore(
                    vectorKernels,
                    a.data,
                    a.rows,
                    row,
                    lower = lower,
                    transpose = !transpose,
                    unitDiag = unitDiag,
                )
            }
        } else {
            requireShape(b.rows == a.rows) { "trsm: B has ${b.rows} rows, expected ${a.rows}" }
            trsmCore(vectorKernels, a.data, a.rows, b.data, b.cols, lower, transpose, unitDiag)
        }
    }

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    public fun trmv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ) {
        requireShape(a.rows == a.cols) { "trmv requires a square matrix; got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "trmv: x length ${x.size} != ${a.rows}" }
        trmvCore(vectorKernels, a.data, a.rows, x, lower = lower, transpose = transpose, unitDiag = unitDiag)
    }

    /** `B = op(T) · B`, or `B = B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    public fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
    ) {
        requireShape(a.rows == a.cols) { "trmm requires a square matrix; got ${a.rows}x${a.cols}" }
        if (right) {
            requireShape(b.cols == a.rows) { "trmm right: B has ${b.cols} cols, expected ${a.rows}" }
            forEachRow(a.rows, b) { row ->
                trmvCore(
                    vectorKernels,
                    a.data,
                    a.rows,
                    row,
                    lower = lower,
                    transpose = !transpose,
                    unitDiag = unitDiag,
                )
            }
        } else {
            requireShape(b.rows == a.rows) { "trmm: B has ${b.rows} rows, expected ${a.rows}" }
            trmmCore(vectorKernels, a.data, a.rows, b.data, b.cols, lower, transpose, unitDiag)
        }
    }

    /** [gemv] over [DenseVector] operands, writing [y] in place. */
    public fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        transpose: Boolean = false,
    ): Unit = gemv(alpha, a, x.data, beta, y.data, transpose)

    /** [gemv] over a [DenseVector], into a fresh result. */
    public fun gemv(a: DenseMatrix, x: DenseVector, transpose: Boolean = false): DenseVector =
        DenseVector.wrap(gemv(a, x.data, transpose))

    /** [trsv] over a [DenseVector], solving in place. */
    public fun trsv(
        a: DenseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ): Unit = trsv(a, x.data, lower, transpose, unitDiag)

    /** [trmv] over a [DenseVector], multiplying [x] in place. */
    public fun trmv(
        a: DenseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ): Unit = trmv(a, x.data, lower, transpose, unitDiag)
}
