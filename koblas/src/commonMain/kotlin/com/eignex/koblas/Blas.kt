@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas

/**
 * The level-2 and level-3 BLAS routines, the seam a native BLAS plugs into.
 *
 * Everything is over flat, contiguous [DenseMatrix.data] / [DoubleArray] buffers, so a native backend
 * passes them across the FFI boundary without repacking. Routines keep their standard mnemonics, and
 * every one of them dispatches: a name here means the standard routine with the standard semantics.
 *
 * Level-1 kernels (`dot`, `axpy`, `scale`) are deliberately absent. They do nanoseconds of work, so a
 * per-call virtual dispatch would cost more than the kernel; they are specialized at compile time
 * instead, and reach a host BLAS through `Level1Kernels` where that pays.
 *
 * Defaults implement every routine in portable Kotlin, so a backend overrides only what it accelerates.
 */
interface Blas : Backend {
    /**
     * In-place matrix-vector accumulate `y = alpha · op(A) · x + beta · y` (full BLAS `dgemv`),
     * where `op(A)` is `Aᵀ` when [transpose]. Per BLAS convention, `beta == 0.0` overwrites [y]
     * without reading it (it may be uninitialized), and `alpha == 0.0` reduces to the `beta` scale.
     */
    fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false)

    /** Matrix-vector product `A · x`, or `Aᵀ · x` when [transpose], into a fresh result (restricted
     *  [gemv] with `alpha = 1, beta = 0`). */
    fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /**
     * In-place matrix-matrix accumulate `C = alpha · op(A) · op(B) + beta · C` (full BLAS `dgemm`),
     * where `op` transposes its operand when [transposeA] / [transposeB] is set. Shapes must satisfy
     * `op(A): m×k`, `op(B): k×n`, `C: m×n`. Per BLAS convention, `beta == 0.0` overwrites [c] without
     * reading it, and `alpha == 0.0` reduces to the `beta` scale.
     */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    )

    /** Matrix-matrix product `A · B` into a fresh matrix (restricted [gemm] with `alpha = 1, beta = 0`);
     *  `A.cols` must equal `B.rows`. */
    fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }

    /**
     * In-place symmetric rank-k accumulate `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C`
     * when [transpose] (BLAS `dsyrk`). With the default [Uplo.FULL] the full symmetric result is
     * produced (the alpha term is applied to both triangles, and beta scales all of [c]); with
     * [Uplo.LOWER] / [Uplo.UPPER] the standard `dsyrk` semantics apply — only the selected triangle is
     * written and beta-scaled, the opposite strict triangle untouched. [c] must be square with
     * dimension `op(A).rows`. Per BLAS convention, `beta == 0.0` overwrites without reading (within
     * the written region), and `alpha == 0.0` reduces to the `beta` scale.
     */
    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo = Uplo.FULL,
        workspace: Workspace? = null,
    )

    /**
     * In-place symmetric matrix-vector accumulate `y = alpha · A · x + beta · y` for a symmetric [a]
     * (BLAS `dsymv`). Only the triangle selected by [lower] (diagonal included) is read; the opposite
     * strict triangle may hold anything. Exploits symmetry for roughly half the memory traffic of
     * [gemv]. Per BLAS convention, `beta == 0.0` overwrites [y] without reading it, and `alpha == 0.0`
     * reduces to the `beta` scale.
     */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean = true)

    /**
     * In-place symmetric matrix-matrix accumulate `C = alpha · A · B + beta · C`, or
     * `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). As with [symv], only the triangle of
     * the symmetric [a] selected by [lower] is read. Shapes: [b] and [c] agree, and [a] is square with
     * dimension `B.rows` (left) or `B.cols` (right). Per BLAS convention, `beta == 0.0` overwrites [c]
     * without reading it, and `alpha == 0.0` reduces to the `beta` scale.
     */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
    )

    /**
     * Rank-one update `A = A + alpha · x · yᵀ` (BLAS `dger`).
     *
     * The free `ger` accepts [VectorView] operands and takes a sparse fast path; this form is the dense
     * one a backend can dispatch.
     */
    fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        require(a.rows == x.size && a.cols == y.size) {
            "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
        }
        if (alpha == 0.0) return
        // One axpy per column of A, each writing a contiguous run: A[:,j] += (alpha·y_j)·x.
        for (j in 0 until a.cols) {
            val scaled = alpha * y[j]
            if (scaled != 0.0) denseAxpy(a.data, a.colOffset(j), scaled, x, 0, a.rows)
        }
    }

    /**
     * Solve `op(T) · x = b` in place (BLAS `dtrsv`), where `T` is the [lower] or upper triangle of the
     * square [a], `op` transposes when [transpose], and [unitDiag] takes the diagonal as 1 without
     * reading it. [x] holds the right-hand side on entry and the solution on return. Only the selected
     * triangle is read, so the rest of [a] may hold anything.
     */
    fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) {
        require(a.rows == a.cols) { "trsv requires a square matrix; got ${a.rows}x${a.cols}" }
        require(x.size == a.rows) { "trsv: x length ${x.size} != ${a.rows}" }
        trsvCore(a.data, a.rows, x, lower = lower, transpose = transpose, unitDiag = unitDiag)
    }

    /**
     * Solve `op(T) · X = B` in place, or `X · op(T) = B` when [right] (BLAS `dtrsm`): [b] holds the
     * right-hand sides on entry and the solutions on return. Flags follow [trsv]. From the left the
     * right-hand sides are the columns of [b]; from the right, its rows.
     */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
    ) {
        require(a.rows == a.cols) { "trsm requires a square matrix; got ${a.rows}x${a.cols}" }
        if (right) {
            require(b.cols == a.rows) { "trsm right: B has ${b.cols} cols, expected ${a.rows}" }
            forEachRow(a.rows, b) { row ->
                trsvCore(a.data, a.rows, row, lower = lower, transpose = !transpose, unitDiag = unitDiag)
            }
        } else {
            require(b.rows == a.rows) { "trsm: B has ${b.rows} rows, expected ${a.rows}" }
            trsmCore(a.data, a.rows, b.data, b.cols, lower, transpose, unitDiag)
        }
    }

    /** Multiply `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) {
        require(a.rows == a.cols) { "trmv requires a square matrix; got ${a.rows}x${a.cols}" }
        require(x.size == a.rows) { "trmv: x length ${x.size} != ${a.rows}" }
        trmvCore(a.data, a.rows, x, lower = lower, transpose = transpose, unitDiag = unitDiag)
    }

    /** Multiply `B = op(T) · B`, or `B = B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of
     *  [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
    ) {
        require(a.rows == a.cols) { "trmm requires a square matrix; got ${a.rows}x${a.cols}" }
        if (right) {
            require(b.cols == a.rows) { "trmm right: B has ${b.cols} cols, expected ${a.rows}" }
            forEachRow(a.rows, b) { row ->
                trmvCore(a.data, a.rows, row, lower = lower, transpose = !transpose, unitDiag = unitDiag)
            }
        } else {
            require(b.rows == a.rows) { "trmm: B has ${b.rows} rows, expected ${a.rows}" }
            trmmCore(a.data, a.rows, b.data, b.cols, lower, transpose, unitDiag)
        }
    }
}
