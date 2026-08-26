@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64VectorLike
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The portable dense matrix routines, the semantic reference a native [F64Blas] is validated against.
 *
 * @param configured the kernels the inner loops use, or null to follow the [F64Context] default.
 */
internal class F64ReferenceBlas(private val configured: F64Kernels? = null) : F64Blas {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** These routines' kernels, or the process default when they were given none. */
    override val kernels: F64Kernels get() = configured ?: koblas.kernels

    override fun gemv(
        alpha: Double,
        a: F64DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        applyBeta(kernels, y, 0, y.size, beta)
        if (alpha == 0.0) return
        val ad = a.data
        val rows = a.rows
        if (!transpose) {
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                if (xj != 0.0) kernels.axpy(y, 0, xj, ad, j * rows, rows)
            }
        } else {
            val quads = DoubleArray(4)
            var j = 0
            val bound = a.cols - 3
            while (j < bound) {
                kernels.dot4(ad, j * rows, rows, x, 0, rows, quads, 0)
                y[j] += alpha * quads[0]
                y[j + 1] += alpha * quads[1]
                y[j + 2] += alpha * quads[2]
                y[j + 3] += alpha * quads[3]
                j += 4
            }
            while (j < a.cols) {
                y[j] += alpha * kernels.dot(ad, j * rows, x, 0, rows)
                j++
            }
        }
    }

    override fun transpose(a: F64DenseMatrix): F64DenseMatrix {
        val t = F64DenseMatrix(a.cols, a.rows)
        val td = t.data
        val ad = a.data
        for (j in 0 until a.cols) {
            val base = j * a.rows
            for (i in 0 until a.rows) td[j + i * a.cols] = ad[base + i]
        }
        return t
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod")
    override fun gemm(
        alpha: Double,
        a: F64DenseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val kB = if (transposeB) b.cols else b.rows
        val n = if (transposeB) b.rows else b.cols
        requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        val cd = c.data
        applyBeta(kernels, cd, 0, cd.size, beta)
        if (alpha == 0.0 || m == 0 || n == 0 || k == 0) return
        val ad = a.data
        val bd = b.data
        val kernels = kernels
        if (transposeA && transposeB) {
            // With both transposed one operand is strided whichever way the loops run, so the smaller one
            // is transposed first. That copy is O(k·n) or O(k·m) against the product's O(m·k·n).
            if (b.rows * b.cols <= a.rows * a.cols) {
                val bt = DoubleArray(k * n)
                for (j in 0 until n) for (p in 0 until k) bt[p + j * k] = bd[j + p * n]
                gemm(alpha, a, true, F64DenseMatrix.wrap(k, n, bt), false, 1.0, c)
            } else {
                val at = DoubleArray(m * k)
                for (i in 0 until m) for (p in 0 until k) at[i + p * m] = ad[p + i * k]
                gemm(alpha, F64DenseMatrix.wrap(m, k, at), false, b, true, 1.0, c)
            }
            return
        }
        when {
            transposeA -> {
                val quads = DoubleArray(4)
                for (j in 0 until n) {
                    var i = 0
                    val bound = m - 3
                    while (i < bound) {
                        kernels.dot4(ad, i * k, k, bd, j * k, k, quads, 0)
                        cd[i + j * m] += alpha * quads[0]
                        cd[i + 1 + j * m] += alpha * quads[1]
                        cd[i + 2 + j * m] += alpha * quads[2]
                        cd[i + 3 + j * m] += alpha * quads[3]
                        i += 4
                    }
                    while (i < m) {
                        cd[i + j * m] += alpha * kernels.dot(ad, i * k, bd, j * k, k)
                        i++
                    }
                }
            }

            !transposeB -> for (j in 0 until n) {
                for (p in 0 until k) {
                    val bpj = alpha * bd[p + j * k]
                    if (bpj != 0.0) kernels.axpy(cd, j * m, bpj, ad, p * m, m)
                }
            }

            else -> for (j in 0 until n) {
                for (p in 0 until k) {
                    val bjp = alpha * bd[j + p * n]
                    if (bjp != 0.0) kernels.axpy(cd, j * m, bjp, ad, p * m, m)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    override fun syrk(
        alpha: Double,
        a: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        uplo: Uplo,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        val cd = c.data
        scaleUplo(kernels, cd, n, beta, uplo)
        if (alpha == 0.0 || n == 0 || k == 0) return
        val ad = a.data
        // The dot form holds each output entry in a register, where accumulating outer products would
        // stream all of C once per column of A. So a non-transposed A is transposed first, an O(n·k) copy
        // against the product's O(n²·k/2), and both orientations run the same loop below.
        if (transpose) {
            accumulateSyrk(alpha, ad, n, k, cd, uplo)
            return
        }
        workspace.borrow(n * k) { at ->
            transposeIntoRows(ad, at, n, k)
            accumulateSyrk(alpha, at, n, k, cd, uplo)
        }
    }

    /** The `dsyrk` accumulation with the operand stored `k×n`, so op(A) row i is column i of the buffer. */
    @Suppress("LongParameterList") // the operand and the shape, all of which the caller holds
    private fun accumulateSyrk(alpha: Double, ad: DoubleArray, n: Int, k: Int, cd: DoubleArray, uplo: Uplo) {
        val kernels = kernels
        // Four rows of op(A) against one column, so column j is read once for the four rather than once
        // each. The rows sit at stride k because the operand arrives transposed, which is the shape
        // [F64Kernels.dot4] is for.
        val quads = DoubleArray(4)
        for (j in 0 until n) {
            var i = j
            val bound = n - 3
            while (i < bound) {
                kernels.dot4(ad, i * k, k, ad, j * k, k, quads, 0)
                addUplo(cd, n, i, j, alpha * quads[0], uplo)
                addUplo(cd, n, i + 1, j, alpha * quads[1], uplo)
                addUplo(cd, n, i + 2, j, alpha * quads[2], uplo)
                addUplo(cd, n, i + 3, j, alpha * quads[3], uplo)
                i += 4
            }
            while (i < n) {
                addUplo(cd, n, i, j, alpha * kernels.dot(ad, i * k, ad, j * k, k), uplo)
                i++
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymv signature
    override fun symv(alpha: Double, a: F64DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireShape(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        requireShape(x.size == n) { "symv: x length ${x.size} != $n" }
        requireShape(y.size == n) { "symv: y length ${y.size} != $n" }
        applyBeta(kernels, y, 0, n, beta)
        if (alpha == 0.0) return
        symvAccumulate(alpha, a.data, n, x, 0, y, 0, lower)
    }

    /** Accumulates alpha times A times x into y for the symmetric `n×n` [ad], reading only the [lower] or
     *  upper triangle. */
    @Suppress("LongParameterList") // two buffers with offsets plus the triangle flag
    private fun symvAccumulate(
        alpha: Double,
        ad: DoubleArray,
        n: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        lower: Boolean,
    ) {
        val mult = DoubleArray(SYMV_BLOCK)
        val dots = DoubleArray(SYMV_BLOCK)
        var j = 0
        while (j + SYMV_BLOCK <= n) {
            for (r in 0 until SYMV_BLOCK) mult[r] = alpha * x[xOff + j + r]
            symvCorner(alpha, ad, n, x, xOff, y, yOff, j, mult, lower)
            // Below the corner for a lower triangle, above it for an upper.
            val runOff = if (lower) j + SYMV_BLOCK else 0
            val len = if (lower) n - j - SYMV_BLOCK else j
            if (len > 0) {
                if (mult.none { it == 0.0 }) {
                    kernels.symvColumn4(
                        ad,
                        runOff + j * n,
                        n,
                        x,
                        xOff + runOff,
                        y,
                        yOff + runOff,
                        mult,
                        dots,
                        len,
                    )
                } else {
                    // A zero multiplier must not write into y.
                    for (r in 0 until SYMV_BLOCK) {
                        dots[r] = symvOneColumn(
                            ad,
                            runOff + (j + r) * n,
                            x,
                            xOff + runOff,
                            y,
                            yOff + runOff,
                            mult[r],
                            len,
                        )
                    }
                }
                for (r in 0 until SYMV_BLOCK) y[yOff + j + r] += alpha * dots[r]
            }
            j += SYMV_BLOCK
        }
        while (j < n) {
            val base = j + j * n
            val xj = alpha * x[xOff + j]
            val runOff = if (lower) j + 1 else 0
            val len = if (lower) n - j - 1 else j
            val dot = symvOneColumn(ad, runOff + j * n, x, xOff + runOff, y, yOff + runOff, xj, len)
            val diagonal = if (xj != 0.0) xj * ad[base] else 0.0
            y[yOff + j] += alpha * dot + diagonal
            j++
        }
    }

    /** The corner at [j], where the block's columns have ragged extents and cannot share one run. */
    @Suppress("LongParameterList") // the operand, both vectors and the block position
    private fun symvCorner(
        alpha: Double,
        ad: DoubleArray,
        n: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        j: Int,
        mult: DoubleArray,
        lower: Boolean,
    ) {
        for (c in j until j + SYMV_BLOCK) {
            val mc = mult[c - j]
            val col = c * n
            if (mc != 0.0) y[yOff + c] += mc * ad[c + col]
            val from = if (lower) c + 1 else j
            val until = if (lower) j + SYMV_BLOCK else c
            for (i in from until until) {
                val aij = ad[i + col]
                if (mc != 0.0) y[yOff + i] += mc * aij
                if (x[xOff + i] != 0.0) y[yOff + c] += alpha * aij * x[xOff + i]
            }
        }
    }

    /**
     * One column's contribution to both halves, returning its dot with x. A zero multiplier takes the dot
     * alone, since `0.0` times an infinite entry is a NaN rather than nothing. The same rule covers the
     * diagonal at both call sites, so a zero entry of x reaches an infinite `A(j, j)` no more than [gemv]
     * reaches an infinite column it skips.
     *
     * The dot itself is the one place the rule cannot reach: it carries the reflected half of the run, whose
     * multipliers are the entries of x it reads, and telling them apart per element would cost the run its
     * kernel. An infinite off-diagonal entry against a zero x entry is a NaN here as it is in `dsymv`.
     */
    @Suppress("LongParameterList") // three runs, the multiplier and the length
    private fun symvOneColumn(
        ad: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: Double,
        len: Int,
    ): Double = if (mult != 0.0) {
        kernels.symvColumn(ad, aOff, x, xOff, y, yOff, mult, len)
    } else {
        kernels.dot(ad, aOff, x, xOff, len)
    }

    /** A(i, j) of a symmetric `n×n` matrix stored in its [lower] or upper triangle only. */
    private fun symEntry(ad: DoubleArray, n: Int, i: Int, j: Int, lower: Boolean): Double {
        val hi = if (i > j) i else j
        val lo = if (i > j) j else i
        return if (lower) ad[hi + lo * n] else ad[lo + hi * n]
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod") // the BLAS dsymm signature
    override fun symm(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        requireShape(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
        val m = a.rows
        requireShape(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        // Both sides check A against B before anything is written, since scaling C is part of the
        // operation and a call that cannot go through must not have performed half of it.
        if (right) {
            requireShape(b.cols == m) { "symm right: B has ${b.cols} cols, expected $m" }
        } else {
            requireShape(b.rows == m) { "symm: B has ${b.rows} rows, expected $m" }
        }
        val cd = c.data
        applyBeta(kernels, cd, 0, cd.size, beta)
        if (right) {
            if (alpha == 0.0 || m == 0 || b.rows == 0) return
            val rows = b.rows
            val ad = a.data
            val bd = b.data
            for (j in 0 until m) {
                for (p in 0 until m) {
                    val apj = alpha * symEntry(ad, m, p, j, lower)
                    if (apj != 0.0) kernels.axpy(cd, j * rows, apj, bd, p * rows, rows)
                }
            }
            return
        }
        if (alpha == 0.0 || m == 0 || b.cols == 0) return
        for (j in 0 until b.cols) {
            symvAccumulate(alpha, a.data, m, b.data, j * m, cd, j * m, lower)
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: F64DenseMatrix) {
        requireShape(a.rows == x.size && a.cols == y.size) {
            "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
        }
        if (alpha == 0.0) return
        val kernels = kernels
        for (j in 0 until a.cols) {
            val scaled = alpha * y[j]
            if (scaled != 0.0) kernels.axpy(a.data, a.colOffset(j), scaled, x, 0, a.rows)
        }
    }

    /**
     * `A += alpha · x · xᵀ` (BLAS `dsyr`), writing the triangles [uplo] selects. [syrk] is the rank-k form.
     *
     * A sparse x updates the outer product of its stored positions and nothing else, which is the whole of
     * the update: a position it does not store contributes a zero row and a zero column.
     */
    override fun syr(alpha: Double, x: F64VectorLike, a: F64DenseMatrix, uplo: Uplo) {
        requireShape(a.rows == a.cols) { "syr: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows) { "syr: x length ${x.size} != ${a.rows}" }
        if (alpha == 0.0) return
        val n = a.rows
        val ad = a.data
        if (x !is F64DenseVector) {
            val positions = storedPositions(x)
            val xs = DoubleArray(positions.size) { x[positions[it]] }
            for (b in positions.indices) {
                val xj = alpha * xs[b]
                if (xj == 0.0) continue
                for (c in b until positions.size) addUplo(ad, n, positions[c], positions[b], xj * xs[c], uplo)
            }
            return
        }
        val xs = x.data
        for (j in 0 until n) {
            val xj = alpha * xs[j]
            if (xj == 0.0) continue
            for (i in j until n) addUplo(ad, n, i, j, xj * xs[i], uplo)
        }
    }

    /**
     * `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing the triangles [uplo] selects.
     *
     * Sparse operands are handled as [syr] handles one, over the positions either of them stores: a position
     * neither stores has `x(i)` and `y(i)` both zero and so contributes nothing to any entry.
     */
    override fun syr2(alpha: Double, x: F64VectorLike, y: F64VectorLike, a: F64DenseMatrix, uplo: Uplo) {
        requireShape(a.rows == a.cols) { "syr2: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(x.size == a.rows && y.size == a.rows) {
            "syr2: operand lengths ${x.size} and ${y.size} must both be ${a.rows}"
        }
        if (alpha == 0.0) return
        val n = a.rows
        val ad = a.data
        if (x !is F64DenseVector || y !is F64DenseVector) {
            syr2Stored(alpha, x, y, ad, n, uplo)
            return
        }
        val xs = x.data
        val ys = y.data
        for (j in 0 until n) {
            // Column j of the update is x(j) times y plus y(j) times x, so both being zero empties it.
            if (xs[j] == 0.0 && ys[j] == 0.0) continue
            for (i in j until n) {
                val v = alpha * (xs[i] * ys[j] + ys[i] * xs[j])
                if (v != 0.0) addUplo(ad, n, i, j, v, uplo)
            }
        }
    }

    /** The positions [x] stores, ascending, which [forEachStored] visits in that order for any storage. */
    private fun storedPositions(x: F64VectorLike): IntArray {
        val gathered = ArrayList<Int>()
        x.forEachStored { i, _ -> gathered.add(i) }
        return gathered.toIntArray()
    }

    /** The positions [x] or [y] stores, ascending and without repeats, merged from the two ascending runs. */
    private fun storedPositions(x: F64VectorLike, y: F64VectorLike): IntArray {
        val left = storedPositions(x)
        val right = storedPositions(y)
        val merged = IntArray(left.size + right.size)
        var a = 0
        var b = 0
        var count = 0
        while (a < left.size || b < right.size) {
            merged[count++] = when {
                b == right.size -> left[a++]
                a == left.size -> right[b++]
                left[a] < right[b] -> left[a++]
                left[a] > right[b] -> right[b++]
                else -> left[a++].also { b++ }
            }
        }
        return merged.copyOf(count)
    }

    /** [syr2] over the positions [x] or [y] stores, gathered so each is read once rather than searched for. */
    @Suppress("LongParameterList") // the dsyr2 operands, less the matrix it has already been taken out of
    private fun syr2Stored(
        alpha: Double,
        x: F64VectorLike,
        y: F64VectorLike,
        ad: DoubleArray,
        n: Int,
        uplo: Uplo,
    ) {
        val positions = storedPositions(x, y)
        val xs = DoubleArray(positions.size) { x[positions[it]] }
        val ys = DoubleArray(positions.size) { y[positions[it]] }
        for (b in positions.indices) {
            if (xs[b] == 0.0 && ys[b] == 0.0) continue
            for (c in b until positions.size) {
                val v = alpha * (xs[c] * ys[b] + ys[c] * xs[b])
                if (v != 0.0) addUplo(ad, n, positions[c], positions[b], v, uplo)
            }
        }
    }

    /** `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     *  [transpose]. Writes the triangles [uplo] selects. */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature plus optional scratch
    override fun syr2k(
        alpha: Double,
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        uplo: Uplo,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(b.rows == a.rows && b.cols == a.cols) {
            "syr2k: B is ${b.rows}x${b.cols}, expected ${a.rows}x${a.cols} to match A"
        }
        requireShape(c.rows == n && c.cols == n) { "syr2k: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        scaleUplo(kernels, c.data, n, beta, uplo)
        if (alpha == 0.0 || n == 0 || k == 0) return
        // Each entry is two dots, which hold their accumulator in a register. A transposed operand already
        // has op(X) row i as column i of its buffer; a non-transposed one has it strided, so both are
        // transposed first, an O(n·k) copy against the product's O(n²·k), as [syrk] does with its one
        // operand. One triangle is computed and addUplo places it, which halves the dots and keeps the two
        // halves identical even against a host dot whose value depends on which operand comes first.
        if (transpose) {
            accumulateSyr2k(alpha, a.data, b.data, n, k, c.data, uplo)
            return
        }
        workspace.borrow(n * k) { at ->
            workspace.borrow(n * k) { bt ->
                transposeIntoRows(a.data, at, n, k)
                transposeIntoRows(b.data, bt, n, k)
                accumulateSyr2k(alpha, at, bt, n, k, c.data, uplo)
            }
        }
    }

    /** Column p of the `n×k` [src] becomes row p of the `k×n` [dst], so a row read turns into a column read. */
    private fun transposeIntoRows(src: DoubleArray, dst: DoubleArray, n: Int, k: Int) {
        for (p in 0 until k) {
            val base = p * n
            for (j in 0 until n) dst[p + j * k] = src[base + j]
        }
    }

    /** The `dsyr2k` accumulation with both operands stored `k×n`, so op(X) row i is column i of the buffer. */
    @Suppress("LongParameterList") // two operands and the shape, all of which the caller holds
    private fun accumulateSyr2k(
        alpha: Double,
        ad: DoubleArray,
        bd: DoubleArray,
        n: Int,
        k: Int,
        cd: DoubleArray,
        uplo: Uplo,
    ) {
        val kernels = kernels
        // Both halves of the pair are the shape dot4 wants, so each column is read once for four rows.
        val fromA = DoubleArray(4)
        val fromB = DoubleArray(4)
        for (j in 0 until n) {
            var i = j
            val bound = n - 3
            while (i < bound) {
                kernels.dot4(ad, i * k, k, bd, j * k, k, fromA, 0)
                kernels.dot4(bd, i * k, k, ad, j * k, k, fromB, 0)
                for (r in 0 until 4) {
                    val s = fromA[r] + fromB[r]
                    if (s != 0.0) addUplo(cd, n, i + r, j, alpha * s, uplo)
                }
                i += 4
            }
            while (i < n) {
                val s = kernels.dot(ad, i * k, bd, j * k, k) + kernels.dot(bd, i * k, ad, j * k, k)
                if (s != 0.0) addUplo(cd, n, i, j, alpha * s, uplo)
                i++
            }
        }
    }

    /** Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     *  `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out. */
    override fun trsv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
        triangularVector(kernels, a, x, lower, transpose, unitDiag, solve = true)

    /** Solve `op(T) · X = B` in place, or `X · op(T) = B` when [right] (BLAS `dtrsm`). Flags follow [trsv];
     *  the right-hand sides are the columns of [b] from the left and its rows from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) =
        triangularMatrix(kernels, a, b, lower, transpose, unitDiag, right, alpha, solve = true)

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    override fun trmv(a: F64DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) =
        triangularVector(kernels, a, x, lower, transpose, unitDiag, solve = false)

    /** `B = op(T) · B`, or `B = B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: F64DenseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) =
        triangularMatrix(kernels, a, b, lower, transpose, unitDiag, right, alpha, solve = false)
}

/** Columns a symmetric product takes per pass, matching the width [F64Kernels.symvColumn4] serves. */
private const val SYMV_BLOCK = 4
