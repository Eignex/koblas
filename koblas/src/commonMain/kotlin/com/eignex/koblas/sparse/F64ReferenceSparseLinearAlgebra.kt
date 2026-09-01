package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.dense.axpyArithmetic
import com.eignex.koblas.dense.borrowTransposed
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64ProductFormBasisSolver
import com.eignex.koblas.sparse.internal.multiplySparse
import com.eignex.koblas.sparse.internal.transposeCsc
import kotlin.math.abs
import kotlin.math.min

/** Dense right-hand sides processed per walk of portable CSC storage. */
internal const val REFERENCE_SPARSE_RHS_WIDTH: Int = 4

/** Visits dense right-hand sides in cache-sized panels. */
private inline fun forEachRhsPanel(columns: Int, action: (start: Int, width: Int) -> Unit) {
    var start = 0
    while (start < columns) {
        val width = min(REFERENCE_SPARSE_RHS_WIDTH, columns - start)
        action(start, width)
        start += width
    }
}

/**
 * A portable sparse backend, available on every target. The sparse seams declare their routines and this
 * implements them, so a binding that means to accelerate one cannot inherit the portable version by accident.
 *
 * @property configuredKernels dense vector kernels used by sparse matrix scaling, or null to follow [koblas].
 */
@Suppress("TooManyFunctions") // the sparse surface a backend half covers
public open class F64ReferenceSparseBackend(public val configuredKernels: F64Kernels? = null) :
    F64SparseLinearAlgebra,
    F64SparseKernels,
    F64GeneralSparseLu,
    F64SparseCholesky,
    F64QuasiDefiniteLdl,
    F64SparseQr,
    F64BasisFactorizations {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    private val denseKernels: F64Kernels get() = configuredKernels ?: koblas.kernels

    /** The product form over this backend's own factorization, so the portable half stays portable. */
    override fun basisSolver(a: F64SparseMatrix): F64BasisSolver = F64ProductFormBasisSolver(a, this)

    override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization {
        requireSquare(basis, "factorBasis")
        return F64RefactoringBasisFactorization(this, basis, factor(basis))
    }

    @Suppress("LongParameterList") // the BLAS dgemv signature
    override fun gemv(
        alpha: Double,
        a: F64SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        if (a.rows == 0 || a.cols == 0) return
        applyBeta(denseKernels, y, 0, y.size, beta)
        if (alpha == 0.0) return
        if (transpose) {
            for (j in 0 until a.cols) {
                var s = 0.0
                a.forEachInColumn(j) { i, v -> s += v * x[i] }
                y[j] += alpha * s
            }
        } else {
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                a.forEachInColumn(j) { i, v -> y[i] += v * xj }
            }
        }
    }

    override fun transpose(a: F64SparseMatrix): F64SparseMatrix = transposeCsc(a)

    override fun trsv(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trsv")
        val n = a.rows
        requireShape(x.size == n) { "trsv: x length ${x.size} != $n" }
        trsvCore(a, x, 0, lower, transpose, unitDiag)
    }

    override fun trmv(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trmv")
        val n = a.rows
        requireShape(x.size == n) { "trmv: x length ${x.size} != $n" }
        triangularMultiply(a.stableFor(x), x, 0, 1, lower, transpose, unitDiag)
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature, plus the side the sparse operand sits on
    override fun gemm(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        right: Boolean,
        workspace: Workspace?,
    ) {
        val aRows = if (transposeA) a.cols else a.rows
        val aCols = if (transposeA) a.rows else a.cols
        val bRows = if (transposeB) b.cols else b.rows
        val bCols = if (transposeB) b.rows else b.cols
        val m = if (right) bRows else aRows
        val n = if (right) aCols else bCols
        requireShape(if (right) bCols == aRows else aCols == bRows) {
            val first = if (right) "${bRows}x$bCols" else "${aRows}x$aCols"
            val second = if (right) "${aRows}x$aCols" else "${bRows}x$bCols"
            "gemm: $first does not meet $second"
        }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        applyBeta(denseKernels, c.data, 0, c.data.size, beta)
        if (alpha == 0.0) return
        if (right) {
            multiplyFromTheRight(alpha, a, transposeA, b, transposeB, c, m, workspace)
        } else {
            multiplyFromTheLeft(alpha, a, transposeA, b, transposeB, c, m, n, aCols, workspace)
        }
    }

    /** `C += alpha · op(A) · op(B)`, reusing each walk of the sparse operand over a small RHS panel. */
    @Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
    private fun multiplyFromTheLeft(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        c: F64DenseMatrix,
        m: Int,
        n: Int,
        k: Int,
        workspace: Workspace?,
    ) {
        val cd = c.data
        val bd = b.data
        val ld = b.rows
        workspace.borrow(REFERENCE_SPARSE_RHS_WIDTH) { work ->
            forEachRhsPanel(n) { columnStart, width ->
                val columnEnd = columnStart + width
                if (transposeA) {
                    for (i in 0 until m) {
                        work.fill(0.0, 0, width)
                        a.forEachInColumn(i) { j, v ->
                            for (rhs in 0 until width) {
                                val l = columnStart + rhs
                                work[rhs] += v * bd[if (transposeB) l + j * ld else j + l * ld]
                            }
                        }
                        for (rhs in 0 until width) cd[(columnStart + rhs) * m + i] += alpha * work[rhs]
                    }
                } else {
                    for (j in 0 until k) {
                        for (rhs in 0 until width) {
                            val l = columnStart + rhs
                            val raw = bd[if (transposeB) l + j * ld else j + l * ld]
                            work[rhs] = alpha * raw
                        }
                        a.forEachInColumn(j) { i, v ->
                            for (rhs in 0 until width) {
                                cd[(columnStart + rhs) * m + i] += v * work[rhs]
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * `C += alpha · op(B) · op(A)`. Each stored entry of the sparse operand scales one column of the dense
     * one into one column of the destination, so the walk is over storage either way the sparse operand is
     * transposed and only which index names which column changes.
     */
    @Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
    private fun multiplyFromTheRight(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        c: F64DenseMatrix,
        m: Int,
        workspace: Workspace?,
    ) {
        val cd = c.data
        if (transposeB) {
            workspace.borrowTransposed(b.data, b.rows, b.cols) { packed ->
                multiplyFromTheRightColumns(alpha, a, transposeA, cd, packed, m, m)
            }
        } else {
            multiplyFromTheRightColumns(alpha, a, transposeA, cd, b.data, m, b.rows)
        }
    }

    private fun multiplyFromTheRightColumns(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        c: DoubleArray,
        b: DoubleArray,
        rows: Int,
        leadingDimension: Int,
    ) {
        for (column in 0 until a.cols) {
            a.forEachInColumn(column) { row, v ->
                val cOff = (if (transposeA) row else column) * rows
                val bColumn = if (transposeA) column else row
                axpyArithmetic(denseKernels, c, cOff, alpha * v, b, bColumn * leadingDimension, rows)
            }
        }
    }

    override fun gemm(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix {
        requireShape(a.cols == b.rows) { "gemm: ${a.rows}x${a.cols} does not meet ${b.rows}x${b.cols}" }
        return multiplySparse(a, b)
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        requireSquare(a, "trsm")
        val n = a.rows
        if (right) {
            requireShape(b.cols == n) { "trsm right: B has ${b.cols} cols, expected $n" }
        } else {
            requireShape(b.rows == n) { "trsm: B has ${b.rows} rows, expected $n" }
        }
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        if (n == 0) return
        if (alpha != 1.0) denseKernels.scale(b.data, 0, alpha, b.data.size)
        val rightHandSides = if (right) b.rows else b.cols
        if (rightHandSides == 0) return
        if (right) {
            withExplicitDiagonal(a, n, unitDiag, workspace) { diagonal ->
                trsmRightCore(a, b, lower, !transpose, diagonal)
            }
        } else {
            workspace.borrow(REFERENCE_SPARSE_RHS_WIDTH) { work ->
                withExplicitDiagonal(a, n, unitDiag, workspace) { diagonal ->
                    trsmLeftCore(a, b, lower, transpose, diagonal, work)
                }
            }
        }
    }

    /** Runs [block] with the diagonal of [a] borrowed from [workspace], or null when [unitDiag] takes it as 1. */
    private inline fun withExplicitDiagonal(
        a: F64SparseMatrix,
        n: Int,
        unitDiag: Boolean,
        workspace: Workspace?,
        block: (DoubleArray?) -> Unit,
    ) {
        if (unitDiag) {
            block(null)
        } else {
            workspace.borrow(n) { diagonal ->
                for (j in 0 until n) diagonal[j] = a[j, j]
                block(diagonal)
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) {
        requireSquare(a, "trmm")
        val n = a.rows
        if (right) {
            requireShape(b.cols == n) { "trmm right: B has ${b.cols} cols, expected $n" }
        } else {
            requireShape(b.rows == n) { "trmm: B has ${b.rows} rows, expected $n" }
        }
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        val triangle = a.stableFor(b.data)
        if (alpha != 1.0) denseKernels.scale(b.data, 0, alpha, b.data.size)
        // Read once for every right-hand side rather than once per triangularMultiply call, as trsm does.
        val diagonal = if (unitDiag) null else DoubleArray(n) { triangle[it, it] }
        if (right) {
            trmmRightCore(triangle, b, lower, transpose, unitDiag, diagonal)
        } else {
            trmmLeftCore(triangle, b, lower, transpose, unitDiag, diagonal)
        }
    }

    /** Sparse triangular multiply over RHS panels, so values and indices are read once for several dense columns. */
    private fun trmmLeftCore(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        diagonal: DoubleArray?,
    ) {
        val n = a.rows
        val bd = b.data
        val work = DoubleArray(REFERENCE_SPARSE_RHS_WIDTH)
        val order = if (lower != transpose) n - 1 downTo 0 else 0 until n
        forEachRhsPanel(b.cols) { columnStart, width ->
            for (j in order) {
                val dj = diagonal?.get(j) ?: 1.0
                if (!transpose) {
                    // A stored zero source lane is skipped entirely, mirroring triangularMultiply's
                    // guardZeroInput default: it keeps a zero lane out of both the diagonal write and
                    // the scatter, so a NaN/Inf coefficient elsewhere in the column never reaches it.
                    var active = 0
                    for (rhs in 0 until width) {
                        val at = (columnStart + rhs) * n + j
                        val xj = bd[at]
                        work[rhs] = xj
                        if (xj != 0.0) {
                            active = active or (1 shl rhs)
                            bd[at] = if (unitDiag) xj else dj * xj
                        }
                    }
                    a.forEachInColumn(j) { i, v ->
                        if (if (lower) i > j else i < j) {
                            for (rhs in 0 until width) {
                                if (active and (1 shl rhs) != 0) bd[(columnStart + rhs) * n + i] += v * work[rhs]
                            }
                        }
                    }
                } else {
                    for (rhs in 0 until width) {
                        val at = (columnStart + rhs) * n + j
                        work[rhs] = if (unitDiag) bd[at] else dj * bd[at]
                    }
                    a.forEachInColumn(j) { i, v ->
                        if (if (lower) i > j else i < j) {
                            for (rhs in 0 until width) work[rhs] += v * bd[(columnStart + rhs) * n + i]
                        }
                    }
                    for (rhs in 0 until width) bd[(columnStart + rhs) * n + j] = work[rhs]
                }
            }
        }
    }

    /**
     * Right multiply over contiguous dense columns, which turns every sparse update into a Level 1 operation,
     * the same trade [trsmRightCore] makes. A row times op(T) is op(T)ᵀ times its column-shaped view, so this
     * walks the triangle exactly as [trmmLeftCore] does with the transpose flag flipped, only every scalar lane
     * of that algorithm is a whole column of [b] here instead of one right-hand side in a panel.
     */
    private fun trmmRightCore(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        diagonal: DoubleArray?,
    ) {
        val rows = b.rows
        if (rows == 0) return
        val bd = b.data
        val n = a.rows
        val gather = !transpose
        val order = if (lower != gather) n - 1 downTo 0 else 0 until n
        for (l in order) {
            val lOff = l * rows
            if (gather) {
                if (!unitDiag) denseKernels.scale(bd, lOff, diagonal!![l], rows)
                a.forEachInColumn(l) { i, v ->
                    if (v != 0.0 && (if (lower) i > l else i < l)) {
                        denseKernels.axpy(bd, lOff, v, bd, i * rows, rows)
                    }
                }
            } else {
                // The diagonal write must follow the scatter: it overwrites this column's own slot, which
                // the scatter below still needs to read at its pre-multiply value.
                a.forEachInColumn(l) { i, v ->
                    if (v != 0.0 && (if (lower) i > l else i < l)) {
                        denseKernels.axpy(bd, i * rows, v, bd, lOff, rows)
                    }
                }
                if (!unitDiag) denseKernels.scale(bd, lOff, diagonal!![l], rows)
            }
        }
    }

    /**
     * Sparse dtrmv over a strided vector. Direction preserves each source before its destination is written,
     * so no work buffer is needed for the ordinary in-place case.
     *
     * [diagonal], when given, is read instead of probing [a] for `a[j, j]`; callers that walk several
     * right-hand sides against the same triangle (see [trmm]) precompute it once rather than paying a
     * binary search per column per call.
     */
    @Suppress("LongParameterList") // the strided vector and BLAS triangle selectors
    private fun triangularMultiply(
        a: F64SparseMatrix,
        x: DoubleArray,
        offset: Int,
        stride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        diagonal: DoubleArray? = null,
        guardZeroInput: Boolean = true,
        guardZeroMatrix: Boolean = false,
    ) {
        val n = a.rows
        if (!transpose) {
            val order = if (lower) n - 1 downTo 0 else 0 until n
            for (j in order) {
                val xj = x[offset + j * stride]
                if (!guardZeroInput || xj != 0.0) {
                    x[offset + j * stride] = if (unitDiag) xj else (diagonal?.get(j) ?: a[j, j]) * xj
                    a.forEachInColumn(j) { i, v ->
                        if ((if (lower) i > j else i < j) && (!guardZeroMatrix || v != 0.0)) {
                            val at = offset + i * stride
                            x[at] += v * xj
                        }
                    }
                }
            }
        } else {
            val order = if (lower) 0 until n else n - 1 downTo 0
            for (j in order) {
                var sum = if (unitDiag) {
                    x[offset + j * stride]
                } else {
                    (diagonal?.get(j) ?: a[j, j]) * x[offset + j * stride]
                }
                a.forEachInColumn(j) { i, v ->
                    if ((if (lower) i > j else i < j) && (!guardZeroMatrix || v != 0.0)) {
                        sum += v * x[offset + i * stride]
                    }
                }
                x[offset + j * stride] = sum
            }
        }
    }

    /**
     * Snapshots the coefficient array only when the in-place destination aliases this matrix's live values.
     * The column pointers and row indices are shared live rather than copied: [triangularMultiply] never
     * mutates them, and they are documented immutable for the life of a [F64SparseMatrix].
     */
    @OptIn(UnsafeKoblasApi::class)
    private fun F64SparseMatrix.stableFor(destination: DoubleArray): F64SparseMatrix = if (values === destination) {
        F64SparseMatrix.wrap(rows, cols, colPtr, rowIdx, values.copyOf())
    } else {
        this
    }

    /** Sparse substitution over RHS panels, so values and indices are read once for several dense columns. */
    private fun trsmLeftCore(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        diagonal: DoubleArray?,
        work: DoubleArray,
    ) {
        val n = a.rows
        val bd = b.data
        val order = if (lower != transpose) 0 until n else n - 1 downTo 0
        forEachRhsPanel(b.cols) { columnStart, width ->
            for (j in order) {
                if (!transpose) {
                    val divisor = diagonal?.get(j) ?: 1.0
                    var active = 0
                    for (rhs in 0 until width) {
                        val at = (columnStart + rhs) * n + j
                        val raw = bd[at]
                        work[rhs] = if (raw == 0.0) 0.0 else raw / divisor
                        if (raw != 0.0) {
                            active = active or (1 shl rhs)
                            bd[at] = work[rhs]
                        }
                    }
                    a.forEachInColumn(j) { i, v ->
                        if (if (lower) i > j else i < j) {
                            for (rhs in 0 until width) {
                                val xj = work[rhs]
                                if (active and (1 shl rhs) != 0) bd[(columnStart + rhs) * n + i] -= v * xj
                            }
                        }
                    }
                } else {
                    for (rhs in 0 until width) work[rhs] = bd[(columnStart + rhs) * n + j]
                    a.forEachInColumn(j) { i, v ->
                        if (if (lower) i > j else i < j) {
                            for (rhs in 0 until width) work[rhs] -= v * bd[(columnStart + rhs) * n + i]
                        }
                    }
                    val divisor = diagonal?.get(j) ?: 1.0
                    for (rhs in 0 until width) bd[(columnStart + rhs) * n + j] = work[rhs] / divisor
                }
            }
        }
    }

    /** Right solve over contiguous dense columns, which turns every sparse update into a Level 1 operation. */
    private fun trsmRightCore(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        diagonal: DoubleArray?,
    ) {
        val rows = b.rows
        if (rows == 0) return
        val bd = b.data
        val order = if (lower != transpose) 0 until a.rows else a.rows - 1 downTo 0
        for (j in order) {
            val jOff = j * rows
            if (!transpose) {
                if (diagonal != null) denseKernels.scale(bd, jOff, 1.0 / diagonal[j], rows)
                a.forEachInColumn(j) { i, v ->
                    if (v != 0.0 && (if (lower) i > j else i < j)) {
                        denseKernels.axpy(bd, i * rows, -v, bd, jOff, rows)
                    }
                }
            } else {
                a.forEachInColumn(j) { i, v ->
                    if (v != 0.0 && (if (lower) i > j else i < j)) {
                        denseKernels.axpy(bd, jOff, -v, bd, i * rows, rows)
                    }
                }
                if (diagonal != null) denseKernels.scale(bd, jOff, 1.0 / diagonal[j], rows)
            }
        }
    }

    /** [trsv] over the `n` entries of [x] from [offset], the columns of a right-hand side matrix being those. */
    @Suppress("LongParameterList") // the three BLAS triangle flags plus the segment
    private fun trsvCore(
        a: F64SparseMatrix,
        x: DoubleArray,
        offset: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
    ) {
        val n = a.rows
        // Forward when a finished unknown feeds later columns, backward when it feeds earlier ones.
        val order = if (lower != transpose) 0 until n else n - 1 downTo 0
        for (j in order) {
            if (!transpose) {
                val raw = x[offset + j]
                if (raw == 0.0) continue
                val xj = if (unitDiag) raw else raw / a[j, j]
                x[offset + j] = xj
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) x[offset + i] -= v * xj
                }
            } else {
                var s = x[offset + j]
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) s -= v * x[offset + i]
                }
                x[offset + j] = if (unitDiag) s else s / a[j, j]
            }
        }
    }

    /** The portable factorization at its default policy; [F64ReferenceSparseDecompositions] carries the knobs. */
    override fun factor(a: F64SparseMatrix): F64SparseLuFactorization = F64ReferenceSparseDecompositions.factor(a)

    override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization =
        F64ReferenceSparseDecompositions.cholesky(a)

    override fun quasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization =
        F64ReferenceSparseDecompositions.quasiDefiniteLdl(a)

    override fun qr(a: F64SparseMatrix): F64SparseQrFactorization = F64ReferenceSparseDecompositions.qr(a)

    override fun dot(x: F64SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var s = 0.0
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) s += vals[k] * y[idx[k]]
        return s
    }

    override fun dot(x: F64SparseVector, y: F64SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        var s = 0.0
        var a = 0
        var b = 0
        while (a < x.indices.size && b < y.indices.size) {
            val ia = x.indices[a]
            val ib = y.indices[b]
            when {
                ia < ib -> a++

                ia > ib -> b++

                else -> {
                    s += x.values[a] * y.values[b]
                    a++
                    b++
                }
            }
        }
        return s
    }

    override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector) {
        requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0) return
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) y[idx[k]] += alpha * vals[k]
    }

    override fun scatter(x: F64SparseVector, out: DoubleArray) {
        requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) out[idx[k]] = vals[k]
    }

    override fun gather(x: F64SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) vals[k] = from[idx[k]]
    }

    override fun gatherZero(x: F64SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
        val idx = x.indices
        val vals = x.values
        for (k in idx.indices) {
            val i = idx[k]
            vals[k] = from[i]
            from[i] = 0.0
        }
    }

    override fun nrm2(x: F64SparseVector): Double = euclideanNorm(x.values, 0, x.values.size)

    override fun asum(x: F64SparseVector): Double {
        var s = 0.0
        for (v in x.values) s += abs(v)
        return s
    }
}

/** The shared portable sparse backend used by the process-wide registry. */
public object F64ReferenceSparseLinearAlgebra : F64ReferenceSparseBackend()
