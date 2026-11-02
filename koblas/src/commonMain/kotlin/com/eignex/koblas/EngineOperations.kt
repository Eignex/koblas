@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.configuration.*
import com.eignex.koblas.internal.dense.*
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.sparse.*
import com.eignex.koblas.vendor.*

/** `xᵀ · y`. */
public fun KoblasEngine.dot(x: DenseVector, y: DenseVector): Double {
    requireSameSize(x.size, y.size, "dot")
    return dot(x.values, x.offset, y.values, y.offset, x.size, x.stride, y.stride)
}

/** `‖x‖₂`, computed by the selected provider's overflow-avoiding method. */
public fun KoblasEngine.nrm2(x: DenseVector): Double = nrm2(x.values, x.offset, x.size, x.stride)

/** `Σ|xᵢ|`. */
public fun KoblasEngine.asum(x: DenseVector): Double = asum(x.values, x.offset, x.size, x.stride)

/**
 * The index of the entry of largest magnitude, or 0 for an empty vector.
 *
 * BLAS defines its three reductions to return zero for a non-positive increment rather than to walk the
 * vector backwards, so a negatively strided vector is passed with the increment's magnitude and reaches
 * the same entries in the opposite order. For [nrm2] and [asum] the order does not matter. For this one it
 * decides ties: among entries of equal largest magnitude a forward vector reports the first and a backward
 * one reports the last. That is a real difference from a single-pass forward scan, and it is documented
 * rather than papered over because no vendor flag expresses the forward-scan choice.
 */
public fun KoblasEngine.iamax(x: DenseVector): Int = if (x.size == 0) 0 else iamax(x.values, x.offset, x.size, x.stride)

/** `y = alpha · x + y`. */
public fun KoblasEngine.axpy(alpha: Double, x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "axpy")
    axpy(y.values, y.offset, alpha, x.values, x.offset, x.size, y.stride, x.stride)
}

/**
 * `x = alpha · x`.
 *
 * A negatively strided vector is passed with the increment's magnitude. BLAS defines `scal` to return
 * without doing anything for a non-positive increment, so keeping the sign would silently leave the vector
 * unscaled; scaling is order-independent, so walking the same entries forwards gives the same result.
 */
public fun KoblasEngine.scale(alpha: Double, x: DenseVector): Unit = scale(x.values, x.offset, alpha, x.size, x.stride)

/** `y = x`. */
public fun KoblasEngine.copy(x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "copy")
    copy(x.values, x.offset, y.values, y.offset, x.size, x.stride, y.stride)
}

/** Exchanges the entries of [x] and [y]. */
public fun KoblasEngine.swap(x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "swap")
    swap(x.values, x.offset, y.values, y.offset, x.size, x.stride, y.stride)
}

/** Applies the plane rotation given by [c] and [s] to [x] and [y]. */
public fun KoblasEngine.rot(x: DenseVector, y: DenseVector, c: Double, s: Double) {
    requireSameSize(x.size, y.size, "rot")
    rot(x.values, x.offset, y.values, y.offset, x.size, c, s, x.stride, y.stride)
}

// Whole-call hooks are reached only after capability selection, validation and alias staging.

/**
 * Allocates a product using this engine. Two sparse operands produce owned CSC storage;
 * other pairings produce dense storage. Custom [Matrix] operands are read once into dense scratch.
 */
public fun KoblasEngine.gemm(a: Matrix, b: Matrix): Matrix = product(1.0, a, false, b, false)

/** Scaled product with independent transpose flags and the storage rule of [gemm]. */
public fun KoblasEngine.gemm(alpha: Double, a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean): Matrix =
    product(alpha, a, transposeA, b, transposeB)

/**
 * Writes a generic product through this engine into addressed dense storage. Validation precedes all
 * writes; zero alpha and empty depth read no inputs, and zero beta reads no destination values.
 * [workspace] lends logical alias snapshots, sparse scratch and custom operand normalization.
 */
@Suppress("LongParameterList") // BLAS operands, flags and workspace
public fun KoblasEngine.gemm(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    workspace: Workspace? = null,
): Unit = productInto(alpha, a, transposeA, b, transposeB, beta, c, workspace)

/** Fresh dense product of two dense regions using this engine. */
public fun KoblasEngine.gemm(a: DenseMatrixRegion, b: DenseMatrixRegion): DenseMatrix =
    product(1.0, a, false, b, false) as DenseMatrix

/** Fresh dense product using portable CSC traversal and this engine's arithmetic. */
public fun KoblasEngine.gemm(a: SparseMatrix, b: DenseMatrixRegion): DenseMatrix =
    product(1.0, a, false, b, false) as DenseMatrix

/** Fresh CSC product owning its arrays, with stored zeros retained. */
public fun KoblasEngine.gemm(a: SparseMatrix, b: SparseMatrix): SparseMatrix = product(
    1.0,
    a,
    false,
    b,
    false,
) as SparseMatrix

/** TRSV over readable dense, sparse or custom operands, using this engine. */
@Suppress("LongParameterList") // triangle flags and workspace
public fun KoblasEngine.trsv(
    a: Matrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    val sparse = storedCsc(a)
    if (sparse != null) {
        trsv(sparse, x, lower, transpose, unitDiag, workspace)
    } else {
        triangleOperand(
            workspace,
            a,
            lower,
            unitDiag,
        ) { dense -> trsv(dense, x, lower, transpose, unitDiag, workspace) }
    }
}

/** TRMV over readable dense, sparse or custom operands, using this engine. */
@Suppress("LongParameterList") // triangle flags and workspace
public fun KoblasEngine.trmv(
    a: Matrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    val sparse = storedCsc(a)
    if (sparse != null) {
        trmv(sparse, x, lower, transpose, unitDiag, workspace)
    } else {
        triangleOperand(
            workspace,
            a,
            lower,
            unitDiag,
        ) { dense -> trmv(dense, x, lower, transpose, unitDiag, workspace) }
    }
}

/** TRSM over readable dense, sparse or custom operands, using this engine. */
@Suppress("LongParameterList") // triangle flags and workspace
public fun KoblasEngine.trsm(
    a: Matrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
) {
    requireTriangularMatrixOperands(a, b, right, "trsm")
    if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
        com.eignex.koblas.internal.operands.scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
        return
    }
    val sparse = storedCsc(a)
    if (sparse != null) {
        trsm(sparse, b, lower, transpose, unitDiag, right, alpha, workspace)
    } else {
        triangleOperand(
            workspace,
            a,
            lower,
            unitDiag,
        ) { dense -> trsm(dense, b, lower, transpose, unitDiag, right, alpha, workspace) }
    }
}

/** TRMM over readable dense, sparse or custom operands, using this engine. */
@Suppress("LongParameterList") // triangle flags and workspace
public fun KoblasEngine.trmm(
    a: Matrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
) {
    requireTriangularMatrixOperands(a, b, right, "trmm")
    if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
        com.eignex.koblas.internal.operands.scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
        return
    }
    val sparse = storedCsc(a)
    if (sparse != null) {
        trmm(sparse, b, lower, transpose, unitDiag, right, alpha, workspace)
    } else {
        triangleOperand(
            workspace,
            a,
            lower,
            unitDiag,
        ) { dense -> trmm(dense, b, lower, transpose, unitDiag, right, alpha, workspace) }
    }
}

private fun KoblasEngine.wholeCall(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int = 1,
    borrowed: Boolean = false,
): KoblasEngine? = this.wholeCallFor(operation, rows, columns, depth, borrowed)

/** Array convenience for the addressed [gemv] operation. */
@Suppress("LongParameterList") // BLAS operands and workspace
public fun KoblasEngine.gemv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean = false,
    workspace: Workspace? = null,
) {
    requireGemvOperands(a, transpose, x.size, y.size)
    if (y.isEmpty()) return
    if (alpha == 0.0 || (if (transpose) a.rows else a.cols) == 0) {
        applyBeta(this, y, 0, y.size, beta)
        return
    }
    if (a.values === y || x === y || wholeCall(
            if (transpose) EngineImplementation.GemvTransposed else EngineImplementation.Gemv,
            a.rows,
            a.cols,
            borrowed = a !is DenseMatrix,
        ) != null
    ) {
        gemv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), transpose, workspace)
    } else {
        gemvCore(alpha, a, x, 0, 1, beta, y, 0, 1, transpose)
    }
}

/**
 * `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose].
 *
 * [workspace] lends the staging a call takes when [a] or [x] shares [y]. A call that needs no
 * staging takes nothing from it.
 */
@Suppress("LongParameterList") // the BLAS dgemv signature plus the workspace
public fun KoblasEngine.gemv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    transpose: Boolean = false,
    workspace: Workspace? = null,
) {
    requireGemvOperands(a, transpose, x.size, y.size)
    if (y.size == 0) return
    val depth = if (transpose) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        applyBeta(this, y.values, y.offset, y.size, beta, y.stride)
        return
    }
    // Snapshot aliased operands before scaling the destination, and retain them for the call.
    val implementation = wholeCall(
        if (transpose) EngineImplementation.GemvTransposed else EngineImplementation.Gemv,
        a.rows,
        a.cols,
        borrowed = a !is DenseMatrix,
    )
    stagedRegion(workspace, a, y) { stableA ->
        staged(workspace, x, x.overlaps(y)) { stableX ->
            if (implementation != null) {
                implementation.computeGemv(alpha, stableA, transpose, stableX, beta, y)
            } else {
                gemvCore(
                    alpha, stableA, stableX.values, stableX.offset, stableX.stride, beta, y.values,
                    y.offset, y.stride, transpose,
                )
            }
        }
    }
}

private fun KoblasEngine.gemvCore(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    transpose: Boolean,
) {
    val rows = a.rows
    if (transpose) {
        forEachPanel(
            a.cols,
            this.executionGroup(PanelWork.MultiDot, rows, a.cols, xStride == 1),
        ) { start, width ->
            this.multiDot(
                alpha, a.values, a.offset + start * a.leadingDimension, a.leadingDimension, x,
                xOffset, xStride, rows, width, beta, y,
                yOffset + start * yStride, yStride,
            )
        }
    } else {
        applyBeta(this, y, yOffset, rows, beta, yStride)
        forEachPanel(
            a.cols,
            this.executionGroup(PanelWork.ColumnUpdate, rows, a.cols, yStride == 1),
        ) { start, width ->
            this.columnUpdate(
                alpha, a.values, a.offset + start * a.leadingDimension, a.leadingDimension, x,
                xOffset + start * xStride, xStride, rows, width, y, yOffset, yStride,
            )
        }
    }
}

/**
 * Fresh transposed [a]. For a product prefer the transpose flags on [gemv] and [gemm], which avoid a
 * caller-visible intermediate; this is for a caller that means to hold the transpose.
 *
 * A storage transform rather than arithmetic, so it stays a Kotlin loop: the standard has no entry point
 * for it, and the BLAS-like `omatcopy` extension is not one every supported library exports.
 */
public fun KoblasEngine.transpose(a: DenseMatrixRegion): DenseMatrix {
    val result = DenseMatrix.zero(a.cols, a.rows)
    for (j in 0 until a.cols) {
        for (i in 0 until a.rows) {
            result.values[j + i * a.cols] =
                a.values[a.offset + i + j * a.leadingDimension]
        }
    }
    return result
}

/**
 * `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
 *
 * An input that shares [c] is snapshotted before dispatch.
 * [workspace] lends that snapshot, packed panels for a blocked product, an accumulating column for an
 * unpacked product, and a coefficient column for a gathered strided reduction. A zero [alpha] or an empty
 * shared dimension scales [c] and reads no operand; a zero [beta] overwrites [c] without reading it.
 *
 * Scaling placement and accumulation order depend on the selected implementation. This includes
 * whether [alpha] scales operand entries or partial sums. Rounding, overflow and non-finite results
 * may therefore differ between implementations and transpose modes.
 */
@Suppress("LongParameterList") // the BLAS dgemm signature
public fun KoblasEngine.gemm(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    workspace: Workspace? = null,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    // Return before borrowing operand-sized scratch for an empty destination.
    if (c.rows == 0 || c.cols == 0) return
    val depth = if (transposeA) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta)
        return
    }
    val implementation = wholeCall(
        EngineImplementation.Gemm,
        c.rows,
        c.cols,
        depth,
        a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
    )
    stagedRegion(workspace, a, c) { sa ->
        stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
            if (implementation != null) {
                implementation.computeGemm(alpha, sa, transposeA, sb, transposeB, beta, c)
                return@stagedRegion
            }
            gemmCore(alpha, sa, transposeA, sb, transposeB, beta, c, depth, workspace)
        }
    }
}

/**
 * The product itself, over operands already staged against an overlap with the destination.
 *
 * Which path runs is the extents' answer rather than the engine's. A destination with one column or one
 * row is a matrix-vector product, so it runs as [gemv] arithmetic over the operand that is a matrix: the
 * product schedule would otherwise make a separate panel call for every destination column, which on a
 * single row is one call per entry. A single entry is a dot product either way and stays on the schedule,
 * which measured slightly ahead of the matrix-vector form for a long shared dimension.
 */
private fun KoblasEngine.gemmCore(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    depth: Int,
    workspace: Workspace?,
) {
    if (c.cols == 1 && c.rows > 1) {
        // The column of op(B) is a stored column, or a stored row when B is transposed.
        gemvCore(
            alpha, a, b.values, b.offset, if (transposeB) b.leadingDimension else 1,
            beta, c.values, c.offset, 1, transposeA,
        )
        return
    }
    if (c.rows == 1 && c.cols > 1) {
        // The destination row is op(B)ᵀ times the row of op(A), a stored column when A is transposed.
        gemvCore(
            alpha, b, a.values, a.offset, if (transposeA) 1 else a.leadingDimension,
            beta, c.values, c.offset, c.leadingDimension, !transposeB,
        )
        return
    }
    productWindow(
        this, alpha, a.values, a.offset, a.leadingDimension, transposeA, b.values, b.offset, b.leadingDimension, transposeB, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth, OutputTriangle.Full, workspace,
    )
}

/**
 * `C = alpha · op(A) · op(B) + beta · C` in only the selected triangle (Netlib `GEMMTR`, commonly exposed
 * as `gemmt`). `op(A)` is `n×k`, `op(B)` is `k×n`, and [c] is `n×n`. The opposite triangle is neither read
 * nor written. Input overlap with [c] is snapshotted before dispatch.
 *
 * Missing host exports retain the shared selected-triangle product schedule.
 */
@Suppress("LongParameterList") // the BLAS gemmt signature
public fun KoblasEngine.gemmt(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    requireGemmtOperands(a, transposeA, b, transposeB, c)
    // An empty destination needs no staging, even when operands share an empty array.
    if (c.rows == 0) return
    val depth = if (transposeA) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    val implementation = wholeCall(
        EngineImplementation.Gemmt,
        c.rows,
        c.cols,
        depth,
        a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
    )
    stagedRegion(workspace, a, c) { sa ->
        stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
            if (implementation != null) {
                implementation.computeGemmt(
                    alpha,
                    sa,
                    transposeA,
                    sb,
                    transposeB,
                    beta,
                    c,
                    symmetricStructure(lower),
                )
                return@stagedRegion
            }
            productWindow(
                this, alpha, sa.values, sa.offset, sa.leadingDimension, transposeA, sb.values, sb.offset, sb.leadingDimension, transposeB, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
                selectedTriangle(
                    lower,
                ),
                workspace,
            )
        }
    }
}

/** Array convenience for the addressed [symv] operation. */
@Suppress("LongParameterList") // BLAS operands and workspace
public fun KoblasEngine.symv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    requireSymvOperands(a, x.size, y.size)
    if (y.isEmpty()) return
    if (alpha == 0.0 || (a.rows) == 0) {
        applyBeta(this, y, 0, y.size, beta)
        return
    }
    if (a.values === y || x === y ||
        wholeCall(EngineImplementation.Symv, a.rows, a.cols, borrowed = a !is DenseMatrix) != null
    ) {
        symv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), lower, workspace)
    } else {
        symvCore(alpha, a, x, 0, 1, beta, y, 0, 1, lower)
    }
}

/** `y = alpha · A · x + beta · y` for a symmetric [a] (BLAS `dsymv`). Only the [lower] triangle is read,
 *  diagonal included. [workspace] lends the staging an overlap with [y] takes, as in [gemv]. */
@Suppress("LongParameterList") // the BLAS dsymv signature plus the workspace
public fun KoblasEngine.symv(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    requireSymvOperands(a, x.size, y.size)
    val n = a.rows
    if (alpha == 0.0 || n == 0) {
        applyBeta(this, y.values, y.offset, y.size, beta, y.stride)
        return
    }
    // Snapshot aliased operands before beta overwrites values still needed by the product.
    val implementation = wholeCall(EngineImplementation.Symv, a.rows, a.cols, 1, a !is DenseMatrix)
    stagedRegion(workspace, a, y, lower = lower) { sa ->
        staged(workspace, x, x.overlaps(y)) { xv ->
            if (implementation != null) {
                implementation.computeSymv(alpha, sa, symmetricStructure(lower), xv, beta, y)
            } else {
                symvCore(alpha, sa, xv.values, xv.offset, xv.stride, beta, y.values, y.offset, y.stride, lower)
            }
        }
    }
}

@Suppress("LongParameterList") // the operands, their snapshots and the BLAS dsymv scalars
private fun KoblasEngine.symvCore(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    lower: Boolean,
) {
    val n = a.rows
    applyBeta(this, y, yOffset, a.rows, beta, yStride)
    val group = this.executionGroup(PanelWork.CoupledDotUpdate, n, n, xStride == 1 && yStride == 1)
    forEachTrianglePanel(n, group, lower, fromDiagonal = false) { start, width, window, rows ->
        symmetricCorner(alpha, a, x, xOffset, xStride, y, yOffset, yStride, start, width, lower)
        if (rows > 0) {
            this.coupledUpdateDot(
                alpha, a.values, a.offset + window + start * a.leadingDimension, a.leadingDimension, x,
                xOffset + window * xStride, rows, width, y, yOffset + window * yStride, x,
                xOffset + start * xStride, y, yOffset + start * yStride, xStride, yStride,
                xStride, yStride,
            )
        }
    }
}

/**
 * The square block where a group's own columns and rows meet, which the shared window cannot reach.
 *
 * Every stored position in it contributes twice, once down its column and once along the mirrored row,
 * except the diagonal which is its own mirror. The arithmetic is the panel's, written out: a coefficient
 * scaled by alpha multiplies down, and the reduction is scaled by alpha on its way into the destination.
 */
private fun KoblasEngine.symmetricCorner(
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    start: Int,
    width: Int,
    lower: Boolean,
) {
    val lda = a.leadingDimension
    for (c in start until start + width) {
        val coefficient = alpha * x[xOffset + c * xStride]
        val diagonal = a.values[a.offset + c + c * lda]
        var mirrored = 0.0
        // Only the stored positions of this column, so no test per entry; the mirrored row sums once.
        val first = if (lower) c + 1 else start
        val last = if (lower) start + width else c
        for (i in first until last) {
            val value = a.values[a.offset + i + c * lda]
            y[yOffset + i * yStride] += coefficient * value
            mirrored += value * x[xOffset + i * xStride]
        }
        y[yOffset + c * yStride] += coefficient * diagonal + alpha * mirrored
    }
}

/** `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
 *  [lower] triangle of [a] is read; input overlap with [c] is snapshotted before dispatch. */
@Suppress("LongParameterList") // the BLAS dsymm signature
public fun KoblasEngine.symm(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean = true,
    right: Boolean = false,
    workspace: Workspace? = null,
) {
    requireSymmOperands(a, b, c, right)
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    val implementation = wholeCall(
        EngineImplementation.Symm,
        c.rows,
        c.cols,
        a.rows,
        a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
    )
    stagedRegion(workspace, a, c, lower = if (sameRegion(a, b)) null else lower) { sa ->
        stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
            if (implementation != null) {
                implementation.computeSymm(alpha, sa, symmetricStructure(lower), sb, beta, c, rightSide = right)
                return@stagedRegion
            }
            // Scale once because multiple symmetric windows accumulate into each destination entry.
            scaleRegion(c, beta, vectors = this)
            symmetricProduct(
                this, alpha, sa.values, a.rows, lower, sb.values, sb.leadingDimension, c.values, c.leadingDimension, c.rows, c.cols, right, workspace, sa.offset, sa.leadingDimension, sb.offset, c.offset,
            )
        }
    }
}

/**
 * `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`).
 * Only the [lower] or upper triangle is written. Overlap with [c] is snapshotted before dispatch.
 *
 * One product, so [alpha] and the accumulation order carry the latitude [gemm] describes.
 */
@Suppress("LongParameterList") // the BLAS dsyrk signature
public fun KoblasEngine.syrk(
    alpha: Double,
    a: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    requireSyrkOperands(a, transpose, c)
    if (c.rows == 0) return
    val depth = if (transpose) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    val implementation = wholeCall(
        EngineImplementation.Syrk,
        c.rows,
        c.cols,
        depth,
        a !is DenseMatrix || c !is DenseMatrix,
    )
    stagedRegion(workspace, a, c) { sa ->
        if (implementation != null) {
            implementation.computeSyrk(alpha, sa, transpose, beta, c, symmetricStructure(lower))
            return@stagedRegion
        }
        productWindow(
            this, alpha, sa.values, sa.offset, sa.leadingDimension, transpose, sa.values, sa.offset, sa.leadingDimension, !transpose, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
            selectedTriangle(
                lower,
            ),
            workspace,
        )
    }
}

/**
 * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
 * [transpose]. Writes only the [lower] or upper triangle and snapshots input overlap before dispatch.
 *
 * The implementation may accumulate the two products separately or in one traversal, with the
 * rounding and overflow differences described by [gemm].
 */
@Suppress("LongParameterList") // the BLAS dsyr2k signature
public fun KoblasEngine.syr2k(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    requireSyr2kOperands(a, b, transpose, c)
    productTriangle(alpha, a, b, transpose, beta, c, lower, workspace)
}

/**
 * `alpha · op(A) · op(B)ᵀ + beta · C` in one triangle, and the same with the operands swapped.
 *
 * A rank-k update is a product whose right operand is the left one transposed, and a rank-2k update is
 * the sum of the two products its definition names. Composing rather than fusing is deliberate: both
 * halves are ordinary windows of shared product scheduling, so each is packed, blocked and tiled like
 * any other product, while a fused traversal would need a second implementation of all of that to beat
 * them together. The local evidence compares the two.
 *
 * `beta` is carried by the first of the two and the second accumulates, which is how it reaches every
 * selected entry exactly once.
 */
private fun KoblasEngine.productTriangle(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    workspace: Workspace?,
) {
    if (c.rows == 0) return
    val depth = if (transpose) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    val selected = selectedTriangle(lower)
    val implementation = wholeCall(
        EngineImplementation.Syr2k,
        c.rows,
        c.cols,
        depth,
        a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
    )
    stagedRegion(workspace, a, c) { sa ->
        stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
            if (implementation != null) {
                implementation.computeSyr2k(alpha, sa, sb, transpose, beta, c, symmetricStructure(lower))
                return@stagedRegion
            }
            productWindow(
                this, alpha, sa.values, sa.offset, sa.leadingDimension, transpose, sb.values, sb.offset, sb.leadingDimension, !transpose, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth, selected, workspace,
            )
            productWindow(
                this, alpha, sb.values, sb.offset, sb.leadingDimension, transpose, sa.values, sa.offset, sa.leadingDimension, !transpose, 1.0, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth, selected, workspace,
            )
        }
    }
}

/** `A = A + alpha · x · yᵀ` (BLAS `dger`). A vector sharing [a]'s buffer is snapshotted, and
 *  [workspace] lends that snapshot. */
public fun KoblasEngine.ger(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrixRegion,
    workspace: Workspace? = null,
) {
    requireGerOperands(x.size, y.size, a)
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    val implementation = wholeCall(EngineImplementation.Ger, a.rows, a.cols, 1, a !is DenseMatrix)
    stagedVectors(workspace, x, y, a) { sx, sy ->
        if (implementation != null) {
            implementation.computeGer(alpha, sx, sy, a)
            return@stagedVectors
        }
        gerCore(
            alpha, sx.values, sx.offset, sx.stride, sy.values, sy.offset, sy.stride,
            a, a.offset, a.rows, a.cols,
        )
    }
}

private fun KoblasEngine.gerCore(
    alpha: Double,
    sx: DoubleArray,
    sxOffset: Int,
    sxStride: Int,
    sy: DoubleArray,
    syOffset: Int,
    syStride: Int,
    a: DenseMatrixRegion,
    aOffset: Int,
    rows: Int,
    columns: Int,
) {
    forEachPanel(
        columns,
        this.executionGroup(PanelWork.RankUpdate, rows, columns, sxStride == 1),
    ) { start, width ->
        this.rankUpdate(
            alpha, a.values, aOffset + start * a.leadingDimension, a.leadingDimension,
            sx, sxOffset, sxStride, rows, width,
            sy, syOffset + start * syStride, syStride,
        )
    }
}

/** Array convenience for the addressed [ger] operation. */
@Suppress("LongParameterList") // BLAS operands and workspace
public fun KoblasEngine.ger(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DenseMatrixRegion,
    workspace: Workspace? = null,
) {
    requireGerOperands(x.size, y.size, a)
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    if (x === a.values || y === a.values ||
        wholeCall(EngineImplementation.Ger, a.rows, a.cols, borrowed = a !is DenseMatrix) != null
    ) {
        ger(alpha, DenseVector.wrap(x), DenseVector.wrap(y), a, workspace)
    } else {
        gerCore(alpha, x, 0, 1, y, 0, 1, a, a.offset, a.rows, a.cols)
    }
}

/**
 * `A(rowOffset + i, columnOffset + j) += alpha · x(xOffset + i) · y(yOffset + j)` for
 * `0 <= i < rows` and `0 <= j < columns`. Entries outside this rectangle are untouched.
 *
 * The destination retains [a]'s column-major leading dimension. Built-in panel kernels update it
 * directly, without copying the rectangle or allocating vector views. All matrix and vector ranges
 * are validated before writing, including when [alpha] is zero or either extent is empty.
 *
 * A vector sharing [a]'s buffer is snapshotted over only its selected range; [workspace] lends that
 * snapshot. A zero [alpha] or an empty rectangle reads no entries and takes no scratch.
 */
@Suppress("LongParameterList") // the rank update and its three independently addressed windows
public fun KoblasEngine.ger(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DenseMatrixRegion,
    rowOffset: Int,
    columnOffset: Int,
    rows: Int,
    columns: Int,
    xOffset: Int = 0,
    yOffset: Int = 0,
    workspace: Workspace? = null,
) {
    requireGerWindowOperands(x.size, y.size, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset)
    if (alpha == 0.0 || rows == 0 || columns == 0) return
    if (x === a.values || y === a.values) {
        val window = a.view(rowOffset, columnOffset, rows, columns)
        stagedVectors(
            workspace,
            StridedVector(x, xOffset, rows, 1),
            StridedVector(y, yOffset, columns, 1),
            window,
        ) { sx, sy ->
            gerCore(
                alpha, sx.values, sx.offset, sx.stride, sy.values, sy.offset, sy.stride,
                window, window.offset, rows, columns,
            )
        }
    } else {
        gerCore(
            alpha, x, xOffset, 1, y, yOffset, 1, a,
            a.offset + rowOffset + columnOffset * a.leadingDimension, rows, columns,
        )
    }
}

/** `A += alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. [x] must be dense or
 *  strided storage, which is what a vendor can address. Overlap staging follows [ger]. */
@Suppress("LongParameterList") // the BLAS dsyr signature plus the workspace
public fun KoblasEngine.syr(
    alpha: Double,
    x: DenseVector,
    a: DenseMatrixRegion,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    requireSyrOperands(a, x.size, "syr")
    val n = a.rows
    if (alpha == 0.0 || n == 0) return
    // Snapshot an aliased vector before updates overwrite coefficients needed by later columns.
    val implementation = wholeCall(EngineImplementation.Syr, n, n, 1, a !is DenseMatrix)
    staged(workspace, x, regionVectorOverlap(a, x)) { sx ->
        if (implementation != null) {
            implementation.computeSyr(alpha, sx, a, symmetricStructure(lower))
            return@staged
        }
        val xv = sx.values
        val origin = sx.offset
        val step = sx.stride
        val group = this.executionGroup(PanelWork.RankUpdate, n, n, sx.stride == 1)
        forEachTrianglePanel(n, group, lower, fromDiagonal = true) { start, width, window, rows ->
            // Only the diagonal corner has differing stored rows; the remaining window is shared by the group.
            val cornerFirst = if (lower) start else start + 1
            val cornerLast = if (lower) start + width - 1 else start + width
            for (c in start until start + width) {
                val coefficient = alpha * xv[origin + c * step]
                for (i in cornerFirst until cornerLast) {
                    if (if (lower) i < c else i > c) continue
                    a.values[a.offset + i + c * a.leadingDimension] += coefficient * xv[origin + i * step]
                }
            }
            this.rankUpdate(
                alpha, a.values, a.offset + window + start * a.leadingDimension, a.leadingDimension, xv,
                origin + window * step, step, rows, width,
                xv, origin + start * step, step,
            )
        }
    }
}

/** `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. Operand
 *  storage and overlap staging follow [syr]. */
@Suppress("LongParameterList") // the BLAS dsyr2 signature plus the workspace
public fun KoblasEngine.syr2(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrixRegion,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    requireSyr2Operands(a, x.size, y.size, "syr2")
    val n = a.rows
    if (alpha == 0.0 || n == 0) return
    val implementation = wholeCall(EngineImplementation.Syr2, n, n, 1, a !is DenseMatrix)
    stagedVectors(workspace, x, y, a) { sx, sy ->
        if (implementation != null) {
            implementation.computeSyr2(alpha, sx, sy, a, symmetricStructure(lower))
            return@stagedVectors
        }
        syr2Core(alpha, sx, sy, a, lower)
    }
}

private fun KoblasEngine.syr2Core(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion, lower: Boolean) {
    val n = a.rows
    val xv = x.values
    val yv = y.values
    val xOrigin = x.offset
    val xStep = x.stride
    val yOrigin = y.offset
    val yStep = y.stride
    val lda = a.leadingDimension
    val group = this.executionGroup(PanelWork.RankUpdate, n, n, xStep == 1 && yStep == 1)
    forEachTrianglePanel(n, group, lower, fromDiagonal = true) { start, width, window, rows ->
        // Only the diagonal corner has differing stored rows; the remaining window is shared by the group.
        val cornerFirst = if (lower) start else start + 1
        val cornerLast = if (lower) start + width - 1 else start + width
        for (c in start until start + width) {
            val xc = xv[xOrigin + c * xStep]
            val yc = yv[yOrigin + c * yStep]
            for (i in cornerFirst until cornerLast) {
                if (if (lower) i < c else i > c) continue
                a.values[a.offset + i + c * lda] +=
                    alpha * (xv[xOrigin + i * xStep] * yc + yv[yOrigin + i * yStep] * xc)
            }
        }
        this.rankTwoUpdate(
            alpha, a.values, a.offset + window + start * lda, lda,
            xv, xOrigin + window * xStep, xStep, yv, yOrigin + window * yStep, yStep, rows, width,
            yv, yOrigin + start * yStep, yStep, xv, xOrigin + start * xStep, xStep,
        )
    }
}

/** Array convenience for the addressed [trsv] operation. */
@Suppress("LongParameterList") // BLAS operands and workspace
public fun KoblasEngine.trsv(
    a: DenseMatrixRegion,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    if (a.rows == 0) return
    if (a.values === x || wholeCall(
            if (transpose) EngineImplementation.TrsvTransposed else EngineImplementation.Trsv,
            a.rows,
            a.cols,
            borrowed = a !is DenseMatrix,
        ) != null
    ) {
        trsv(a, DenseVector.wrap(x), lower, transpose, unitDiag, workspace)
    } else {
        trsvCore(a, x, 0, 1, lower, transpose, unitDiag)
    }
}

/**
 * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
 * `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out.
 *
 * As in BLAS `dtrsv`, no singularity or conditioning check is performed. Singular inputs have no
 * guaranteed numerical result.
 *
 * The substitution overwrites [x] as it goes, so a triangle sharing that buffer is snapshotted first and
 * [workspace] lends the snapshot.
 */
@Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace
public fun KoblasEngine.trsv(
    a: DenseMatrixRegion,
    x: DenseVector,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    val n = a.rows
    if (n == 0) return
    val implementation = wholeCall(
        if (transpose) EngineImplementation.TrsvTransposed else EngineImplementation.Trsv,
        n,
        n,
        borrowed = a !is DenseMatrix,
    )
    stagedRegion(workspace, a, x, lower = lower, unitDiag = unitDiag) { sa ->
        if (implementation != null) {
            implementation.computeTrsv(sa, triangle(lower, unitDiag), transpose, x)
            return@stagedRegion
        }
        trsvCore(sa, x.values, x.offset, x.stride, lower, transpose, unitDiag)
    }
}

/**
 * Solve `op(T) · x = b` in place, one stored column at a time.
 *
 * Untransposed, a finished entry is divided by its diagonal and then removed from every entry that
 * depends on it, which is one column update. Transposed, an entry is finished by reducing the column
 * against the entries already done, which is one multi-dot. Either way the dependency order is the
 * substitution's and cannot be grouped away; what a panel covers is the independent arithmetic within
 * one step.
 */
private fun KoblasEngine.trsvCore(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val n = a.rows
    val lda = a.leadingDimension
    val ascending = lower != transpose
    forEachTriangularBlock(n, ascending) { start, end ->
        // The rest of the stored triangle beside this block: below it when lower, above it when upper.
        val outside = if (lower) end else 0
        val outsideRows = if (lower) n - end else start
        val block = a.offset + outside + start * lda
        if (transpose && outsideRows > 0) {
            this.multiDot(
                -1.0, a.values, block, lda, x, xOffset + outside * xStride, xStride, outsideRows, end - start,
                1.0, x, xOffset + start * xStride, xStride,
            )
        }
        this.diagonalSolve(
            a.values, lda, start, end - start, transpose, lower != transpose, unitDiag,
            x, xOffset + start * xStride, xStride, 1, 1, a.offset,
        )
        if (!transpose && outsideRows > 0) {
            this.columnUpdate(
                -1.0, a.values, block, lda, x, xOffset + start * xStride, xStride, outsideRows, end - start,
                x, xOffset + outside * xStride, xStride,
            )
        }
    }
}

/** Array convenience for the addressed [trmv] operation. */
@Suppress("LongParameterList") // BLAS operands and workspace
public fun KoblasEngine.trmv(
    a: DenseMatrixRegion,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    if (a.rows == 0) return
    if (a.values === x || wholeCall(
            if (transpose) EngineImplementation.TrmvTransposed else EngineImplementation.Trmv,
            a.rows,
            a.cols,
            borrowed = a !is DenseMatrix,
        ) != null
    ) {
        trmv(a, DenseVector.wrap(x), lower, transpose, unitDiag, workspace)
    } else {
        trmvCore(a, x, 0, 1, lower, transpose, unitDiag)
    }
}

/** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv], whose overlap staging and
 *  [workspace] it shares. */
@Suppress("LongParameterList") // the BLAS dtrmv signature plus the workspace
public fun KoblasEngine.trmv(
    a: DenseMatrixRegion,
    x: DenseVector,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    val n = a.rows
    val implementation = wholeCall(
        if (transpose) EngineImplementation.TrmvTransposed else EngineImplementation.Trmv,
        n,
        n,
        borrowed = a !is DenseMatrix,
    )
    stagedRegion(workspace, a, x, lower = lower, unitDiag = unitDiag) { sa ->
        if (implementation != null) {
            implementation.computeTrmv(sa, triangle(lower, unitDiag), transpose, x)
            return@stagedRegion
        }
        trmvCore(sa, x.values, x.offset, x.stride, lower, transpose, unitDiag)
    }
}

private fun KoblasEngine.trmvCore(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val n = a.rows
    val lda = a.leadingDimension
    // Traverse in opposite directions for transpose variants to preserve values until their last use.
    val ascending = transpose == lower
    forEachTriangularBlock(n, ascending) { start, end ->
        // The rest of the stored triangle beside this block: below it when lower, above it when upper.
        val outside = if (lower) end else 0
        val outsideRows = if (lower) n - end else start
        val block = a.offset + outside + start * lda
        if (!transpose && outsideRows > 0) {
            this.columnUpdate(
                1.0, a.values, block, lda, x, xOffset + start * xStride, xStride, outsideRows, end - start,
                x, xOffset + outside * xStride, xStride,
            )
        }
        this.diagonalMultiply(
            a.values, lda, start, end - start, transpose, lower != transpose, unitDiag,
            x, xOffset + start * xStride, xStride, 1, 1, a.offset,
        )
        if (transpose && outsideRows > 0) {
            this.multiDot(
                1.0, a.values, block, lda, x, xOffset + outside * xStride, xStride, outsideRows, end - start,
                1.0, x, xOffset + start * xStride, xStride,
            )
        }
    }
}

/** `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`). Flags
 *  follow [trsv]; the right-hand sides are the columns of [b] from the left and its rows from the right. */
@Suppress("LongParameterList") // the BLAS dtrsm signature
public fun KoblasEngine.trsm(
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
): Unit = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true, workspace = workspace)

/** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of
 *  [trsm]. */
@Suppress("LongParameterList") // the BLAS dtrmm signature
public fun KoblasEngine.trmm(
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
): Unit = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false, workspace = workspace)

/**
 * Both triangular matrix routines, over a triangle staged against an overlap with the block it works on.
 *
 * The staging comes first and the scaling second. `alpha` is spent on the right-hand sides before the
 * substitution, and where the triangle shares their buffer that write would otherwise reach the
 * coefficients the substitution is about to read.
 */
@Suppress("LongParameterList") // the BLAS dtrsm signature, which of the two it is, and the workspace
private fun KoblasEngine.triangularMatrix(
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    solve: Boolean,
    workspace: Workspace?,
) {
    val what = if (solve) "trsm" else "trmm"
    requireTriangularMatrixOperands(a, b, right, what)
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    val sides = if (right) b.rows else b.cols
    val order = a.rows
    if (sides == 0 || order == 0) return
    val implementation = wholeCall(
        if (solve) EngineImplementation.Trsm else EngineImplementation.Trmm,
        b.rows,
        b.cols,
        order,
        a !is DenseMatrix || b !is DenseMatrix,
    )
    stagedRegion(workspace, a, b, lower = lower, unitDiag = unitDiag) { sa ->
        if (implementation != null) {
            val structure = triangle(lower, unitDiag)
            if (solve) {
                implementation.computeTrsm(alpha, sa, structure, transpose, b, rightSide = right)
            } else {
                implementation.computeTrmm(alpha, sa, structure, transpose, b, rightSide = right)
            }
            return@stagedRegion
        }
        triangularMatrix(
            this, sa.values, order, b.values, b.leadingDimension, sides, lower, transpose, unitDiag, right, alpha, solve, workspace, sa.offset, sa.leadingDimension, b.offset,
        )
    }
}

/** [gemv] with `alpha = 1, beta = 0`, into a fresh result. */
public fun KoblasEngine.gemv(a: DenseMatrixRegion, x: DoubleArray, transpose: Boolean = false): DoubleArray {
    val size = if (transpose) a.cols else a.rows
    requireGemvOperands(a, transpose, x.size, size)
    val y = DoubleArray(size)
    gemv(1.0, a, x, 0.0, y, transpose)
    return y
}

/**
 * In-place `y = alpha · op(A) · x + beta · y`, where `op(A)` is `Aᵀ` when [transpose]. Per BLAS
 * convention `beta == 0.0` overwrites [y] without reading it, so it may arrive uninitialized, and
 * `alpha == 0.0` reads neither [a] nor [x].
 *
 * Only stored entries are visited, so an implicit zero of [a] forms no product: a column [a] does not
 * store leaves [y] untouched even where [x] holds an infinity. A stored zero does form its product.
 *
 * [x] and [a]'s coefficients may share [y]'s buffer and are snapshotted before [y] is scaled, with
 * [workspace] lending the snapshot. A structural array is never a destination, so an aliased sparse
 * operand shares it and only its coefficients are copied.
 */
@Suppress("LongParameterList") // the BLAS dgemv signature plus the workspace
public fun KoblasEngine.gemv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean = false,
    workspace: Workspace? = null,
) {
    requireGemvOperands(a, transpose, x.size, y.size)
    // Zero alpha or zero extent reads neither operand; only destination scaling remains.
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
        applyBeta(this, y, 0, y.size, beta)
        return
    }
    // Snapshot aliased operands before destination scaling overwrites their coefficients.
    staged(workspace, x, x === y) { stableX ->
        staged(workspace, a, a.values === y) { stableA ->
            gemvCore(alpha, stableA, stableX, beta, y, transpose)
        }
    }
}

@Suppress("LongParameterList") // the BLAS dgemv signature, over the snapshots taken for it
private fun KoblasEngine.gemvCore(
    alpha: Double,
    stableA: SparseMatrix,
    stableX: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean,
) {
    applyBeta(this, y, 0, y.size, beta)
    if (transpose) {
        for (j in 0 until stableA.cols) {
            val start = stableA.colPointers[j]
            val end = stableA.colPointers[j + 1]
            if (start == end) continue
            val sum = this.dotDense(
                stableA.rowIndices,
                start,
                stableA.values,
                start,
                end - start,
                stableX,
            )
            y[j] += alpha * sum
        }
    } else {
        for (j in 0 until stableA.cols) {
            val xj = alpha * stableX[j]
            val start = stableA.colPointers[j]
            this.axpy(
                stableA.rowIndices,
                start,
                stableA.values,
                start,
                stableA.colPointers[j + 1] - start,
                xj,
                y,
            )
        }
    }
}

/**
 * Fresh transposed [a], still CSC, which makes this the CSC-to-CSR conversion as well. Explicitly stored
 * zeros survive, since the transpose is structural rather than arithmetic.
 *
 * On the seam rather than beside the other whole-matrix operations because the two directions of [gemv]
 * leave a caller with no reason to materialize a transpose unless it means to hold it.
 */
public fun KoblasEngine.transpose(a: SparseMatrix): SparseMatrix = transposeCsc(a)

/**
 * `y = alpha · A · x + beta · y` for a square sparse [a] interpreted as symmetric. Exactly the selected
 * [lower] or upper stored triangle is read; the other triangle and implicit zeros are ignored. A stored
 * off-diagonal entry contributes to both mirrored positions. `alpha == 0.0` reads neither [a] nor [x],
 * and `beta == 0.0` overwrites [y] without reading it. Aliasing [x] with [y], or [a]'s values with [y],
 * is supported through the staging [gemv] describes, which [workspace] lends.
 */
@Suppress("LongParameterList")
public fun KoblasEngine.symv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    requireSymvOperands(a, x.size, y.size)
    if (alpha == 0.0) {
        applyBeta(this, y, 0, y.size, beta)
        return
    }
    staged(workspace, x, x === y) { stableX ->
        staged(workspace, a, a.values === y) { stableA ->
            applyBeta(this, y, 0, y.size, beta)
            for (column in 0 until stableA.cols) {
                this.sparseSymmetricVectorColumn(
                    alpha,
                    column,
                    stableA.rowIndices,
                    stableA.values,
                    stableA.colPointers[column],
                    stableA.colPointers[column + 1],
                    stableX,
                    y,
                    lower,
                )
            }
        }
    }
}

/**
 * `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right], interpreting square
 * sparse [a] as symmetric from exactly its selected [lower] or upper triangle. `alpha == 0.0` reads
 * neither input; `beta == 0.0` does not read [c]. Shared input and destination buffers are staged, with
 * [workspace] reused when supplied.
 *
 * Which operand [alpha] is folded into, and the order the terms of one entry are summed in, are the
 * implementation's, as in every product here.
 */
@Suppress("LongParameterList")
public fun KoblasEngine.symm(
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean = true,
    right: Boolean = false,
    workspace: Workspace? = null,
) {
    requireSymmOperands(a, b, c, right)
    // Return before staging or loans: an empty axis may be paired with an enormous one.
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    staged(workspace, a, a.values === c.values) { stableA ->
        stagedRegion(workspace, b, c) { stableB ->
            scaleRegion(c, beta, vectors = this)
            if (right) {
                for (column in 0 until stableA.cols) {
                    this.sparseSymmetricRightColumn(
                        alpha,
                        column,
                        stableA.rowIndices,
                        stableA.values,
                        stableA.colPointers[column],
                        stableA.colPointers[column + 1],
                        stableB.values,
                        c.values,
                        stableB.rows,
                        lower, stableB.offset, stableB.leadingDimension, c.offset, c.leadingDimension,
                    )
                }
            } else {
                multiplySymmetricFromTheLeft(this, alpha, stableA, stableB, c, lower, workspace)
            }
        }
    }
}

/**
 * Solve `op(T) · x = b` in place, `op` transposing when [transpose]. [x] holds the right-hand side on
 * entry and the solution on return. Only the [lower] or upper triangle of [a] is read, and [unitDiag]
 * takes the diagonal as 1 without reading it, as the dense [com.eignex.koblas.KoblasEngine.trsv] does.
 *
 * A missing diagonal entry represents zero. As in dense `dtrsv`, no singularity check is performed;
 * singular inputs have no guaranteed numerical result.
 * A right-hand side that is already exactly zero contributes no update, which is what keeps a column of
 * zeros from forming `0 · infinity` against a stored coefficient.
 *
 * The substitution overwrites [x] as it goes, so coefficients sharing that buffer are snapshotted before
 * the first write and [workspace] lends the snapshot.
 */
@Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace
public fun KoblasEngine.trsv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    // Snapshot an aliased triangle before substitution overwrites coefficients needed by later columns.
    staged(workspace, a, a.values === x) { stable ->
        trsvCore(this, stable, x, lower, transpose, unitDiag)
    }
}

/**
 * Multiply `x = op(T) · x` in place (BLAS `dtrmv`), where `op` transposes when [transpose]. Only the
 * [lower] or upper triangle is read; [unitDiag] treats its diagonal as one without reading it.
 *
 * Unlike [trsv], this performs no division and therefore has no singularity behavior: a missing diagonal
 * is simply zero. The destination may share [a]'s value buffer, staged as in [trsv].
 */
@Suppress("LongParameterList") // the BLAS dtrmv signature plus the workspace
public fun KoblasEngine.trmv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    staged(workspace, a, a.values === x) { stable ->
        trmvCore(this, stable, x, lower, transpose, unitDiag)
    }
}

/**
 * `C = alpha · op(A) · op(B) + beta · C` (Sparse BLAS `usmm`) for a sparse [a] against a dense [b] and
 * [c], or `C = alpha · op(B) · op(A) + beta · C` when [right]. `beta == 0.0` overwrites [c] without
 * reading it, as [gemv] does.
 *
 * A sparse matrix times a dense one stays dense and fills in nowhere, which is what puts this on the
 * sparse half at all: it is [gemv] over the columns of the dense operand.
 *
 * [right] is here for the reason `dtrsm` has a side and `dgemm` does not. `dgemm` needs none because its
 * operands are the same kind and a caller picks which is which; here [a] is distinguished by being the
 * sparse one and cannot be swapped, so without a side the product of a dense matrix by a sparse one is
 * only reachable by transposing the result. [workspace] reuses dense staging and transpose scratch.
 */
@Suppress("LongParameterList") // the BLAS dgemm signature, plus the side the sparse operand sits on
public fun KoblasEngine.gemm(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    right: Boolean = false,
    workspace: Workspace? = null,
) {
    // Right-side multiplication uses the same derivation with operands reversed.
    if (right) {
        requireGemmOperands(b, transposeB, a, transposeA, c)
    } else {
        requireGemmOperands(a, transposeA, b, transposeB, c)
    }
    val m = c.rows
    val n = c.cols
    val k = if (right) {
        if (transposeB) b.rows else b.cols
    } else {
        if (transposeA) a.rows else a.cols
    }
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    staged(workspace, a, a.values === c.values) { stableA ->
        stagedRegion(workspace, b, c) { stableB ->
            scaleRegion(c, beta, vectors = this)
            if (right) {
                multiplyFromTheRight(
                    this, alpha, stableA, transposeA, stableB, transposeB, c, m, workspace,
                )
            } else {
                multiplyFromTheLeft(
                    this, alpha, stableA, transposeA, stableB, transposeB, c, m, n, k, workspace,
                )
            }
        }
    }
}

/**
 * Fresh CSC `alpha · op(A) · op(B)`. Product support is discovered from stored positions, including
 * explicit zeros, retained cancellations, and underflowed values; rows are sorted within every column
 * and all result arrays are owned. When `alpha == 0.0`, the same structure is discovered without reading
 * operand values, and every entry of it is zero.
 */
@Suppress("LongParameterList")
public fun KoblasEngine.gemm(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: SparseMatrix,
    transposeB: Boolean,
): SparseMatrix {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    val aRows = if (transposeA) a.cols else a.rows
    val bCols = if (transposeB) b.rows else b.cols
    // Return before orienting: transpose pointers are row-indexed and may exceed array limits
    // even when the product is empty.
    if (a.nnz == 0 || b.nnz == 0 || aRows == 0 || bCols == 0) return emptyResult(aRows, bCols, "gemm")
    val left = oriented(a, transposeA, alpha != 0.0)
    val right = oriented(b, transposeB, alpha != 0.0)
    return multiplySparse(left, right, alpha)
}

/**
 * Direct dense-destination `C = alpha · op(A) · op(B) + beta · C` for two sparse operands. No sparse
 * intermediate is created and implicit sparse zeros are not evaluated. `alpha == 0.0` reads neither
 * sparse operand's values; `beta == 0.0` does not read [c]. Sparse value buffers shared with [c] are
 * staged using [workspace].
 */
@Suppress("LongParameterList")
public fun KoblasEngine.gemm(
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: SparseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    workspace: Workspace? = null,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    if (alpha == 0.0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    if (a.nnz == 0 || b.nnz == 0) {
        // Only destination scaling remains; avoid row-indexed orientation scratch.
        scaleRegion(c, beta, vectors = this)
        return
    }
    staged(workspace, a, a.values === c.values) { stableA ->
        staged(workspace, b, b.values === c.values) { stableB ->
            val left = oriented(stableA, transposeA, true)
            val right = oriented(stableB, transposeB, true)
            scaleRegion(c, beta, vectors = this)
            multiplySparseInto(alpha, left, right, c)
        }
    }
}

/**
 * Dense selected-triangle `C = alpha · op(A) · op(A)ᵀ + beta · C`. Only the selected [lower] or upper
 * triangle of [c] is read or written, and products involving implicit sparse zeros are not evaluated.
 */
@Suppress("LongParameterList")
public fun KoblasEngine.syrk(
    alpha: Double,
    a: SparseMatrix,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean = true,
    workspace: Workspace? = null,
) {
    val n = if (transpose) a.cols else a.rows
    requireSyrkOperands(a, transpose, c)
    if (alpha == 0.0) {
        scaleRegion(c, beta, lower)
        return
    }
    if (a.nnz == 0 || n == 0) {
        // An empty pattern needs only triangle scaling, avoiding potentially oversized row-indexed adjacency.
        scaleRegion(c, beta, lower)
        return
    }
    staged(workspace, a, a.values === c.values) { stableA ->
        scaleRegion(c, beta, lower)
        withSymmetricRankScratch(workspace, n, stableA.rows, stableA.nnz) {
                sums,
                touchedAt,
                touched,
                rowPointers,
                adjacentColumns,
                adjacentPositions,
                rowCursor,
            ->
            symmetricRankInto(
                alpha, stableA, transpose, c, lower, sums, touchedAt, touched,
                rowPointers, adjacentColumns, adjacentPositions, rowCursor,
            )
        }
    }
}

/**
 * Fresh CSC selected triangle of `op(A) · op(A)ᵀ`. Stored-product support and cancellations are
 * retained, rows are sorted, and result arrays are owned. The opposite triangle is absent rather than
 * implicitly mirrored; consume the result with [symv] or [symm] when symmetric meaning is intended.
 */
public fun KoblasEngine.syrk(a: SparseMatrix, transpose: Boolean = false, lower: Boolean = true): SparseMatrix =
    symmetricRankProduct(a, transpose, lower)

/**
 * Fresh CSC `alpha · op(A) + B`. The structural union is retained, including explicit zeros,
 * cancellations, and A-only positions when alpha is zero. Zero alpha does not read A's values; B values
 * are copied. Output rows are strictly ascending and all arrays are independently owned.
 */
public fun KoblasEngine.addScaled(alpha: Double, a: SparseMatrix, transposeA: Boolean, b: SparseMatrix): SparseMatrix {
    val rows = if (transposeA) a.cols else a.rows
    val cols = if (transposeA) a.rows else a.cols
    requireSameShape(rows, cols, b, "addScaled")
    if (a.nnz == 0 && b.nnz == 0) return emptyResult(rows, cols, "addScaled")
    val left = oriented(a, transposeA, alpha != 0.0)
    // An identical pattern needs neither a structural merge nor room for a second copy of every entry.
    // Different patterns usually disagree at an early row, before every column pointer needs comparing.
    if (left.rowIndices.contentEquals(b.rowIndices) && left.colPointers.contentEquals(b.colPointers)) {
        val values = if (alpha == 0.0) {
            b.values.copyOf()
        } else {
            DoubleArray(left.nnz) { alpha * left.values[it] + b.values[it] }
        }
        return SparseMatrix.wrapTrusted(rows, cols, left.copyColumnPointers(), left.copyRowIndices(), values)
    }
    val union = left.nnz.toLong() + b.nnz
    requireShape(
        union <= Int.MAX_VALUE,
    ) { "scaled addition needs $union stored entries, more than one array can hold" }
    val pointers = IntArray(left.cols + 1)
    val rowIndices = IntArray(union.toInt())
    val values = DoubleArray(rowIndices.size)
    var count = 0
    for (column in 0 until left.cols) {
        count += SparseAccumulationKernels.mergeScaledColumns(
            alpha,
            left.rowIndices,
            left.values,
            left.colPointers[column],
            left.colPointers[column + 1],
            b.rowIndices,
            b.values,
            b.colPointers[column],
            b.colPointers[column + 1],
            rowIndices,
            values,
            count,
        )
        pointers[column + 1] = count
    }
    return SparseMatrix.wrapTrusted(
        left.rows,
        left.cols,
        pointers,
        if (count == rowIndices.size) rowIndices else rowIndices.copyOf(count),
        if (count == values.size) values else values.copyOf(count),
    )
}

/**
 * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (Sparse BLAS `ussm`).
 * The triangle flags follow [trsv]; the right-hand sides are the columns of [b] from the left and its
 * rows from the right. [workspace] reuses the diagonal and right-hand-side panel scratch.
 * A missing diagonal entry behaves as zero, and singularity is not reported, as in dense `dtrsm`.
 */
@Suppress("LongParameterList") // the BLAS dtrsm signature
public fun KoblasEngine.trsm(
    a: SparseMatrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
) {
    requireTriangularMatrixOperands(a, b, right, "trsm")
    val n = a.rows
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    if (n == 0) return
    val rightHandSides = if (right) b.rows else b.cols
    if (rightHandSides == 0) return
    // Snapshot an aliased triangle before alpha scales the right-hand sides and changes its coefficients.
    staged(workspace, a, a.values === b.values) { triangle ->
        scaleRegion(b, alpha, vectors = this)
        if (right) {
            withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                trsmRightCore(this, triangle, b, lower, !transpose, diagonal)
            }
        } else {
            withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                triangularLeftCore(
                    this, true, triangle, b, lower, transpose, unitDiag, diagonal, workspace,
                )
            }
        }
    }
}

/**
 * `B = alpha · op(T) · B` in place, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`). The triangle
 * flags follow [trmv]. This is multiplication rather than a solve, so zero or missing diagonal entries
 * participate normally and are never treated as singular. The destination may share [a]'s value buffer.
 */
@Suppress("LongParameterList") // the BLAS dtrmm signature
public fun KoblasEngine.trmm(
    a: SparseMatrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
) {
    requireTriangularMatrixOperands(a, b, right, "trmm")
    val n = a.rows
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    if (n == 0) return
    val rightHandSides = if (right) b.rows else b.cols
    if (rightHandSides == 0) return
    staged(workspace, a, a.values === b.values) { triangle ->
        scaleRegion(b, alpha, vectors = this)
        // Unlike trsm, trmm reads the diagonal once per right-hand side.
        withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
            if (right) {
                trmmRightCore(this, triangle, b, lower, transpose, unitDiag, diagonal)
            } else {
                triangularLeftCore(
                    this, false, triangle, b, lower, transpose, unitDiag, diagonal, workspace,
                )
            }
        }
    }
}

/**
 * A CSC result of this shape with nothing stored, for a call whose operands reach no position. [what]
 * names the operation in the one error this raises, a column count no array of pointers can hold.
 */
private fun KoblasEngine.emptyResult(rows: Int, cols: Int, what: String): SparseMatrix = SparseMatrix.wrapTrusted(
    rows,
    cols,
    IntArray(pointerLength(cols, what)),
    IntArray(0),
    DoubleArray(0),
)

/**
 * [a] in the requested orientation, materializing a transpose only where one is asked for.
 *
 * A zero alpha discovers the same structure without reading coefficients, so the transposed pattern is
 * built without copying coefficients: the contract says alpha of zero reads no operand value, and a
 * transpose that copied them would break it before the product ever ran.
 */
private fun KoblasEngine.oriented(a: SparseMatrix, transpose: Boolean, readValues: Boolean): SparseMatrix {
    if (!transpose) return a
    return transposeCsc(a, readValues)
}

/** `A · x`, or `Aᵀ · x` when [transpose], into a fresh result. */
public fun KoblasEngine.gemv(a: SparseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
    val size = if (transpose) a.cols else a.rows
    requireGemvOperands(a, transpose, x.size, size)
    val y = DoubleArray(size)
    gemv(1.0, a, x, 0.0, y, transpose)
    return y
}

/** `xᵀ·y` for a sparse [x] against a dense [y] (Sparse BLAS `usdot`); walks only the stored entries. */
public fun KoblasEngine.dot(x: SparseVector, y: DoubleArray): Double {
    requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
    return this.dotDense(x.indices, 0, x.values, 0, x.values.size, y)
}

/**
 * Dot product of [column] of [matrix] with [dense], whose length must equal [SparseMatrix.rows].
 * Only stored entries participate, including explicit zeros; absent positions are never evaluated.
 * Reads the current [SparseMatrix.values] on every call and permits that array to be [dense].
 *
 * Built-in implementations reuse the matrix's validated CSC support and call the indexed dot leaf
 * at the column's offsets without copying or rescanning indices. Custom implementations default to the
 * validated raw [dot] overload. The matrix's structural arrays must remain unchanged, as
 * [SparseMatrix.wrap] requires; this call retains no arrays or column view after returning.
 *
 * @throws IndexOutOfBoundsException if [column] is outside the matrix's columns.
 * @throws IllegalArgumentException if [dense] has the wrong length.
 */
public fun KoblasEngine.dotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double {
    validateColumnDot(matrix, column, dense)
    val start = matrix.colPointers[column]
    val count = matrix.colPointers[column + 1] - start
    return this.dotDense(matrix.rowIndices, start, matrix.values, start, count, dense)
}

/**
 * Returns the dot product of [count] indexed [values] and [dense]. Index and value windows are independent.
 * Repeated and unsorted indices are permitted: every supplied entry contributes once. A zero [count] permits
 * offsets at the corresponding array ends. Both windows and every selected index are validated before
 * arithmetic. [values] may be [dense] because this operation only reads its inputs.
 */
@Suppress("LongParameterList")
public fun KoblasEngine.dot(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    dense: DoubleArray,
): Double {
    requireSparseWindow(indices.size, indexOffset, count, "indices")
    requireSparseWindow(values.size, valueOffset, count, "values")
    validateSparseIndices(indices, indexOffset, count, dense.size, strictlyIncreasing = false)
    return this.dotDense(indices, indexOffset, values, valueOffset, count, dense)
}

/**
 * `xᵀ·y` for two sparse vectors, merging their index lists in one pass. Relies on both operands being
 * strictly ascending, which `SparseVector` validates.
 */
public fun KoblasEngine.dot(x: SparseVector, y: SparseVector): Double {
    requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
    return this.dotSparse(x.indices, 0, x.values, 0, x.values.size, y.indices, 0, y.values, 0, y.values.size)
}

/** `y += alpha·x` for a sparse [x] into a dense [y] (Sparse BLAS `usaxpy`), touching only x's stored
 *  positions. */
public fun KoblasEngine.axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
    requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
    if (alpha == 0.0) return
    this.axpy(x.indices, 0, x.values, 0, x.values.size, alpha, y)
}

/**
 * Performs `destination[indices(k)] += alpha * values(k)` over the supplied windows.
 *
 * Selected indices must be strictly increasing, making every destination unique. [values] and [destination]
 * must be distinct arrays. All preconditions are validated before mutation. When [alpha] is either signed
 * zero, no value or destination element is read and [destination] is unchanged, following BLAS `axpy`.
 */
@Suppress("LongParameterList")
public fun KoblasEngine.axpy(
    destination: DoubleArray,
    alpha: Double,
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
) {
    validateMutation(indices, indexOffset, values, valueOffset, count, destination)
    if (alpha == 0.0) return
    this.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
}

/** Write [x]'s stored entries into [out] at their positions, leaving the rest of [out] alone (Sparse
 *  BLAS `ussc`). Zero-fill [out] first for a plain densification. */
public fun KoblasEngine.scatter(x: SparseVector, out: DoubleArray) {
    requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
    computeIndexedScatter(x.indices, 0, x.values, 0, x.values.size, out)
}

/**
 * Assigns [count] indexed [values] into [destination], leaving every other element unchanged.
 * Selected indices must be strictly increasing, so duplicate-destination assignment is not permitted.
 * [values] and [destination] must be distinct arrays. Windows, indices, ordering, and aliasing are validated
 * before mutation; explicit stored values, including either signed zero, are assigned unchanged.
 */
@Suppress("LongParameterList")
public fun KoblasEngine.scatter(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    destination: DoubleArray,
) {
    validateMutation(indices, indexOffset, values, valueOffset, count, destination)
    computeIndexedScatter(indices, indexOffset, values, valueOffset, count, destination)
}

/**
 * Read [from] at [x]'s stored positions into [x]'s values (Sparse BLAS `usga`), the inverse of
 * [scatter]. The pattern is what [x] already stores and does not change; a position of [from] that is
 * nonzero and unstored stays unread, so this narrows a dense vector to a pattern rather than sparsifying
 * it.
 *
 * [x]'s values and [from] must be distinct arrays, as they must be for [scatter]. Reading and writing one
 * array through two index orders has no single answer: a scalar loop moves an entry per step while a
 * vector kernel reads a whole block before storing it, so the two would disagree on the overlap.
 */
public fun KoblasEngine.gather(x: SparseVector, from: DoubleArray) {
    requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
    require(x.values !== from) { "values and source must use distinct buffers" }
    this.gather(x.indices, 0, x.values, 0, x.values.size, from)
}

/**
 * [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`), leaving the rest of [from]
 * alone. The pair is one pass rather than two because the caller of a gather-and-zero wants [from]
 * emptied of exactly what it took.
 *
 * [x]'s values and [from] must be distinct arrays, for the reason [gather] gives.
 */
public fun KoblasEngine.gatherZero(x: SparseVector, from: DoubleArray) {
    requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
    require(x.values !== from) { "values and source must use distinct buffers" }
    this.gatherZero(x.indices, 0, x.values, 0, x.values.size, from)
}

/** Euclidean norm over the stored entries, rescaled as the dense [euclideanNorm] is. */
public fun KoblasEngine.nrm2(x: SparseVector): Double = this.nrm2(x.values, 0, x.values.size)

/**
 * Euclidean norm of [values] at the [count] positions in [indices]. Repeated and unsorted indices are
 * permitted and each occurrence contributes once. Finite-range rescaling protects against avoidable
 * square-sum overflow or underflow; unspecified exceptional results follow the selected implementation.
 * The touched window and all selected indices are validated before arithmetic.
 */
public fun KoblasEngine.nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
    requireSparseWindow(indices.size, indexOffset, count, "indices")
    validateSparseIndices(indices, indexOffset, count, values.size, strictlyIncreasing = false)
    return computeIndexedNrm2(indices, indexOffset, count, values)
}

/** `Sum |x_i|` over the stored entries. */
public fun KoblasEngine.asum(x: SparseVector): Double = this.asum(x.values, 0, x.values.size)

private fun KoblasEngine.validateMutation(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    destination: DoubleArray,
) {
    requireSparseWindow(indices.size, indexOffset, count, "indices")
    requireSparseWindow(values.size, valueOffset, count, "values")
    require(values !== destination) { "values and destination must use distinct buffers" }
    validateSparseIndices(indices, indexOffset, count, destination.size, strictlyIncreasing = true)
}
