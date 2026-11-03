@file:kotlin.jvm.JvmName("DenseBlas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.configuration.*
import com.eignex.koblas.internal.dense.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.internal.sparse.*
import com.eignex.koblas.matrix.*
import com.eignex.koblas.sparse.*

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
    if (alpha == 0.0) return
    val source = x.stableFor(y)
    axpy(y.values, y.offset, alpha, source.values, source.offset, x.size, y.stride, source.stride)
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
    if (sameVectorMapping(x, y)) return
    val contiguous = x.stride == 1 && y.stride == 1
    val provider = if (this is CombinedEngine) copyProvider(x.size, contiguous) else this
    // Portable contiguous copy is a memmove, so it needs no alias snapshot.
    val source = if (provider === PortableEngine && contiguous) x else x.stableFor(y)
    provider.copy(source.values, source.offset, y.values, y.offset, x.size, source.stride, y.stride)
}

/** Exchanges the entries of [x] and [y]. */
public fun KoblasEngine.swap(x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "swap")
    if (sameVectorMapping(x, y)) return
    withStablePair(x, y) { a, b -> swap(a.values, a.offset, b.values, b.offset, a.size, a.stride, b.stride) }
}

/** Applies [rot] through this engine, including its overlap and negative stride semantics. */
public fun KoblasEngine.rot(x: DenseVector, y: DenseVector, c: Double, s: Double) {
    requireSameSize(x.size, y.size, "rot")
    if (c == 1.0 && s == 0.0) return
    withStablePair(x, y) { a, b -> rot(a.values, a.offset, b.values, b.offset, a.size, c, s, a.stride, b.stride) }
}

/** Both inputs precede every write; at a shared entry the final write belongs to [y]. */
private inline fun withStablePair(x: DenseVector, y: DenseVector, operation: (DenseVector, DenseVector) -> Unit) {
    if (!x.overlaps(y)) return operation(x, y)
    val a = DenseVector.wrap(x.toDoubleArray())
    val b = DenseVector.wrap(y.toDoubleArray())
    operation(a, b)
    for (i in 0 until x.size) x[i] = a[i]
    for (i in 0 until y.size) y[i] = b[i]
}

/** Fresh dense product of two dense regions using this engine. */
public fun KoblasEngine.gemm(a: DenseMatrixRegion, b: DenseMatrixRegion): DenseMatrix =
    product(1.0, a, false, b, false) as DenseMatrix

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
    if (a.values === y || x === y) {
        gemv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), transpose, workspace)
        return
    }
    val implementation = wholeCallFor(
        if (transpose) EngineImplementation.GemvTransposed else EngineImplementation.Gemv,
        a.rows,
        a.cols,
        borrowed = a !is DenseMatrix,
    )
    if (implementation != null) {
        implementation.computeGemv(alpha, a, transpose, DenseVector.wrap(x), beta, DenseVector.wrap(y))
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
    val implementation = wholeCallFor(
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
    val panels = panelProvider
    val rows = a.rows
    if (transpose) {
        forEachPanel(
            a.cols,
            panels.executionGroup(PanelWork.MultiDot, rows, a.cols, xStride == 1),
        ) { start, width ->
            panels.multiDot(
                alpha, a.values, a.offset + start * a.leadingDimension, a.leadingDimension, x,
                xOffset, xStride, rows, width, beta, y,
                yOffset + start * yStride, yStride,
            )
        }
    } else {
        applyBeta(this, y, yOffset, rows, beta, yStride)
        forEachPanel(
            a.cols,
            panels.executionGroup(PanelWork.ColumnUpdate, rows, a.cols, yStride == 1),
        ) { start, width ->
            panels.columnUpdate(
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
    val implementation = wholeCallFor(
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
    val implementation = wholeCallFor(
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
    if (a.values === y || x === y) {
        symv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), lower, workspace)
        return
    }
    val implementation = wholeCallFor(EngineImplementation.Symv, a.rows, a.cols, borrowed = a !is DenseMatrix)
    if (implementation != null) {
        implementation.computeSymv(alpha, a, symmetricStructure(lower), DenseVector.wrap(x), beta, DenseVector.wrap(y))
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
    val implementation = wholeCallFor(EngineImplementation.Symv, a.rows, a.cols, 1, a !is DenseMatrix)
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
    val panels = panelProvider
    val n = a.rows
    applyBeta(this, y, yOffset, a.rows, beta, yStride)
    val group = panels.executionGroup(PanelWork.CoupledDotUpdate, n, n, xStride == 1 && yStride == 1)
    forEachTrianglePanel(n, group, lower, fromDiagonal = false) { start, width, window, rows ->
        symmetricCorner(alpha, a, x, xOffset, xStride, y, yOffset, yStride, start, width, lower)
        if (rows > 0) {
            panels.coupledUpdateDot(
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
    val implementation = wholeCallFor(
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
    val implementation = wholeCallFor(
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
    val implementation = wholeCallFor(
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
    val implementation = wholeCallFor(EngineImplementation.Ger, a.rows, a.cols, 1, a !is DenseMatrix)
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
    val panels = panelProvider
    forEachPanel(
        columns,
        panels.executionGroup(PanelWork.RankUpdate, rows, columns, sxStride == 1),
    ) { start, width ->
        panels.rankUpdate(
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
    if (x === a.values || y === a.values) {
        ger(alpha, DenseVector.wrap(x), DenseVector.wrap(y), a, workspace)
        return
    }
    val implementation = wholeCallFor(EngineImplementation.Ger, a.rows, a.cols, borrowed = a !is DenseMatrix)
    if (implementation != null) {
        implementation.computeGer(alpha, DenseVector.wrap(x), DenseVector.wrap(y), a)
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
    val implementation = wholeCallFor(EngineImplementation.Syr, n, n, 1, a !is DenseMatrix)
    staged(workspace, x, regionVectorOverlap(a, x)) { sx ->
        if (implementation != null) {
            implementation.computeSyr(alpha, sx, a, symmetricStructure(lower))
            return@staged
        }
        val panels = panelProvider
        val xv = sx.values
        val origin = sx.offset
        val step = sx.stride
        val group = panels.executionGroup(PanelWork.RankUpdate, n, n, sx.stride == 1)
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
            panels.rankUpdate(
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
    val implementation = wholeCallFor(EngineImplementation.Syr2, n, n, 1, a !is DenseMatrix)
    stagedVectors(workspace, x, y, a) { sx, sy ->
        if (implementation != null) {
            implementation.computeSyr2(alpha, sx, sy, a, symmetricStructure(lower))
            return@stagedVectors
        }
        syr2Core(alpha, sx, sy, a, lower)
    }
}

private fun KoblasEngine.syr2Core(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion, lower: Boolean) {
    val panels = panelProvider
    val n = a.rows
    val xv = x.values
    val yv = y.values
    val xOrigin = x.offset
    val xStep = x.stride
    val yOrigin = y.offset
    val yStep = y.stride
    val lda = a.leadingDimension
    val group = panels.executionGroup(PanelWork.RankUpdate, n, n, xStep == 1 && yStep == 1)
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
        panels.rankTwoUpdate(
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
    if (a.values === x) {
        trsv(a, DenseVector.wrap(x), lower, transpose, unitDiag, workspace)
        return
    }
    val implementation = wholeCallFor(
        if (transpose) EngineImplementation.TrsvTransposed else EngineImplementation.Trsv,
        a.rows,
        a.cols,
        borrowed = a !is DenseMatrix,
    )
    if (implementation != null) {
        implementation.computeTrsv(a, triangle(lower, unitDiag), transpose, DenseVector.wrap(x))
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
    val implementation = wholeCallFor(
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
    val panels = panelProvider
    val triangles = triangularProvider
    val n = a.rows
    val lda = a.leadingDimension
    val ascending = lower != transpose
    forEachTriangularBlock(n, ascending) { start, end ->
        // The rest of the stored triangle beside this block: below it when lower, above it when upper.
        val outside = if (lower) end else 0
        val outsideRows = if (lower) n - end else start
        val block = a.offset + outside + start * lda
        if (transpose && outsideRows > 0) {
            panels.multiDot(
                -1.0, a.values, block, lda, x, xOffset + outside * xStride, xStride, outsideRows, end - start,
                1.0, x, xOffset + start * xStride, xStride,
            )
        }
        triangles.diagonalSolve(
            a.values, lda, start, end - start, transpose, lower != transpose, unitDiag,
            x, xOffset + start * xStride, xStride, 1, 1, a.offset,
        )
        if (!transpose && outsideRows > 0) {
            panels.columnUpdate(
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
    if (a.values === x) {
        trmv(a, DenseVector.wrap(x), lower, transpose, unitDiag, workspace)
        return
    }
    val implementation = wholeCallFor(
        if (transpose) EngineImplementation.TrmvTransposed else EngineImplementation.Trmv,
        a.rows,
        a.cols,
        borrowed = a !is DenseMatrix,
    )
    if (implementation != null) {
        implementation.computeTrmv(a, triangle(lower, unitDiag), transpose, DenseVector.wrap(x))
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
    if (n == 0) return
    val implementation = wholeCallFor(
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
    val panels = panelProvider
    val triangles = triangularProvider
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
            panels.columnUpdate(
                1.0, a.values, block, lda, x, xOffset + start * xStride, xStride, outsideRows, end - start,
                x, xOffset + outside * xStride, xStride,
            )
        }
        triangles.diagonalMultiply(
            a.values, lda, start, end - start, transpose, lower != transpose, unitDiag,
            x, xOffset + start * xStride, xStride, 1, 1, a.offset,
        )
        if (transpose && outsideRows > 0) {
            panels.multiDot(
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
    val implementation = wholeCallFor(
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
