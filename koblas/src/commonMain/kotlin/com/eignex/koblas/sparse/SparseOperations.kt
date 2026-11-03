@file:kotlin.jvm.JvmName("SparseBlas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.dense.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.dense.euclideanNorm
import com.eignex.koblas.internal.portable.sparse.*
import com.eignex.koblas.internal.sparse.*
import com.eignex.koblas.matrix.*

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
                sparseSymmetricVectorColumn(
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
        trsvCore(stable, x, lower, transpose, unitDiag)
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
        trmvCore(stable, x, lower, transpose, unitDiag)
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
