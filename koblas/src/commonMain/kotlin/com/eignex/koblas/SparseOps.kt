@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import kotlin.jvm.JvmOverloads

/** Prepares an immutable snapshot of this matrix for repeated products. */
public fun SparseMatrix.prepare(): PreparedSparseMatrix = koblas.prepare(this)

/** `A · x`, or `Aᵀ · x` when [transpose], into a fresh dense result over the stored entries alone. */
@JvmOverloads
public fun SparseMatrix.gemv(x: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.gemv(this, x, transpose)

/** In-place `y = alpha · op(A) · x + beta · y`; see [com.eignex.koblas.sparse.SparseBlas.gemv]. */
@Suppress("LongParameterList") // the BLAS dgemv signature
@JvmOverloads
public fun SparseMatrix.gemvInto(
    alpha: Double,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean = false,
): Unit = koblas.gemv(alpha, this, x, beta, y, transpose)

/** Symmetric product with [x] into a fresh dense vector, reading only this matrix's selected triangle. */
@JvmOverloads
public fun SparseMatrix.symv(x: DoubleArray, lower: Boolean = true): DoubleArray =
    DoubleArray(rows).also { koblas.symv(1.0, this, x, 0.0, it, lower) }

/** Symmetric `y = alpha · A · x + beta · y`, reading only this matrix's selected triangle. */
@Suppress("LongParameterList") // the BLAS dsymv signature
@JvmOverloads
public fun SparseMatrix.symv(
    alpha: Double,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean = true,
): Unit = koblas.symv(alpha, this, x, beta, y, lower)

/** Symmetric sparse-dense product into [c], with this selected-triangle matrix on either side. */
@Suppress("LongParameterList") // the BLAS dsymm signature plus the workspace
@JvmOverloads
public fun SparseMatrix.symm(
    alpha: Double,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean = true,
    right: Boolean = false,
    workspace: MatrixWorkspace? = null,
): Unit = koblas.symm(alpha, this, b, beta, c, lower, right, workspace)

/** `C = alpha · op(A) · op(B) + beta · C` into a caller-owned dense destination, or the mirrored product
 *  when [right]; see [com.eignex.koblas.sparse.SparseBlas.gemm]. */
@Suppress("LongParameterList") // the BLAS dgemm signature, the side, and the workspace
@JvmOverloads
public fun SparseMatrix.gemmInto(
    alpha: Double,
    transpose: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    right: Boolean = false,
    workspace: MatrixWorkspace? = null,
): Unit = koblas.gemm(alpha, this, transpose, b, transposeB, beta, c, right, workspace)

/** Fresh CSC `alpha · op(A) · op(B)` for two sparse operands, retaining discovered structure. */
public fun SparseMatrix.gemm(alpha: Double, transpose: Boolean, b: SparseMatrix, transposeB: Boolean): SparseMatrix =
    koblas.gemm(alpha, this, transpose, b, transposeB)

/** Direct dense-destination `C = alpha · op(A) · op(B) + beta · C` for two sparse operands. */
@Suppress("LongParameterList") // the BLAS dgemm signature plus the workspace
@JvmOverloads
public fun SparseMatrix.gemmInto(
    alpha: Double,
    transpose: Boolean,
    b: SparseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    workspace: MatrixWorkspace? = null,
): Unit = koblas.gemm(alpha, this, transpose, b, transposeB, beta, c, workspace)

/** Fresh selected CSC triangle of `op(A) · op(A)ᵀ`. */
@JvmOverloads
public fun SparseMatrix.syrk(transpose: Boolean = false, lower: Boolean = true): SparseMatrix =
    koblas.syrk(this, transpose, lower)

/** Dense selected-triangle sparse rank-k product into [c]. */
@Suppress("LongParameterList") // the BLAS dsyrk signature plus the workspace
@JvmOverloads
public fun SparseMatrix.syrk(
    alpha: Double,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean = true,
    workspace: MatrixWorkspace? = null,
): Unit = koblas.syrk(alpha, this, transpose, beta, c, lower, workspace)

/** Fresh CSC `alpha · op(A) + B`, retaining the structural union. */
public fun SparseMatrix.addScaled(alpha: Double, transpose: Boolean, b: SparseMatrix): SparseMatrix =
    koblas.addScaled(alpha, this, transpose, b)
