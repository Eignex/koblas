@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import kotlin.jvm.JvmOverloads

/** Prepares an immutable snapshot of this matrix for repeated products. */
public fun SparseMatrix.prepare(): PreparedSparseMatrix = koblas.prepare(this)

/** Symmetric product with [x] into a fresh dense vector, reading only this matrix's selected triangle. */
@JvmOverloads
public fun SparseMatrix.symv(x: DoubleArray, lower: Boolean = true): DoubleArray =
    DoubleArray(rows).also { koblas.symv(1.0, this, x, 0.0, it, lower) }

/** Symmetric `y = alpha · A · x + beta · y`, reading only this matrix's selected triangle. */
@Suppress("LongParameterList")
@JvmOverloads
public fun SparseMatrix.symv(
    alpha: Double,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean = true,
): Unit = koblas.symv(alpha, this, x, beta, y, lower)

/** Symmetric sparse-dense product into [c], with this selected-triangle matrix on either side. */
@Suppress("LongParameterList")
@JvmOverloads
public fun SparseMatrix.symm(
    alpha: Double,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean = true,
    right: Boolean = false,
    workspace: Workspace? = null,
): Unit = koblas.symm(alpha, this, b, beta, c, lower, right, workspace)

/** Fresh selected CSC triangle of `op(A) · op(A)ᵀ`. */
@JvmOverloads
public fun SparseMatrix.syrk(transpose: Boolean = false, lower: Boolean = true): SparseMatrix =
    koblas.syrk(this, transpose, lower)

/** Dense selected-triangle sparse rank-k product into [c]. */
@Suppress("LongParameterList")
@JvmOverloads
public fun SparseMatrix.syrk(
    alpha: Double,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean = true,
    workspace: Workspace? = null,
): Unit = koblas.syrk(alpha, this, transpose, beta, c, lower, workspace)

/** Fresh CSC `alpha · op(A) + B`, retaining the structural union. */
public fun SparseMatrix.addScaled(alpha: Double, transpose: Boolean, b: SparseMatrix): SparseMatrix =
    koblas.addScaled(alpha, this, transpose, b)
