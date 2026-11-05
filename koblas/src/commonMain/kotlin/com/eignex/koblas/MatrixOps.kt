@file:Suppress("LongParameterList") // Matrix operations expose their scaling and triangle flags.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.internal.operations.requireGemvOperands
import com.eignex.koblas.internal.operations.requireGerOperands
import com.eignex.koblas.internal.operations.requireSyrOperands
import com.eignex.koblas.internal.operations.requireTriangularVectorOperands
import com.eignex.koblas.internal.operations.sumMatrix

/**
 * Plain sum of all entries. Dense storage uses [KoblasEngine.sum] through the default [koblas] engine,
 * sparse storage sums only its stored values, and custom matrices are read through [Matrix.get].
 * An empty matrix sums to zero. Summation order and rounding depend on the storage and engine.
 */
public fun Matrix.sum(): Double = koblas.sumMatrix(this)

/**
 * Sums along [axis] into a fresh dense vector: `axis = 0` reduces rows and returns [Matrix.cols]
 * column sums; `axis = 1` reduces columns and returns [Matrix.rows] row sums. A zero-length reduced
 * dimension contributes zero to each result. The result owns its storage.
 *
 * Dense views address only their logical entries. Sparse matrices visit only stored entries without
 * densifying; custom matrices are read through [Matrix.get]. Summation order and rounding depend on
 * the storage and default [koblas] engine.
 *
 * @throws IllegalArgumentException if [axis] is neither zero nor one.
 */
public fun Matrix.sum(axis: Int): DenseVector = DenseVector.wrap(koblas.sumMatrix(this, axis))

/**
 * Writes `destination = alpha * op(A) * x + beta * destination`, where this region is `A` and
 * [transpose] selects `Aᵀ`. Vector sizes must match the dimensions of `op(A)` exactly; offsets and
 * positive or negative strides are taken from the vectors.
 *
 * Uses the immutable default [koblas] engine and [KoblasEngine.gemv] contracts, including zero multipliers
 * and snapshots of inputs sharing [destination]'s buffer. [scratch] reuses temporary storage.
 */
public fun DenseMatrixRegion.gemvInto(
    x: DenseVector,
    destination: DenseVector,
    alpha: Double = 1.0,
    beta: Double = 0.0,
    transpose: Boolean = false,
    scratch: Scratch? = null,
) {
    requireGemvOperands(this, transpose, x.size, destination.size)
    koblas.gemv(
        alpha, this, x.values, x.offset, x.stride,
        beta, destination.values, destination.offset, destination.stride, transpose, scratch,
    )
}

/**
 * Writes `destination = alpha * op(A) * op(B) + beta * destination`, where this region is `A` and
 * [b] is `B`. [transposeA] and [transposeB] independently select the transposed operands without
 * materializing them. By default, writes the ordinary product `A * B`.
 *
 * Uses the default [koblas] engine and [KoblasEngine.gemm] contracts, including zero multipliers and
 * snapshots of operands sharing [destination]'s buffer. [scratch] reuses temporary storage.
 */
public fun DenseMatrixRegion.gemmInto(
    b: DenseMatrixRegion,
    destination: DenseMatrixRegion,
    alpha: Double = 1.0,
    beta: Double = 0.0,
    transposeA: Boolean = false,
    transposeB: Boolean = false,
    scratch: Scratch? = null,
): Unit = koblas.gemm(alpha, this, transposeA, b, transposeB, beta, destination, scratch)

/**
 * Updates this region as `A = A + alpha * x * yᵀ`. [x] must have exactly [Matrix.rows] entries and [y]
 * exactly [Matrix.cols] entries; their offsets and positive or negative strides are preserved.
 *
 * Uses the default [koblas] engine and [KoblasEngine.ger] contracts. Overlapping vectors are
 * snapshotted before writing, borrowing temporary storage from [scratch] when supplied.
 */
public fun DenseMatrixRegion.ger(x: DenseVector, y: DenseVector, alpha: Double = 1.0, scratch: Scratch? = null) {
    requireGerOperands(x.size, y.size, this)
    koblas.ger(alpha, x.values, x.offset, x.stride, y.values, y.offset, y.stride, this, scratch)
}

/**
 * Updates this square region as `A = A + alpha * x * xᵀ`, writing only the [lower] or upper triangle.
 * [x] must have exactly the matrix order; its offset and stride are preserved.
 *
 * Uses the default [koblas] engine and [KoblasEngine.syr] contracts. An overlapping [x] is snapshotted
 * before writing, borrowing temporary storage from [scratch] when supplied.
 */
public fun DenseMatrixRegion.syr(x: DenseVector, alpha: Double = 1.0, lower: Boolean = true, scratch: Scratch? = null) {
    requireSyrOperands(this, x.size, "syr")
    koblas.syr(alpha, x.values, x.offset, x.stride, this, lower, scratch)
}

/**
 * Solves `op(A) * x = b` in place, with [x] holding `b` on entry and the solution on return.
 * This square region supplies the [lower] or upper triangle, [transpose] selects `Aᵀ`, and [unitDiag]
 * takes the diagonal as one. [x] must have exactly the matrix order, with any valid offset and stride.
 *
 * Uses the default [koblas] engine and [KoblasEngine.trsv] contracts, including snapshots when this
 * region shares [x]'s buffer. [scratch] reuses temporary storage. No singularity check is performed.
 */
public fun DenseMatrixRegion.trsv(
    x: DenseVector,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    requireTriangularVectorOperands(this, x.size, "trsv")
    koblas.trsv(this, x.values, x.offset, x.stride, lower, transpose, unitDiag, scratch)
}

/**
 * Solves `op(A) * X = alpha * B` in place, with [destination] holding `B` on entry and `X` on return.
 * With [right], solves `X * op(A) = alpha * B`. This square region supplies the [lower] or upper
 * triangle; [transpose] selects `Aᵀ` and [unitDiag] takes the diagonal as one.
 *
 * Uses the default [koblas] engine and [KoblasEngine.trsm] contracts. [destination] may share this
 * region's buffer; [scratch] reuses temporary storage. No singularity check is performed.
 */
public fun DenseMatrixRegion.trsm(
    destination: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    scratch: Scratch? = null,
): Unit = koblas.trsm(this, destination, lower, transpose, unitDiag, right, alpha, scratch)
