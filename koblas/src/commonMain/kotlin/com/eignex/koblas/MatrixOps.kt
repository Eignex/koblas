@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.dense.asVector
import com.eignex.koblas.dense.symmetricStructure
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.MissingVendorException
import kotlin.jvm.JvmOverloads

/**
 * `y = alpha * A * x + beta * y` (BLAS `dgemv`) into [destination].
 *
 * Both operands are dense storage, which is what a vendor can address: [x] may be contiguous or strided, and
 * either reaches the library as a pointer and an increment. A [SparseVector] carries a pattern rather than an
 * increment and there is no entry point that takes one, so it is excluded by the type rather than refused at
 * the call.
 *
 * Operands may share [destination]'s backing array. They are snapshotted before [destination] is scaled or
 * written, so aliasing has the same result as a call over independent inputs.
 */
public fun DenseMatrix.gemvInto(alpha: Double, x: DenseVector, beta: Double, destination: DoubleArray) {
    val a = this
    requireShape(a.cols == x.size) { "gemvInto shape mismatch: A is ${a.rows}x${a.cols}, x size ${x.size}" }
    requireShape(destination.size == a.rows) {
        "gemvInto: destination size ${destination.size} != rows ${a.rows}"
    }
    // The seams quick-return on a zero-extent operand before scaling, which is netlib's rule for gemv but
    // not the contract above: this one promises that `beta == 0.0` overwrites a destination that may arrive
    // holding NaN. Settling it here keeps a zero-column matrix answering the same way as any other.
    if (alpha == 0.0 || a.cols == 0) {
        destination.prescale(beta)
        return
    }
    val stableX = x.stableFor(destination)
    val stableA = a.stableFor(destination)
    vendorBlas().gemv(alpha, stableA, transposeA = false, x = stableX, beta = beta, y = destination.asVector())
}

/** [gemvInto] with `alpha = 1, beta = 0`, so `destination` receives `A * x`. */
public fun DenseMatrix.gemvInto(x: DenseVector, destination: DoubleArray): Unit = gemvInto(1.0, x, 0.0, destination)

/**
 * `y = alpha * A * x + beta * y` for a symmetric `A` (BLAS `dsymv`) into [destination]. Only the [lower]
 * triangle is read, diagonal included, and `beta == 0.0` overwrites [destination] without reading it.
 *
 * Reading one triangle is what lets a caller maintain its symmetric matrix with [DenseMatrix.syr], which
 * touches half the entries, rather than the full-matrix [DenseMatrix.ger]. Outside the stored triangle
 * each entry is taken from its mirror, so the other half may hold anything. [x] and this matrix may share
 * [destination]'s backing array; built-in aliases are snapshotted before any output is written.
 */
@Suppress("LongParameterList") // the BLAS dsymv signature
@JvmOverloads
public fun DenseMatrix.symvInto(
    alpha: Double,
    x: DenseVector,
    beta: Double,
    destination: DoubleArray,
    lower: Boolean = true,
) {
    requireSquare(this, "symvInto")
    requireShape(cols == x.size) { "symvInto shape mismatch: A is ${rows}x$cols, x size ${x.size}" }
    requireShape(destination.size == rows) { "symvInto: destination size ${destination.size} != rows $rows" }
    if (alpha == 0.0) {
        destination.prescale(beta)
        return
    }
    val stableX = x.stableFor(destination)
    val stableA = stableFor(destination)
    vendorBlas().symv(alpha, stableA, symmetricStructure(lower), stableX, beta, destination.asVector())
}

/** [symvInto] with `alpha = 1, beta = 0`, so `destination` receives `A * x`. */
@JvmOverloads
public fun DenseMatrix.symvInto(x: DenseVector, destination: DoubleArray, lower: Boolean = true): Unit =
    symvInto(1.0, x, 0.0, destination, lower)

/** The `beta * y` half of a matvec. A zero [beta] overwrites without reading, as BLAS specifies, so the
 *  destination's previous contents cannot poison the result. */
private fun DoubleArray.prescale(beta: Double) = applyBeta(koblas.vectorKernels, this, 0, size, beta)

/**
 * A snapshot when [destination] is this vector's live backing array, and the vector itself otherwise.
 *
 * BLAS leaves a destination overlapping an input undefined, so an operand that would be rewritten under the
 * call is copied first and the call sees the values it was given. Exhaustive over the two dense shapes: the
 * copy keeps the window's own origin and spacing, because those address the copy exactly as they addressed
 * the original.
 */
private fun DenseVector.stableFor(destination: DoubleArray): DenseVector = when {
    data !== destination -> this
    this is StridedVector -> StridedVector(data.copyOf(), offset, size, stride)
    else -> DenseVector.wrap(data.copyOf())
}

/** Stable dense matrix storage when [destination] is its live backing array. */
private fun DenseMatrix.stableFor(destination: DoubleArray): DenseMatrix =
    if (data === destination) DenseMatrix.wrap(rows, cols, data.copyOf()) else this

/**
 * Rank-one update `A = A + alpha * x * yT` (BLAS `dger`) in place. Subtract by passing
 * `alpha = -1.0`.
 */
public fun DenseMatrix.ger(alpha: Double, x: DenseVector, y: DenseVector) {
    requireShape(rows == x.size && cols == y.size) {
        "ger shape mismatch: A is ${rows}x$cols, x ${x.size}, y ${y.size}"
    }
    if (alpha == 0.0) return
    vendorBlas().ger(alpha, x, y, this)
}

/** Symmetric rank-1 update `A += alpha * x * xT` (BLAS `dsyr`) in place. See [DenseBlas.syr]. */
@JvmOverloads
public fun DenseMatrix.syr(alpha: Double, x: DenseVector, lower: Boolean = true): Unit = koblas.syr(
    alpha,
    x,
    this,
    lower,
)

/** Symmetric rank-2 update `A += alpha * (x * yT + y * xT)` (BLAS `dsyr2`) in place. See [DenseBlas.syr2]. */
@JvmOverloads
public fun DenseMatrix.syr2(alpha: Double, x: DenseVector, y: DenseVector, lower: Boolean = true): Unit =
    koblas.syr2(alpha, x, y, this, lower)

/** The selected library, or the failure a Level 2 call owes a host without one. */
private fun vendorBlas(): Blas = koblas.vendor ?: throw MissingVendorException()
