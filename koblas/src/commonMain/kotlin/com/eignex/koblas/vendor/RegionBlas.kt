@file:Suppress("LongParameterList") // the BLAS signatures

package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrixRegion
import com.eignex.koblas.DenseVector
import com.eignex.koblas.dense.MatrixStructure

/**
 * The [Blas] entry points that [com.eignex.koblas.dense.DenseBlas] accepts borrowed regions in, reached with each
 * region's own offset and leading dimension.
 *
 * CBLAS addresses every matrix operand as a pointer and a column spacing, so a borrowed rectangle is already an
 * ordinary operand to it. A binding that can hand caller storage over in place implements this, and a host
 * composition then passes a region through instead of packing it into scratch and copying the destination back.
 * A binding that copies operands into foreign memory anyway gains nothing from it and need not.
 *
 * Each call has the contract of its [Blas] counterpart. Overlap is the caller's to stage first: a region here
 * is passed as it lies, and no input may share an entry with the destination.
 */
internal interface RegionBlas {
    /** [Blas.gemv] over a borrowed [a]. */
    fun gemv(alpha: Double, a: DenseMatrixRegion, transposeA: Boolean, x: DenseVector, beta: Double, y: DenseVector)

    /** [Blas.gemm] over borrowed operands. */
    fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
    )

    /** [Blas.syrk] over borrowed operands. */
    fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    )

    /** [Blas.trsv] over a borrowed [a]. */
    fun trsv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector)

    /** [Blas.trsm] over borrowed operands. */
    fun trsm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean = false,
    )
}
