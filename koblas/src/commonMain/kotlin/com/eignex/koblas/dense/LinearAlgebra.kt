package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.koblas

/**
 * Both halves of the compute seam at once: the [Blas] routines and the [Lapack] factorizations built on
 * them. [koblas] is one of these, composed from whichever backend won each half.
 *
 * Implement this when a backend provides both, which is the usual case for a host library. Implement
 * [Blas] or [Lapack] alone when it does not — the two are ranked and installed independently, so a host
 * with CBLAS but no LAPACKE still accelerates its level-2 and level-3 work.
 */
interface LinearAlgebra :
    Blas,
    Lapack

/** LU-factorize this square matrix with the active backend ([koblas]). */
fun DenseMatrix.lu(): LuDecomposition = koblas.factor(this)

/** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for this factorization with the active backend. */
fun LuDecomposition.solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.solve(this, b, transpose)

/** Matrix-matrix product `this · other` with the active backend. */
fun DenseMatrix.matMul(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)
