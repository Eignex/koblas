package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector

internal interface SparseComparator {
    val identity: String
    val threading: String
    fun dot(x: SparseVector, y: DoubleArray): Double
    fun axpy(alpha: Double, x: SparseVector, y: DoubleArray)
    fun scatter(x: SparseVector, y: DoubleArray)
    fun gather(x: SparseVector, from: DoubleArray, out: DoubleArray)
    fun gatherZero(x: SparseVector, from: DoubleArray, out: DoubleArray)
    fun prepare(a: SparseMatrix, triangular: Boolean = false, lower: Boolean = true, unitDiag: Boolean = false): PreparedSparseComparator
    fun sparseProduct(a: SparseMatrix, b: SparseMatrix): SparseMatrix
}

internal interface PreparedSparseComparator : AutoCloseable {
    fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false)
    fun gemm(alpha: Double, b: DenseMatrix, beta: Double, c: DenseMatrix, transpose: Boolean = false)
    fun trsv(x: DoubleArray, out: DoubleArray, transpose: Boolean = false)
    fun trmv(x: DoubleArray, out: DoubleArray, transpose: Boolean = false)
    fun trsm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean = false)
    fun trmm(b: DenseMatrix, out: DenseMatrix, transpose: Boolean = false)
    fun sparseProduct(right: PreparedSparseComparator): SparseMatrix
}

internal expect fun oneMklSparseComparator(): SparseComparator?
