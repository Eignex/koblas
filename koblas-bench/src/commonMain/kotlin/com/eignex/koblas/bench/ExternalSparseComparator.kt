package com.eignex.koblas.bench

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector

internal interface SparseComparator {
    val identity: String
    val threading: String
    fun dot(x: F64SparseVector, y: DoubleArray): Double
    fun axpy(alpha: Double, x: F64SparseVector, y: DoubleArray)
    fun scatter(x: F64SparseVector, y: DoubleArray)
    fun gather(x: F64SparseVector, from: DoubleArray, out: DoubleArray)
    fun gatherZero(x: F64SparseVector, from: DoubleArray, out: DoubleArray)
    fun prepare(a: F64SparseMatrix, triangular: Boolean = false, lower: Boolean = true, unitDiag: Boolean = false): PreparedSparseComparator
    fun sparseProduct(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix
}

internal interface PreparedSparseComparator : AutoCloseable {
    fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false)
    fun gemm(alpha: Double, b: F64DenseMatrix, beta: Double, c: F64DenseMatrix, transpose: Boolean = false)
    fun trsv(x: DoubleArray, out: DoubleArray, transpose: Boolean = false)
    fun trmv(x: DoubleArray, out: DoubleArray, transpose: Boolean = false)
    fun trsm(b: F64DenseMatrix, out: F64DenseMatrix, transpose: Boolean = false)
    fun trmm(b: F64DenseMatrix, out: F64DenseMatrix, transpose: Boolean = false)
    fun sparseProduct(right: PreparedSparseComparator): F64SparseMatrix
}

internal expect fun oneMklSparseComparator(): SparseComparator?
