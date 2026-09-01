package com.eignex.koblas.sparse.basis

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireShape
import kotlin.math.abs
import kotlin.math.max

/** Computes a basis solve residual without asking a factorization to expose its factors. */
internal fun basisSolveQuality(
    a: F64SparseMatrix,
    basicIndex: IntArray,
    rhs: DoubleArray,
    solution: F64IndexedVector,
    transpose: Boolean,
): F64BasisSolveQuality {
    val n = a.rows
    requireShape(basicIndex.size == n) { "basis has ${basicIndex.size} columns, expected $n" }
    requireShape(rhs.size == n) { "rhs size ${rhs.size} != $n" }
    requireShape(solution.size == n) { "solution size ${solution.size} != $n" }
    val x = solution.toDoubleArray()
    val product = DoubleArray(n)
    if (transpose) {
        for (slot in 0 until n) {
            var sum = 0.0
            a.forEachInColumn(basicIndex[slot]) { row, value -> sum += value * x[row] }
            product[slot] = sum
        }
    } else {
        for (slot in 0 until n) {
            val multiplier = x[slot]
            if (multiplier != 0.0) {
                a.forEachInColumn(basicIndex[slot]) { row, value ->
                    product[row] += value * multiplier
                }
            }
        }
    }
    var residual = 0.0
    var scale = 0.0
    for (i in 0 until n) {
        residual = max(residual, abs(product[i] - rhs[i]))
        scale = max(scale, max(abs(product[i]), abs(rhs[i])))
    }
    return F64BasisSolveQuality(residual, residual / max(1.0, scale))
}
