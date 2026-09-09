package com.eignex.koblas.sparse.basis

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.requireHfactorShape
import kotlin.math.abs
import kotlin.math.max

/**
 * Computes a basis solve residual without asking a factorization to expose its factors.
 *
 * [unitRows] carries a repaired basis: where an entry is not negative its slot holds the unit column of
 * that row rather than a column of [a], and [basicIndex] names nothing there. A basis of columns alone
 * passes null and every slot is read from [a].
 */
@Suppress("LongParameterList") // the basis in two arrays, its operands, and the direction
internal fun basisSolveQuality(
    a: SparseMatrix,
    basicIndex: IntArray,
    unitRows: IntArray?,
    rhs: DoubleArray,
    solution: IndexedVector,
    transpose: Boolean,
): BasisSolveQuality {
    val n = a.rows
    requireHfactorShape(basicIndex.size == n) { "basis has ${basicIndex.size} columns, expected $n" }
    requireHfactorShape(rhs.size == n) { "rhs size ${rhs.size} != $n" }
    requireHfactorShape(solution.size == n) { "solution size ${solution.size} != $n" }
    val x = solution.toDoubleArray()
    val product = DoubleArray(n)
    if (transpose) {
        for (slot in 0 until n) {
            val unit = unitRows?.get(slot) ?: -1
            if (unit >= 0) {
                product[slot] = x[unit]
                continue
            }
            var sum = 0.0
            a.forEachInColumn(basicIndex[slot]) { row, value -> sum += value * x[row] }
            product[slot] = sum
        }
    } else {
        for (slot in 0 until n) {
            val multiplier = x[slot]
            if (multiplier == 0.0) continue
            val unit = unitRows?.get(slot) ?: -1
            if (unit >= 0) {
                product[unit] += multiplier
                continue
            }
            a.forEachInColumn(basicIndex[slot]) { row, value ->
                product[row] += value * multiplier
            }
        }
    }
    var residual = 0.0
    var scale = 0.0
    for (i in 0 until n) {
        residual = max(residual, abs(product[i] - rhs[i]))
        scale = max(scale, max(abs(product[i]), abs(rhs[i])))
    }
    return BasisSolveQuality(residual, residual / max(1.0, scale))
}
