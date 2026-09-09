@file:Suppress("LongParameterList")

package com.eignex.koblas.dense

/**
 * Netlib GEMMTR arithmetic for one selected triangle. A non-transposed left operand scales B before each
 * column update; a transposed left operand scales the completed ordered dot. These placements are observably
 * different for overflow and non-finite values and are intentionally separate branches.
 */
internal fun orderedGemmtUpdate(
    vectorKernels: DenseVectorKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    beta: Double,
    c: DoubleArray,
    n: Int,
    k: Int,
    lower: Boolean,
) {
    if (!transposeA) scaleTriangle(vectorKernels, c, n, beta, lower)
    repeat(n) { column ->
        val from = if (lower) column else 0
        val until = if (lower) n else column + 1
        if (!transposeA) {
            repeat(k) { inner ->
                val bValue = if (transposeB) b[column + inner * ldb] else b[inner + column * ldb]
                val scaledB = alpha * bValue
                var row = from
                while (row < until) {
                    c[row + column * n] += scaledB * a[row + inner * lda]
                    row++
                }
            }
        } else {
            var row = from
            while (row < until) {
                var sum = 0.0
                repeat(k) { inner ->
                    val bValue = if (transposeB) b[column + inner * ldb] else b[inner + column * ldb]
                    sum += a[inner + row * lda] * bValue
                }
                val destination = row + column * n
                c[destination] = if (beta == 0.0) alpha * sum else alpha * sum + beta * c[destination]
                row++
            }
        }
    }
}
