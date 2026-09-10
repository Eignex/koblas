package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector

internal const val FIXTURE_SEED: Long = 0x243f6a8885a308d3L

internal class PortableRandom(seed: Long) {
    private var state = seed

    fun nextLong(): Long {
        state += -7046029254386353131L
        var z = state
        z = (z xor (z ushr 30)) * -4658895280553007687L
        z = (z xor (z ushr 27)) * -7723592293110705685L
        return z xor (z ushr 31)
    }

    fun nextDouble(): Double = (nextLong().ushr(11).toDouble() * 2.220446049250313e-16) - 1.0
}

internal object Fixtures {
    fun stream(operand: Int): PortableRandom = PortableRandom(FIXTURE_SEED xor (operand.toLong() * -7046029254386353131L))

    fun vector(size: Int, operand: Int): DoubleArray = DoubleArray(size).also { values ->
        val random = stream(operand)
        for (i in values.indices) values[i] = random.nextDouble()
    }

    fun matrix(rows: Int, cols: Int, operand: Int): DenseMatrix {
        val values = vector(rows * cols, operand)
        return DenseMatrix.ofColumns(Array(cols) { column -> values.copyOfRange(column * rows, (column + 1) * rows) })
    }

    fun triangular(order: Int, operand: Int, lower: Boolean): DenseMatrix {
        val matrix = matrix(order, order, operand)
        for (j in 0 until order) for (i in 0 until order) {
            if ((lower && i < j) || (!lower && i > j)) matrix[i, j] = 0.0
        }
        for (i in 0 until order) matrix[i, i] = 2.0 + kotlin.math.abs(matrix[i, i])
        return matrix
    }

    fun sparse(rows: Int, cols: Int, density: Double, operand: Int, triangular: Boolean = false, lower: Boolean = true): SparseMatrix {
        val rowIndices = ArrayList<Int>()
        val columnIndices = ArrayList<Int>()
        val values = ArrayList<Double>()
        val count = ((rows * density) + 0.5).toInt().coerceIn(1, rows)
        for (j in 0 until cols) {
            val columnSeed = FIXTURE_SEED xor ((operand * 65537L + j) * -7046029254386353131L)
            val candidates = Array(rows) { row -> Candidate(mix(columnSeed + row * -7046029254386353131L), row) }
            candidates.sortWith(compareBy<Candidate> { it.key }.thenBy { it.row })
            val selected = candidates.take(count).map { it.row }.sorted()
            val random = stream(operand * 104729 + j)
            for (row in selected) {
                if (triangular && ((lower && row < j) || (!lower && row > j))) continue
                rowIndices += row
                columnIndices += j
                values += if (triangular && row == j) 2.0 + kotlin.math.abs(random.nextDouble()) else random.nextDouble()
            }
            if (triangular && j < rows && rowIndices.lastOrNull() != j && selected.none { it == j }) {
                rowIndices += j
                columnIndices += j
                values += 2.0 + kotlin.math.abs(random.nextDouble())
            }
        }
        val order = rowIndices.indices.sortedWith(compareBy<Int> { columnIndices[it] }.thenBy { rowIndices[it] })
        return SparseMatrix.ofTriplets(
            rows, cols,
            IntArray(order.size) { rowIndices[order[it]] },
            IntArray(order.size) { columnIndices[order[it]] },
            DoubleArray(order.size) { values[order[it]] },
        )
    }

    fun sparseVector(size: Int, density: Double, operand: Int): SparseVector {
        val matrix = sparse(size, 1, density, operand)
        return SparseVector.of(size, matrix.copyRowIndices(), matrix.values)
    }

    private data class Candidate(val key: Long, val row: Int)

    private fun mix(input: Long): Long {
        var z = input
        z = (z xor (z ushr 30)) * -4658895280553007687L
        z = (z xor (z ushr 27)) * -7723592293110705685L
        return z xor (z ushr 31)
    }
}

internal fun digest(values: DoubleArray): String {
    var hash = -3750763034362895579L
    for (value in values) {
        var bits = value.toBits()
        repeat(8) {
            hash = (hash xor (bits and 0xffL)) * 1099511628211L
            bits = bits ushr 8
        }
    }
    return hash.toULong().toString(16).padStart(16, '0')
}
