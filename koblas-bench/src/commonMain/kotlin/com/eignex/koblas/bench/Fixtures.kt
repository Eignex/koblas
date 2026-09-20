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

    /**
     * A sparse operand of the named support distribution.
     *
     * Where the stored entries are costs as much as how many there are, so a case names the distribution
     * beside the density and the two are independent. `uniform` scatters the same count over every column;
     * `banded` keeps a column's entries next to the diagonal, which is where a discretisation puts them and
     * where a scatter's destination stays resident; `skewed` gives an eighth of the columns most of the
     * entries; `empty` leaves a quarter of the columns with nothing stored; and `mixed` alternates short
     * columns with long ones. The stored count per column is what changes, never the shape or the
     * requested logical work.
     */
    @Suppress("LongParameterList") // the shape, the density, the distribution and the triangle flags
    fun sparse(
        rows: Int,
        cols: Int,
        density: Double,
        operand: Int,
        triangular: Boolean = false,
        lower: Boolean = true,
        support: String = "uniform",
    ): SparseMatrix {
        val rowIndices = ArrayList<Int>()
        val columnIndices = ArrayList<Int>()
        val values = ArrayList<Double>()
        val average = ((rows * density) + 0.5).toInt().coerceIn(1, rows)
        for (j in 0 until cols) {
            val count = countFor(support, average, rows, j)
            val columnSeed = FIXTURE_SEED xor ((operand * 65537L + j) * -7046029254386353131L)
            val candidates = Array(rows) { row ->
                Candidate(key(support, rows, cols, j, row, mix(columnSeed + row * -7046029254386353131L)), row)
            }
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

    /** How many entries one column of this distribution holds, around [average]. */
    private fun countFor(support: String, average: Int, rows: Int, column: Int): Int = when (support) {
        "skewed" -> if (column % 8 == 0) (average * 6).coerceAtMost(rows) else (average / 4).coerceAtLeast(1)
        "empty" -> if (column % 4 == 0) 0 else average
        "mixed" -> if (column % 2 == 0) (average / 8).coerceAtLeast(1) else (average * 4).coerceAtMost(rows)
        else -> average
    }

    /**
     * The key a row is selected by, which is a hash except where the band decides the order.
     *
     * The lowest keys are taken, so putting the distance from the diagonal in the high bits and the hash in
     * the low ones makes a banded column fall next to its diagonal without changing how many entries it
     * holds or how they are drawn. The distance is at most the row count, which the case grammar bounds well
     * below what the shift can carry, so the two halves never run into each other.
     */
    @Suppress("LongParameterList") // the distribution, the shape, the position and the hash
    private fun key(support: String, rows: Int, cols: Int, column: Int, row: Int, hash: Long): Long {
        if (support != "banded") return hash
        val centre = if (cols <= 1) 0L else column.toLong() * (rows - 1) / (cols - 1)
        val distance = (row - centre).let { if (it < 0) -it else it }
        return (distance shl BAND_SHIFT) or (hash ushr (Long.SIZE_BITS - BAND_SHIFT))
    }

    /** Bits left for the hash under the distance, which is every bit the distance itself cannot need. */
    private const val BAND_SHIFT = 40

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
