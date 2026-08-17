package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.SparseOrdering
import com.eignex.koblas.sparse.SparseSymbolic
import com.eignex.koblas.sparse.analyze
import kotlinx.benchmark.*

/**
 * The symbolic phase on its own, which the fill-reducing ordering dominates. Both shapes have `O(n)`
 * nonzeros, so anything here that grows faster than `n` is the ordering and not the matrix.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class SymbolicBenchmark {
    @Param("500", "2000", "8000")
    var n: Int = 0

    private lateinit var band: SparseMatrix
    private lateinit var grid: SparseMatrix

    @Setup
    fun setup() {
        band = bandUpperTriangle(n)
        grid = gridUpperTriangle(n)
    }

    @Benchmark
    fun bandMinimumDegree(): SparseSymbolic = band.analyze()

    @Benchmark
    fun bandNatural(): SparseSymbolic = band.analyze(SparseOrdering.Natural)

    @Benchmark
    fun gridMinimumDegree(): SparseSymbolic = grid.analyze()
}

/** The upper triangle of a tridiagonal band of dimension [n]. */
private fun bandUpperTriangle(n: Int): SparseMatrix {
    val rowIdx = IntArray(2 * n - 1)
    val colIdx = IntArray(2 * n - 1)
    val values = DoubleArray(2 * n - 1)
    var k = 0
    for (j in 0 until n) {
        if (j > 0) {
            rowIdx[k] = j - 1
            colIdx[k] = j
            values[k] = -1.0
            k++
        }
        rowIdx[k] = j
        colIdx[k] = j
        values[k] = 4.0
        k++
    }
    return SparseMatrix.ofTriplets(n, n, rowIdx, colIdx, values)
}

/** The upper triangle of a 5-point Laplacian on the squarest grid with at most [n] points. */
private fun gridUpperTriangle(n: Int): SparseMatrix {
    var side = 1
    while ((side + 1) * (side + 1) <= n) side++
    val points = side * side
    val rowIdx = ArrayList<Int>()
    val colIdx = ArrayList<Int>()
    val values = ArrayList<Double>()
    for (y in 0 until side) {
        for (x in 0 until side) {
            val i = y * side + x
            rowIdx.add(i)
            colIdx.add(i)
            values.add(8.0)
            if (x + 1 < side) {
                rowIdx.add(i)
                colIdx.add(i + 1)
                values.add(-1.0)
            }
            if (y + 1 < side) {
                rowIdx.add(i)
                colIdx.add(i + side)
                values.add(-1.0)
            }
        }
    }
    return SparseMatrix.ofTriplets(points, points, rowIdx.toIntArray(), colIdx.toIntArray(), values.toDoubleArray())
}
