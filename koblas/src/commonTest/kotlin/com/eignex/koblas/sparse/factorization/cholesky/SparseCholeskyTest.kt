package com.eignex.koblas.sparse.factorization.cholesky

import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.randomVector
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.cholesky
import com.eignex.koblas.sparse.gemv
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

class SparseCholeskyTest {

    /**
     * A symmetric positive-definite matrix as its stored lower triangle and as the full dense twin. Made
     * diagonally dominant, which is sufficient for positive definiteness and keeps the pattern sparse.
     */
    private fun spd(n: Int, rng: Random, density: Double = 0.3): Pair<F64SparseMatrix, F64DenseMatrix> {
        val dense = F64DenseMatrix.zero(n, n)
        val offDiagonal = Array(n) { HashMap<Int, Double>() }
        for (j in 0 until n) {
            for (i in j + 1 until n) {
                if (rng.nextDouble() >= density) continue
                val v = rng.nextDouble(-1.0, 1.0)
                offDiagonal[j][i] = v
                dense[i, j] = v
                dense[j, i] = v
            }
        }
        val columns = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            var weight = 0.0
            for (i in 0 until n) weight += kotlin.math.abs(dense[i, j])
            val d = weight + 1.0
            dense[j, j] = d
            val column = ArrayList<Pair<Int, Double>>()
            column.add(j to d)
            for (i in j + 1 until n) offDiagonal[j][i]?.let { column.add(i to it) }
            columns.add(column)
        }
        return F64SparseMatrix.ofColumns(n, n, columns) to dense
    }

    @Test
    fun `the solution reproduces the right-hand side`() {
        val rng = Random(20260903)
        val n = 12
        val (sparse, dense) = spd(n, rng)
        val b = randomVector(n, rng)

        val x = sparse.cholesky().solve(b)

        assertClose(b, dense.let { full -> koblas.gemv(full, x) }, "residual", tolerance = 1e-9)
    }

    @Test
    fun `every solve agrees with the dense Cholesky`() {
        val rng = Random(20260904)
        for (n in intArrayOf(1, 2, 5, 13)) {
            val (sparse, dense) = spd(n, rng)
            val b = randomVector(n, rng)

            val fromDense = koblas.solve(koblas.cholesky(dense), b)
            val fromSparse = sparse.cholesky().solve(b)

            assertClose(fromDense, fromSparse, "n=$n", tolerance = 1e-9)
        }
    }

    @Test
    fun `the factor times its transpose is the matrix it came from`() {
        val rng = Random(20260905)
        val n = 9
        val (sparse, dense) = spd(n, rng)

        val l = (F64ReferenceSparseLinearAlgebra.cholesky(sparse) as F64SparseUpLookingCholesky).l

        for (j in 0 until n) {
            for (i in j until n) {
                var s = 0.0
                for (k in 0..j) s += l[i, k] * l[j, k]
                assertClose(dense[i, j], s, "entry $i $j", tolerance = 1e-9)
            }
        }
    }

    @Test
    fun `the factor is lower triangular`() {
        val rng = Random(20260906)
        val n = 8
        val (sparse, _) = spd(n, rng)

        val l = (F64ReferenceSparseLinearAlgebra.cholesky(sparse) as F64SparseUpLookingCholesky).l

        for (j in 0 until n) {
            var aboveDiagonal = 0
            l.forEachInColumn(j) { i, _ -> if (i < j) aboveDiagonal++ }
            assertEquals(0, aboveDiagonal, "column $j stores $aboveDiagonal entries above the diagonal")
        }
    }

    @Test
    fun `anything stored above the diagonal is ignored`() {
        val n = 4
        val (lowerOnly, _) = spd(n, Random(20260907))
        // The same lower triangle with NaN through the strict upper one, which is never read.
        val poisoned = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until j) column.add(i to Double.NaN)
            lowerOnly.forEachInColumn(j) { i, v -> column.add(i to v) }
            poisoned.add(column)
        }
        val b = randomVector(n, Random(20260908))

        val fromPoisoned = F64SparseMatrix.ofColumns(n, n, poisoned).cholesky().solve(b)

        assertClose(lowerOnly.cholesky().solve(b), fromPoisoned, "the upper triangle reached the answer", 0.0)
    }

    @Test
    fun `a matrix that is not positive definite names the column that failed`() {
        // Positive definite through column 1, then a diagonal too small to carry column 2.
        val a = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(
                listOf(0 to 4.0, 2 to 3.0),
                listOf(1 to 4.0),
                listOf(2 to 2.0),
            ),
        )

        val failure = assertFailsWith<NotPositiveDefinite> { a.cholesky() }

        assertEquals(2, failure.position)
    }

    @Test
    fun `an identity factors to itself`() {
        val n = 5
        val identity = F64SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })

        val f = F64ReferenceSparseLinearAlgebra.cholesky(identity)

        assertEquals(n, f.nnz, "an identity should fill nowhere")
        assertEquals(1.0, f.rcond)
        assertClose(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0), f.solve(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)), "solve")
    }

    @Test
    fun `the transposed solve is the same solve`() {
        val rng = Random(20260909)
        val n = 7
        val (sparse, _) = spd(n, rng)
        val b = randomVector(n, rng)
        val f = sparse.cholesky()

        assertClose(f.solve(b), f.solve(b, transpose = true), "a symmetric matrix has one solve", 0.0)
    }

    @Test
    fun `a solve may write into the vector it read`() {
        val rng = Random(20260910)
        val n = 6
        val (sparse, _) = spd(n, rng)
        val b = randomVector(n, rng)
        val f = sparse.cholesky()
        val expected = f.solve(b)

        val inPlace = b.copyOf()
        val returned = f.solveInto(inPlace, inPlace)

        assertSame(inPlace, returned)
        assertClose(expected, inPlace, "an aliased solve", 0.0)
    }

    @Test
    fun `it fills where the elimination tree says it must`() {
        // An arrow, dense in its last row, which factors with no fill at all in this ordering.
        val n = 6
        val columns = List(n) { j ->
            if (j == n - 1) listOf(j to n.toDouble()) else listOf(j to 4.0, n - 1 to 1.0)
        }
        val arrow = F64SparseMatrix.ofColumns(n, n, columns)

        val f = F64ReferenceSparseLinearAlgebra.cholesky(arrow)

        assertTrue(f.nnz == arrow.nnz, "an arrow pointing at its last row fills nowhere, got ${f.nnz}")
    }
}
