package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** The sparse symmetric factorization: the symbolic phase, the numeric one, and the three pivot policies. */
class SparseLdlTest {

    /** A symmetric positive-definite matrix, stored in full, with [density] off-diagonal fill. */
    private fun spd(n: Int, rng: Random, density: Double = 0.2): SparseMatrix {
        val entries = Array(n) { HashMap<Int, Double>() }
        for (i in 0 until n) entries[i][i] = n.toDouble()
        for (i in 0 until n) {
            for (j in 0 until i) {
                if (rng.nextDouble() < density) {
                    val v = rng.nextDouble(-1.0, 1.0)
                    entries[i][j] = v
                    entries[j][i] = v
                }
            }
        }
        return SparseMatrix.ofColumns(n, n, List(n) { j -> entries[j].map { (i, v) -> i to v } })
    }

    /** `A·x`, computed straight from the CSC arrays so no seam is involved in checking a seam. */
    private fun multiply(a: SparseMatrix, x: DoubleArray): DoubleArray {
        val y = DoubleArray(a.rows)
        for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> y[i] += v * x[j] }
        return y
    }

    @Test
    fun `the analysis predicts a pattern the numeric phase fits into`() {
        val rng = Random(20260810)
        for (n in intArrayOf(1, 2, 5, 17, 64)) {
            val a = spd(n, rng)
            val symbolic = a.analyze()
            assertEquals(n, symbolic.n)
            assertEquals(n + 1, symbolic.columnPointers.size)
            assertTrue(symbolic.columnPointers.asList() == symbolic.columnPointers.sorted(), "n=$n: not ascending")
            // Filling it is what proves the counts: the numeric phase throws if a column needs more room.
            val f = symbolic.factorLdl(a)
            assertTrue(!f.singular, "n=$n: a dominant diagonal cannot be singular")
            assertEquals(symbolic.nnz + n, f.nnz, "n=$n: nnz should be L's entries plus D")
        }
    }

    @Test
    fun `the elimination tree is a forest with parents above their children`() {
        val a = spd(32, Random(4), density = 0.15)
        val parent = a.analyze().parent
        for (j in parent.indices) {
            val p = parent[j]
            assertTrue(p == -1 || p > j, "parent of $j is $p, which is not above it")
        }
    }

    @Test
    fun `the factorization solves the system it came from`() {
        val rng = Random(20260811)
        for (n in intArrayOf(1, 3, 9, 40)) {
            val a = spd(n, rng)
            val x = randomVector(n, rng)
            val b = multiply(a, x)
            val f = a.cholesky()
            assertClose(x, f.solve(b), "cholesky solve n=$n", tolerance = 1e-8)
            // Symmetric, so the transposed solve is the same system and must give the same answer.
            assertClose(x, f.solve(b, transpose = true), "transposed solve n=$n", tolerance = 1e-8)
        }
    }

    @Test
    fun `an indefinite matrix factors under the indefinite policy and not under strict`() {
        // [[1, 2], [2, 1]] is symmetric with eigenvalues 3 and -1: a real LDL, no Cholesky.
        val a = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 2.0, 1 to 1.0)),
        )
        assertFailsWith<IllegalArgumentException> { a.cholesky() }
        val f = a.ldl()
        assertTrue(!f.singular)
        val x = doubleArrayOf(0.5, -1.5)
        assertClose(x, f.solve(multiply(a, x)), "indefinite solve", tolerance = 1e-12)
        // det [[1,2],[2,1]] = 1 - 4 = -3, which only an indefinite factorization can report.
        assertClose(-3.0, f.determinant(), "indefinite determinant", tolerance = 1e-12)
    }

    @Test
    fun `regularize floors a pivot that strict would reject`() {
        val a = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0), listOf(1 to -0.5)),
        )
        assertFailsWith<IllegalArgumentException> { a.cholesky() }
        val floored = a.cholesky(SparseLdlPolicy.Regularize(minimumPivot = 1e-8))
        assertTrue(!floored.singular, "the floored pivot must produce a usable factorization")
        assertClose(1e-8, floored.determinant(), "det is 1 · the floored pivot", tolerance = 1e-20)
    }

    @Test
    fun `a zero pivot is singular whatever the policy`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf()))
        for (policy in listOf(SparseLdlPolicy.Indefinite, SparseLdlPolicy.Regularize())) {
            val f = a.ldl(policy)
            assertTrue(f.singular, "a zero column is singular under $policy")
            assertEquals(1, f.failedAt, "the second pivot is the one that failed")
            assertEquals(0.0, f.determinant())
        }
    }

    /** The point of the split: one analysis, many value sets. */
    @Test
    fun `one analysis factorizes every matrix with its pattern`() {
        val rng = Random(20260812)
        val first = spd(24, rng)
        val symbolic = first.analyze()
        // Same pattern, different values: rebuild with each stored entry perturbed.
        val second = SparseMatrix(
            first.rows,
            first.cols,
            first.colPtr.copyOf(),
            first.rowIdx.copyOf(),
            DoubleArray(first.values.size) { first.values[it] * 1.5 },
        )
        for (a in listOf(first, second)) {
            val x = randomVector(24, rng)
            val f = symbolic.factorLdl(a, SparseLdlPolicy.Strict)
            assertClose(x, f.solve(multiply(a, x)), "reused analysis", tolerance = 1e-8)
        }
    }

    @Test
    fun `a matrix outside the analysed pattern is rejected rather than silently truncated`() {
        val sparse = SparseMatrix.ofColumns(3, 3, listOf(listOf(0 to 4.0), listOf(1 to 4.0), listOf(2 to 4.0)))
        val denser = SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 4.0, 1 to 1.0), listOf(0 to 1.0, 1 to 4.0), listOf(2 to 4.0)),
        )
        assertFailsWith<IllegalArgumentException> { sparse.analyze().factorLdl(denser) }
    }

    @Test
    fun `a lower-only matrix is rejected instead of analysed as diagonal`() {
        val lowerOnly = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 4.0, 1 to 1.0), listOf(1 to 4.0)),
        )
        assertFailsWith<IllegalArgumentException> { lowerOnly.analyze() }
        // The same matrix stored in full analyses and factors.
        val full = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 4.0, 1 to 1.0), listOf(0 to 1.0, 1 to 4.0)),
        )
        assertTrue(!full.cholesky().singular)
    }

    @Test
    fun `the cholesky factor reproduces the matrix and refuses an indefinite one`() {
        val rng = Random(20260813)
        val a = spd(12, rng)
        val f = a.cholesky() as SparseLdl
        val l = f.choleskyFactor()
        val perm = f.symbolic.permutation
        // L·Lᵀ is P·A·Pᵀ, not A — the factor is of the matrix that was eliminated, which the analysis
        // reordered. Comparing against A directly is the mistake this asserts against.
        val dense = Array(12) { DoubleArray(12) }
        for (j in 0 until 12) l.forEachInColumn(j) { i, v -> dense[i][j] = v }
        for (i in 0 until 12) {
            for (j in 0 until 12) {
                var s = 0.0
                for (p in 0 until 12) s += dense[i][p] * dense[j][p]
                val expected = a[perm[i], perm[j]]
                assertTrue(abs(s - expected) < 1e-9, "L·Lᵀ differs from P·A·Pᵀ at ($i,$j): $s vs $expected")
            }
        }
        val indefinite = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 2.0, 1 to 1.0)),
        )
        assertFailsWith<IllegalArgumentException> { (indefinite.ldl() as SparseLdl).choleskyFactor() }
    }

    @Test
    fun `fill stays near the input on a banded matrix`() {
        // A tridiagonal matrix has no fill at all under a symmetric factorization: its tree is a path.
        val n = 200
        val columns = List(n) { j ->
            buildList {
                if (j > 0) add(j - 1 to -1.0)
                add(j to 4.0)
                if (j < n - 1) add(j + 1 to -1.0)
            }
        }
        val a = SparseMatrix.ofColumns(n, n, columns)
        val symbolic = a.analyze()
        assertEquals(n - 1, symbolic.nnz, "a tridiagonal L holds exactly one subdiagonal")
        val f = a.cholesky()
        val x = randomVector(n, Random(9))
        assertClose(x, f.solve(multiply(a, x)), "tridiagonal solve", tolerance = 1e-9)
    }

    @Test
    fun `a non-square matrix is rejected`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))
        assertFailsWith<IllegalArgumentException> { a.analyze() }
        assertFailsWith<IllegalArgumentException> { a.ldl() }
    }
}
