package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

@Category(HostLibraryTest::class)
class SpqrQrTest {
    private val spqr = SpqrQr()

    private fun requireSpqr() {
        Assume.assumeTrue("SPQR is not installed; conformance cannot run", spqr.isAvailable)
    }

    @Test
    fun `the least squares solution matches the portable factorization`() {
        requireSpqr()
        val rng = Random(20260827)
        for ((m, n) in listOf(1 to 1, 9 to 4, 60 to 25, 400 to 180)) {
            val a = tall(m, n, rng)
            val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }

            val native = assertNotNull(spqr.factor(a), "SPQR reported available but produced no factor")
            native.use {
                assertEquals(m, it.m, "${m}x$n rows")
                assertEquals(n, it.n, "${m}x$n cols")
                assertClose(
                    F64ReferenceSparseLinearAlgebra.qr(a).solve(b),
                    it.solve(b),
                    "${m}x$n",
                    tolerance = 1e-7,
                )
            }
        }
    }

    @Test
    fun `the solution satisfies the normal equations`() {
        requireSpqr()
        val rng = Random(20260901)
        val a = tall(120, 45, rng)
        val b = DoubleArray(120) { rng.nextDouble(-1.0, 1.0) }

        val x = assertNotNull(spqr.factor(a)).use { it.solve(b) }

        assertClose(transposeTimes(a, b), transposeTimes(a, times(a, x)), "normal equations", tolerance = 1e-7)
    }

    @Test
    fun `the natural ordering reaches the same solution as the default one`() {
        requireSpqr()
        val rng = Random(20260902)
        val a = tall(80, 30, rng)
        val b = DoubleArray(80) { rng.nextDouble(-1.0, 1.0) }
        val natural = SpqrQr(SpqrConfig(options = SpqrOptions(ordering = SpqrOrdering.NATURAL)))

        val reordered = assertNotNull(spqr.factor(a)).use { it.solve(b) }
        val asGiven = assertNotNull(natural.factor(a)).use { it.solve(b) }

        assertClose(reordered, asGiven, "ordering", tolerance = 1e-7)
    }

    @Test
    fun `a rank deficient matrix is reported as such`() {
        requireSpqr()
        // Column 2 repeats column 0, so SPQR's tolerance should find a rank of two.
        val a = F64SparseMatrix.ofColumns(
            4,
            3,
            listOf(listOf(0 to 1.0, 1 to 2.0), listOf(1 to 1.0, 2 to 3.0), listOf(0 to 1.0, 1 to 2.0)),
        )

        assertNotNull(spqr.factor(a)).use { qr ->
            assertEquals(2, qr.rank)
            assertTrue(qr.rankDeficient)
        }
    }

    @Test
    fun `rank detection can be switched off to match the reference arithmetic`() {
        requireSpqr()
        val a = F64SparseMatrix.ofColumns(
            4,
            3,
            listOf(listOf(0 to 1.0, 1 to 2.0), listOf(1 to 1.0, 2 to 3.0), listOf(0 to 1.0, 1 to 2.0)),
        )
        val exact = SpqrQr(SpqrConfig(options = SpqrOptions(rankTolerance = -1.0)))

        assertNotNull(exact.factor(a)).use { qr ->
            assertEquals(3, qr.rank)
        }
    }

    @Test
    fun `a closed factorization refuses to solve and closes idempotently`() {
        requireSpqr()
        val a = tall(20, 8, Random(20260903))
        val factorization = assertNotNull(spqr.factor(a))

        factorization.close()
        factorization.close()

        assertFailsWith<IllegalStateException> { factorization.solve(DoubleArray(20)) }
    }

    @Test
    fun `a wider than tall matrix is rejected before the library is reached`() {
        requireSpqr()
        val a = F64SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))

        assertFailsWith<IllegalArgumentException> { spqr.factor(a) }
    }
}

private fun tall(m: Int, n: Int, rng: Random): F64SparseMatrix {
    val columns = ArrayList<List<Pair<Int, Double>>>(n)
    for (j in 0 until n) {
        val rows = (listOf(j) + List(3) { rng.nextInt(m) }).distinct().sorted()
        columns.add(rows.map { row -> row to if (row == j) 2.0 + rng.nextDouble() else rng.nextDouble(-1.0, 1.0) })
    }
    return F64SparseMatrix.ofColumns(m, n, columns)
}

private fun times(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.rows)
    for (j in 0 until a.cols) a.forEachInColumn(j) { row, value -> y[row] += value * x[j] }
    return y
}

private fun transposeTimes(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.cols)
    for (j in 0 until a.cols) a.forEachInColumn(j) { row, value -> y[j] += value * x[row] }
    return y
}
