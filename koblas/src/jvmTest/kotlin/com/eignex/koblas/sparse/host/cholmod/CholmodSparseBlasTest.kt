package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@Category(HostLibraryTest::class)
class CholmodSparseBlasTest {

    // Gated at zero so these small operands reach the library instead of the portable routines.
    private val cholmod = CholmodSparseBlas(level2Min = 0)

    private fun requireCholmod() {
        Assume.assumeTrue("CHOLMOD is not installed; conformance cannot run", cholmod.isAvailable)
    }

    private fun sparse(rows: Int, cols: Int, rng: Random, density: Double = 0.3): F64SparseMatrix {
        val columns = List(cols) { j ->
            (0 until rows).mapNotNull { i ->
                if (rng.nextDouble() < density) i to rng.nextDouble(-1.0, 1.0) else null
            }
        }
        return F64SparseMatrix.ofColumns(rows, cols, columns)
    }

    @Test
    fun `it resolves and fills the sparse matrix half alone`() {
        requireCholmod()
        assertEquals("cholmod", cholmod.name)
        assertEquals(null, cholmod.unavailableReason)
    }

    /**
     * Portable by decision rather than by omission: a single right-hand side cannot amortise the call, and it
     * measured slower than the portable loop at every size. The answers still have to agree.
     */
    @Test
    fun `gemv agrees with the portable product in both directions`() {
        requireCholmod()
        val rng = Random(20260930)
        for (transpose in booleanArrayOf(false, true)) {
            val a = sparse(9, 6, rng)
            val x = randomVector(if (transpose) 9 else 6, rng)
            val y = randomVector(if (transpose) 6 else 9, rng)

            val expected = y.copyOf().also { F64ReferenceSparseLinearAlgebra.gemv(0.75, a, x, -0.5, it, transpose) }
            val actual = y.copyOf().also { cholmod.gemv(0.75, a, x, -0.5, it, transpose) }

            assertClose(expected, actual, "transpose=$transpose", tolerance = 1e-12)
        }
    }

    @Test
    fun `gemm agrees with the portable product over several right-hand sides`() {
        requireCholmod()
        val rng = Random(20260931)
        for (transposeA in booleanArrayOf(false, true)) {
            val a = sparse(7, 5, rng)
            val m = if (transposeA) 5 else 7
            val k = if (transposeA) 7 else 5
            val b = randomMatrix(k, 4, rng)
            val c = randomMatrix(m, 4, rng)

            val expected = F64DenseMatrix.wrap(m, 4, c.data.copyOf())
            F64ReferenceSparseLinearAlgebra.gemm(0.5, a, transposeA, b, false, 2.0, expected)
            val actual = F64DenseMatrix.wrap(m, 4, c.data.copyOf())
            cholmod.gemm(0.5, a, transposeA, b, false, 2.0, actual)

            assertClose(expected, actual, "transposeA=$transposeA", tolerance = 1e-12)
        }
    }

    /** The routine takes the sparse operand on the left, so the other side has to stay portable and agree. */
    @Test
    fun `a product from the right agrees with the portable one`() {
        requireCholmod()
        val rng = Random(20260932)
        val a = sparse(5, 4, rng)
        val b = randomMatrix(6, 5, rng)
        val c = randomMatrix(6, 4, rng)

        val expected = F64DenseMatrix.wrap(6, 4, c.data.copyOf())
        F64ReferenceSparseLinearAlgebra.gemm(1.0, a, false, b, false, 0.0, expected, right = true)
        val actual = F64DenseMatrix.wrap(6, 4, c.data.copyOf())
        cholmod.gemm(1.0, a, false, b, false, 0.0, actual, right = true)

        assertClose(expected, actual, "right", tolerance = 1e-12)
    }

    /** Portable by decision too: `cholmod_ssmult` fell further behind as the operands grew. */
    @Test
    fun `the sparse product agrees with the portable one`() {
        requireCholmod()
        val rng = Random(20260934)
        for (shape in listOf(Triple(1, 1, 1), Triple(7, 5, 4), Triple(12, 3, 9))) {
            val left = sparse(shape.first, shape.second, rng)
            val right = sparse(shape.second, shape.third, rng)

            val expected = F64ReferenceSparseLinearAlgebra.gemm(left, right)
            val actual = cholmod.gemm(left, right)

            assertEquals(expected.rows, actual.rows, "shape=$shape rows")
            assertEquals(expected.cols, actual.cols, "shape=$shape cols")
            for (i in 0 until expected.rows) {
                for (j in 0 until expected.cols) {
                    assertClose(expected[i, j], actual[i, j], "shape=$shape entry $i $j", tolerance = 1e-12)
                }
            }
        }
    }

    @Test
    fun `the sparse product keeps its rows ascending within a column`() {
        requireCholmod()
        val rng = Random(20260935)
        val product = cholmod.gemm(sparse(14, 9, rng, density = 0.5), sparse(9, 8, rng, density = 0.5))

        for (j in 0 until product.cols) {
            var previous = -1
            product.forEachInColumn(j) { i, _ ->
                assertTrue(i > previous, "column $j has $i after $previous")
                previous = i
            }
        }
    }

    @Test
    fun `the routines it does not carry answer portably`() {
        requireCholmod()
        val rng = Random(20260933)
        val triangle = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 2.0, 2 to 1.0), listOf(1 to 3.0), listOf(2 to 4.0)),
        )
        val b = randomVector(3, rng)

        val expected = b.copyOf().also { F64ReferenceSparseLinearAlgebra.trsv(triangle, it, lower = true) }
        val actual = b.copyOf().also { cholmod.trsv(triangle, it, lower = true) }

        assertClose(expected, actual, "trsv", tolerance = 1e-12)
        assertTrue(cholmod.transpose(triangle).nnz == triangle.nnz, "the transpose keeps every stored entry")
    }
}
