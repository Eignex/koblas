// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.discoverBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import kotlin.random.Random
import kotlin.test.*

/** Checks the Kotlin/Native sparse BLAS binding against the portable implementation. */
class CholmodSparseBlasNativeTest {
    private val cholmod = CholmodSparseBlas(level2Min = 0).also {
        require(it.isAvailable) { "host CHOLMOD expected in the test environment" }
    }

    @Test
    fun `the native sparse matrix half resolves`() {
        discoverBackends()
        assertEquals("cholmod", cholmod.name)
        assertEquals(HOST_BACKEND_PRIORITY, cholmod.priority)
        assertNull(cholmod.unavailableReason)
        assertEquals("cholmod", koblas.sparseBlas.name)
    }

    @Test
    fun `dense products agree with the reference in both directions`() {
        val rng = Random(20261009)
        for (transposeA in booleanArrayOf(false, true)) {
            val a = sparse(11, 7, rng)
            val m = if (transposeA) a.cols else a.rows
            val k = if (transposeA) a.rows else a.cols
            val b = randomMatrix(k, 5, rng)
            val initial = randomMatrix(m, b.cols, rng)
            val expected = F64DenseMatrix.wrap(m, b.cols, initial.data.copyOf())
            val actual = F64DenseMatrix.wrap(m, b.cols, initial.data.copyOf())

            F64ReferenceSparseLinearAlgebra.gemm(0.75, a, transposeA, b, false, -0.25, expected)
            cholmod.gemm(0.75, a, transposeA, b, false, -0.25, actual)

            assertClose(expected, actual, "transposeA=$transposeA", tolerance = 1e-12)
        }
    }

    @Test
    fun `one right hand side remains portable and agrees`() {
        val rng = Random(20261010)
        val a = sparse(9, 6, rng)
        val x = DoubleArray(6) { rng.nextDouble(-1.0, 1.0) }
        val initial = DoubleArray(9) { rng.nextDouble(-1.0, 1.0) }
        val expected = initial.copyOf()
        val actual = initial.copyOf()

        F64ReferenceSparseLinearAlgebra.gemv(1.5, a, x, 0.5, expected)
        cholmod.gemv(1.5, a, x, 0.5, actual)

        assertClose(expected, actual, "gemv", tolerance = 1e-12)
    }

    private fun sparse(rows: Int, cols: Int, rng: Random): F64SparseMatrix = F64SparseMatrix.ofColumns(
        rows,
        cols,
        List(cols) {
            (0 until rows).mapNotNull { row ->
                if (rng.nextDouble() < 0.35) row to rng.nextDouble(-1.0, 1.0) else null
            }
        },
    )
}
