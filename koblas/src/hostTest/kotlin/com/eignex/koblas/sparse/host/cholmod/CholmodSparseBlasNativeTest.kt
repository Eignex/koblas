// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.BackendExecution
import com.eignex.koblas.F64RouteQuery
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.PreparedSparseProductKind
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
    private val cholmod = CholmodSparseBlas(
        level2Min = 0,
        preparedGemvMin = 0,
        preparedGemmMin = 0,
        preparedSparseProductMin = 0,
    ).also {
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
    fun `prepared routes report their independent gates`() {
        val defaults = CholmodSparseBlas()
        val gemv = defaults.route(F64RouteQuery.PreparedSparseProduct(1_000, PreparedSparseProductKind.GEMV))!!
        val gemm = defaults.route(F64RouteQuery.PreparedSparseProduct(1_000, PreparedSparseProductKind.DENSE_GEMM))!!
        assertEquals(BackendExecution.PORTABLE, gemv.execution)
        assertEquals(BackendExecution.PORTABLE, gemm.execution)
        assertTrue(gemv.gate?.minimum != gemm.gate?.minimum)
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

    @Test
    fun `a prepared descriptor reuses one sparse snapshot and closes`() {
        val rng = Random(20261011)
        val source = sparse(9, 6, rng)
        val expectedSource = F64SparseMatrix.wrap(
            source.rows,
            source.cols,
            source.copyColumnPointers(),
            source.copyRowIndices(),
            source.values.copyOf(),
        )
        val prepared = cholmod.prepare(source)
        source.values.fill(0.0)

        repeat(3) { iteration ->
            val x = DoubleArray(6) { rng.nextDouble(-1.0, 1.0) }
            val expected = F64ReferenceSparseLinearAlgebra.gemv(expectedSource, x)
            val actual = DoubleArray(9)
            prepared.gemv(1.0, x, 0.0, actual)
            assertClose(expected, actual, "iteration=$iteration", tolerance = 1e-12)
        }
        val right = sparse(6, 4, rng)
        assertEquals(F64ReferenceSparseLinearAlgebra.gemm(expectedSource, right), prepared.gemm(right))
        prepared.close()
        prepared.close()
        assertFailsWith<IllegalStateException> {
            prepared.gemv(1.0, DoubleArray(6), 0.0, DoubleArray(9))
        }
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
