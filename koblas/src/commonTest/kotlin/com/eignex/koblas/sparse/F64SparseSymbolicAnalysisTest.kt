package com.eignex.koblas.sparse

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * A symbolic analysis reused across numeric factorizations has to produce what a fresh factorization would,
 * and has to refuse a matrix whose structure it did not analyze rather than fill a factor from the wrong
 * column bounds.
 */
class F64SparseSymbolicAnalysisTest {

    private val portable = F64ReferenceSparseDecompositions

    @Test
    fun `a reused Cholesky analysis factors changed values the way a fresh call does`() {
        val a = sparseSymmetricConformanceSystem(24, Random(20260921))
        val changed = withRescaledValues(a, offDiagonal = 0.4, diagonal = 1.7)

        portable.analyzeCholesky(a).use { analysis ->
            analysis.factor(changed).use { reused ->
                portable.cholesky(changed).use { fresh ->
                    assertEquals(fresh.l, reused.l, "the reused analysis produced a different factor")
                }
            }
        }
    }

    @Test
    fun `a reused LDL analysis factors changed values the way a fresh call does`() {
        val a = indefiniteConformanceSystem(20, Random(20260922))
        val changed = withRescaledValues(a, offDiagonal = 0.4, diagonal = 1.7)

        portable.analyzeQuasiDefiniteLdl(a).use { analysis ->
            analysis.factor(changed).use { reused ->
                portable.quasiDefiniteLdl(changed).use { fresh ->
                    assertEquals(fresh.l, reused.l, "the reused analysis produced a different factor")
                    assertClose(fresh.d, reused.d, "diagonal factor", tolerance = 0.0)
                }
            }
        }
    }

    @Test
    fun `a reused QR analysis factors changed values the way a fresh call does`() {
        val a = sparseQrConformanceSystem(40, 15, Random(20260923))
        val changed = withRescaledValues(a, offDiagonal = 0.4, diagonal = 1.7)

        portable.analyzeQr(a).use { analysis ->
            analysis.factor(changed).use { reused ->
                portable.qr(changed).use { fresh ->
                    assertEquals(fresh.r, reused.r, "the reused analysis produced a different factor")
                    assertEquals(fresh.rank, reused.rank, "rank")
                }
            }
        }
    }

    /** The reuse is worth having only if a solve through it answers the changed system, not the analyzed one. */
    @Test
    fun `a reused Cholesky analysis solves the system it was handed`() {
        val a = sparseSymmetricConformanceSystem(18, Random(20260924))
        val changed = withRescaledValues(a, offDiagonal = 0.4, diagonal = 1.7)
        val b = DoubleArray(18) { it * 0.25 - 1.0 }
        val expected = portable.cholesky(changed).use { it.solve(b) }

        val actual = portable.analyzeCholesky(a).use { analysis -> analysis.factor(changed).use { it.solve(b) } }

        assertClose(expected, actual, "solution", tolerance = 1e-12)
    }

    @Test
    fun `the Cholesky analysis rejects a matrix of another pattern`() {
        val a = sparseSymmetricConformanceSystem(12, Random(20260925))
        val other = sparseSymmetricConformanceSystem(12, Random(20260926))

        portable.analyzeCholesky(a).use { analysis ->
            val failure = assertFailsWith<IllegalArgumentException> { analysis.factor(other) }
            assertTrue(failure.message.orEmpty().contains("does not match"), failure.message.orEmpty())
        }
    }

    @Test
    fun `the LDL analysis rejects a matrix of another pattern`() {
        val a = indefiniteConformanceSystem(12, Random(20260927))
        val other = indefiniteConformanceSystem(12, Random(20260928))

        portable.analyzeQuasiDefiniteLdl(a).use { analysis ->
            assertFailsWith<IllegalArgumentException> { analysis.factor(other) }
        }
    }

    @Test
    fun `the QR analysis rejects a matrix of another pattern`() {
        val a = sparseQrConformanceSystem(30, 10, Random(20260929))
        val other = sparseQrConformanceSystem(30, 10, Random(20260930))

        portable.analyzeQr(a).use { analysis ->
            assertFailsWith<IllegalArgumentException> { analysis.factor(other) }
        }
    }

    @Test
    fun `a closed analysis rejects further work and repeated close is safe`() {
        val a = sparseSymmetricConformanceSystem(6, Random(20260931))
        val analysis = portable.analyzeCholesky(a)

        analysis.close()
        analysis.close()

        assertFailsWith<IllegalStateException> { analysis.factor(a) }
    }

    /** A provider that has no reusable structure of its own still owes the caller the pattern check. */
    @Test
    fun `the seam default checks the pattern around an ordinary factorization`() {
        val provider = object : F64SparseCholesky {
            override val name: String get() = "counting"
            var factorizations = 0

            override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization {
                factorizations++
                return portable.cholesky(a)
            }
        }
        val a = sparseSymmetricConformanceSystem(8, Random(20260932))
        val other = sparseSymmetricConformanceSystem(8, Random(20260933))

        provider.analyzeCholesky(a).use { analysis ->
            analysis.factor(a).close()
            assertFailsWith<IllegalArgumentException> { analysis.factor(other) }
        }

        assertEquals(1, provider.factorizations, "the rejected matrix must not reach the numeric pass")
    }

    /**
     * The analyzed pattern is copied, so a caller reusing the arrays it built the matrix from cannot move the
     * bounds the numeric pass will fill against.
     */
    @OptIn(UnsafeKoblasApi::class)
    @Test
    fun `the analysis keeps its own copy of the pattern`() {
        val a = sparseSymmetricConformanceSystem(10, Random(20260934))
        val analysis = portable.analyzeCholesky(a)

        a.rowIdx[a.rowIdx.size - 1] = 0

        assertFailsWith<IllegalArgumentException> { analysis.factor(a) }
        analysis.close()
    }

    /** Same structure, different numbers: a diagonal grown and off-diagonals shrunk stay factorable. */
    private fun withRescaledValues(a: F64SparseMatrix, offDiagonal: Double, diagonal: Double): F64SparseMatrix {
        val colPtr = a.copyColumnPointers()
        val rowIdx = a.copyRowIndices()
        val values = DoubleArray(rowIdx.size)
        for (j in 0 until a.cols) {
            for (p in colPtr[j] until colPtr[j + 1]) {
                values[p] = a.values[p] * if (rowIdx[p] == j) diagonal else offDiagonal
            }
        }
        return F64SparseMatrix.wrap(a.rows, a.cols, colPtr, rowIdx, values)
    }
}
