package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseMatrix
import kotlin.test.*

class F64SparseLuAnalysisTest {
    private class RepeatedProvider : F64RepeatedSparseLu {
        override val name: String get() = "repeated"
        var factors = 0
        var refactors = 0

        override fun factor(a: F64SparseMatrix): F64SparseFactorization {
            factors++
            return F64ReferenceSparseLinearAlgebra.factor(a)
        }

        override fun refactor(previous: F64SparseFactorization, a: F64SparseMatrix): F64SparseFactorization {
            refactors++
            previous.close()
            return F64ReferenceSparseLinearAlgebra.factor(a)
        }
    }

    @Test
    fun `analysis accepts changed values with the same pattern`() {
        val provider = RepeatedProvider()
        val first = matrix(doubleArrayOf(4.0, 1.0, 3.0))
        val changed = matrix(doubleArrayOf(5.0, 2.0, 6.0))

        provider.analyze(first).use { analysis ->
            var factor = analysis.factor(first)
            factor = analysis.refactor(factor, changed)
            factor.use {
                val solution = it.solve(doubleArrayOf(1.0, 1.0))
                assertEquals(0.2, solution[0], 1e-12)
                assertEquals(0.1, solution[1], 1e-12)
            }
        }

        assertEquals(1, provider.factors)
        assertEquals(1, provider.refactors)
    }

    @Test
    fun `analysis rejects another pattern before numeric work`() {
        val provider = RepeatedProvider()
        val first = matrix(doubleArrayOf(4.0, 1.0, 3.0))
        val changed = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 3.0)))
        val analysis = provider.analyze(first)

        val failure = assertFailsWith<IncompatibleSparsePatternException> { analysis.factor(changed) }

        assertSame(changed, failure.actual)
        assertEquals(0, provider.factors)
        analysis.close()
    }

    @Test
    fun `closed analysis rejects numeric work and repeated close is safe`() {
        val provider = RepeatedProvider()
        val matrix = matrix(doubleArrayOf(4.0, 1.0, 3.0))
        val analysis = provider.analyze(matrix)

        analysis.close()
        analysis.close()

        assertFailsWith<IllegalStateException> { analysis.factor(matrix) }
    }

    private fun matrix(values: DoubleArray): F64SparseMatrix = F64SparseMatrix.ofColumns(
        2,
        2,
        listOf(listOf(0 to values[0], 1 to values[1]), listOf(1 to values[2])),
    )
}

internal fun assertSymbolicAnalysisReuses(provider: F64RepeatedSparseLu) {
    val first = F64SparseMatrix.ofColumns(
        2,
        2,
        listOf(listOf(0 to 2.0, 1 to 1.0), listOf(1 to 3.0)),
    )
    val changed = F64SparseMatrix.ofColumns(
        2,
        2,
        listOf(listOf(0 to 4.0, 1 to 2.0), listOf(1 to 5.0)),
    )
    val incompatible = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 5.0)))

    provider.analyze(first).use { analysis ->
        var factor = analysis.factor(first)
        factor = analysis.refactor(factor, changed)
        factor.use {
            val expected = F64ReferenceSparseLinearAlgebra.factor(changed).solve(doubleArrayOf(8.0, 12.0))
            val actual = it.solve(doubleArrayOf(8.0, 12.0))
            for (i in expected.indices) assertEquals(expected[i], actual[i], 1e-12, "solution at $i")
            assertFailsWith<IncompatibleSparsePatternException> { analysis.refactor(it, incompatible) }
        }
    }
}
