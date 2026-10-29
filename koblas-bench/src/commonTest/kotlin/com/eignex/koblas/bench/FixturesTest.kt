package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertTrue

class FixturesTest {
    @Test
    fun `dense triangular fixtures bound the off diagonal row norm at every order`() {
        for (order in listOf(1, 31, 257)) {
            for (lower in listOf(true, false)) {
                val matrix = Fixtures.triangular(order, 1, lower)

                val maximum = (0 until order).maxOf { row ->
                    (0 until order).filter { it != row }.sumOf { column -> kotlin.math.abs(matrix[row, column]) }
                }

                assertTrue(maximum < 0.5, "order=$order lower=$lower offDiagonalNorm=$maximum")
            }
        }
    }

    @Test
    fun `a lower triangular column retains its diagonal when selected rows are filtered out`() {
        val matrix = Fixtures.sparse(3, 3, 2.0 / 3.0, 1, triangular = true, lower = true)

        assertTrue(matrix[2, 2] in 2.0..3.0, "the final column must store its diagonal")
    }

    @Test
    fun `sparse triangular fixtures retain every diagonal across support distributions`() {
        for (support in listOf("uniform", "banded", "skewed", "empty", "mixed")) {
            for (lower in listOf(true, false)) {
                for (density in listOf(0.0, 1.0 / 9.0, 2.0 / 3.0, 1.0)) {
                    for (operand in 1..2) {
                        val matrix = Fixtures.sparse(
                            9, 9, density, operand, triangular = true, lower = lower, support = support,
                        )

                        for (column in 0 until matrix.cols) {
                            assertTrue(
                                matrix[column, column] in 2.0..3.0,
                                "$support lower=$lower density=$density operand=$operand column=$column",
                            )
                        }
                    }
                }
            }
        }
    }
}
