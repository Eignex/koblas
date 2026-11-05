package com.eignex.koblas.bench

import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
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
    fun `sparse triangular fixtures retain every diagonal across support distributions`() {
        for (order in listOf(3, 9)) {
            for (support in listOf("uniform", "banded", "skewed", "empty", "mixed")) {
                for (lower in listOf(true, false)) {
                    for (density in listOf(0.0, 1.0 / 9.0, 2.0 / 3.0, 1.0)) {
                        for (operand in 1..2) {
                            val matrix = Fixtures.sparse(
                                order, order, density, operand, triangular = true, lower = lower, support = support,
                            )

                            for (column in 0 until matrix.cols) {
                                assertTrue(
                                    matrix[column, column] in 2.0..3.0,
                                    "order=$order $support lower=$lower density=$density operand=$operand column=$column",
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Each distribution puts a different number of entries in a column at one density, which is what a
     * measurement over them is comparing.
     */
    @Test
    fun `support distributions differ in what each column stores`() {
        val counts = listOf("uniform", "banded", "skewed", "empty", "mixed").associateWith { support ->
            val matrix = Fixtures.sparse(32, 8, 0.25, 3, support = support)
            (0 until matrix.cols).map { matrix.copyColumnPointers()[it + 1] - matrix.copyColumnPointers()[it] }
        }
        assertEquals(List(8) { 8 }, counts.getValue("uniform"))
        assertEquals(List(8) { 8 }, counts.getValue("banded"))
        assertEquals(listOf(32, 2, 2, 2, 2, 2, 2, 2), counts.getValue("skewed"))
        assertEquals(listOf(0, 8, 8, 8, 0, 8, 8, 8), counts.getValue("empty"))
        assertEquals(listOf(1, 32, 1, 32, 1, 32, 1, 32), counts.getValue("mixed"))
        val banded = Fixtures.sparse(32, 8, 0.0625, 3, support = "banded")
        for (j in 0 until banded.cols) {
            val centre = j * 31 / 7
            banded.forEachInColumn(j) { row, _ ->
                assertTrue(kotlin.math.abs(row - centre) <= 8, "column $j stored row $row away from $centre")
            }
        }
    }

    @Test
    fun `fixtures have stable golden digests`() {
        assertEquals("173cc3a80546954d", digest(Fixtures.vector(16, 1)))
        val sparse = Fixtures.sparse(17, 5, 0.2, 2)
        assertEquals("e6ac2de9cae9ebe8", digest(sparse.values))
        assertEquals(listOf(0, 3, 6, 9, 12, 15), sparse.copyColumnPointers().toList())
    }
}
