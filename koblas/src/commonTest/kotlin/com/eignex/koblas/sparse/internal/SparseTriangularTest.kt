package com.eignex.koblas.sparse.internal

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.internal.dense.PanelWork
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.*

class SparseTriangularTest {

    private fun triangle(n: Int, lower: Boolean, rng: Random, density: Double = 0.4): Pair<SparseMatrix, DenseMatrix> {
        val dense = DenseMatrix(n, n)
        val columns = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                val inTriangle = if (lower) i >= j else i <= j
                if (!inTriangle) continue
                val v = when {
                    i == j -> 2.0 + rng.nextDouble()
                    rng.nextDouble() < density -> rng.nextDouble(-1.0, 1.0)
                    else -> 0.0
                }
                if (v != 0.0) {
                    column.add(i to v)
                    dense[i, j] = v
                }
            }
            columns.add(column)
        }
        return SparseMatrix.ofColumns(n, n, columns) to dense
    }

    @Test
    fun `every direction agrees with the dense solve`() {
        val rng = Random(20260811)
        for (n in intArrayOf(1, 2, 6, 11)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    val (sparse, dense) = triangle(n, lower, rng)
                    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

                    val fromDense = b.copyOf()
                    koblas.trsv(dense, fromDense, lower, transpose)
                    val fromSparse = b.copyOf()
                    koblas.trsv(sparse, fromSparse, lower, transpose)

                    assertClose(
                        fromDense,
                        fromSparse,
                        "n=$n lower=$lower transpose=$transpose",
                        tolerance = 1e-9,
                    )
                }
            }
        }
    }

    @Test
    fun `every direction agrees with dense triangular multiplication`() {
        val rng = Random(20260901)
        for (n in intArrayOf(0, 1, 2, 6, 11)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (sparse, dense) = triangle(n, lower, rng)
                        val input = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
                        val expected = input.copyOf().also { koblas.trmv(dense, it, lower, transpose, unitDiag) }
                        val actual = input.copyOf().also { koblas.trmv(sparse, it, lower, transpose, unitDiag) }

                        assertClose(expected, actual, "n=$n lower=$lower transpose=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `matrix solves agree with the scalar reference across triangle flags`() {
        assertTriangularAgreesWithReference(solve = true)
    }

    @Test
    fun `matrix multiplies agree with the scalar reference across triangle flags`() {
        assertTriangularAgreesWithReference(solve = false)
    }

    private fun assertTriangularAgreesWithReference(solve: Boolean) {
        val rng = Random(20261013)
        val order = 32
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).distinct()) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    val work = if (transpose) PanelWork.SparseRightHandSideReduction else PanelWork.SparseRightHandSides
                    val group = engine.executionGroup(
                        work,
                        order,
                        1024,
                        contiguous = engine == BuiltinEngines.simd,
                    )
                    val sides = group + 1
                    for (unitDiag in booleanArrayOf(false, true)) {
                        // Small off-diagonal entries keep the implicit unit triangle well conditioned too.
                        val dense = wellConditioned(order, rng)
                        val sparse = SparseMatrix.ofColumns(
                            order,
                            order,
                            List(order) { j ->
                                val rows = if (lower) j until order else 0..j
                                rows.map { i ->
                                    if (i != j) dense[i, j] /= order
                                    i to if (unitDiag && i == j) Double.NaN else dense[i, j]
                                }
                            },
                        )
                        for (right in booleanArrayOf(false, true)) {
                            val input = if (right) randomMatrix(sides, order, rng) else randomMatrix(order, sides, rng)
                            val expected = input.copyOf()
                            val actual = input.copyOf()
                            if (solve) {
                                ReferenceBlas.trsm(dense, expected, lower, transpose, unitDiag, right, alpha = -0.75)
                                engine.trsm(sparse, actual, lower, transpose, unitDiag, right, -0.75, Workspace())
                            } else {
                                ReferenceBlas.trmm(dense, expected, lower, transpose, unitDiag, right, alpha = -0.75)
                                engine.trmm(sparse, actual, lower, transpose, unitDiag, right, -0.75, Workspace())
                            }
                            val context = "${engine.name} solve=$solve lower=$lower transpose=$transpose " +
                                "unit=$unitDiag right=$right"
                            assertClose(expected, actual, context)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `triangular multiplication treats a missing diagonal as zero without division`() {
        val triangle = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 3.0), listOf()))
        val vector = doubleArrayOf(2.0, 4.0)
        koblas.trmv(triangle, vector, lower = true)

        assertContentEquals(doubleArrayOf(4.0, 6.0), vector)
    }

    @Test
    fun `triangular multiplication snapshots an aliased triangular value buffer`() {
        val triangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 2.0, 1 to 3.0), listOf(0 to 0.0, 1 to 4.0)),
        )
        val b = DenseMatrix.wrap(2, 2, triangle.values)

        koblas.trmm(triangle, b, lower = true)

        assertContentEquals(doubleArrayOf(4.0, 18.0, 0.0, 16.0), b.values)
    }

    @Test
    fun `an alpha of zero empties a multiply destination without reading the triangle`() {
        val poisoned = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to Double.NaN), listOf(1 to Double.NaN)))
        val b = DenseMatrix.wrap(2, 1, doubleArrayOf(1.0, 1.0))

        koblas.trmm(poisoned, b, lower = true, alpha = 0.0)

        assertContentEquals(doubleArrayOf(0.0, 0.0), b.values)
    }

    @Test
    fun `alpha scales the right hand side before the solve`() {
        val rng = Random(20260823)
        val n = 5
        val (sparse, _) = triangle(n, lower = true, rng)
        val b = randomMatrix(n, 2, rng)

        val scaled = DenseMatrix.wrap(n, 2, DoubleArray(b.values.size) { 2.5 * b.values[it] })
        koblas.trsm(sparse, scaled, lower = true)
        val withAlpha = DenseMatrix.wrap(n, 2, b.values.copyOf())
        koblas.trsm(sparse, withAlpha, lower = true, alpha = 2.5)

        assertClose(scaled, withAlpha, "alpha must scale B", tolerance = 1e-9)
    }

    @Test
    fun `an alpha of zero empties the right hand side without solving`() {
        val n = 3
        // A NaN diagonal, which a solve would spread through the answer.
        val poisoned = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to Double.NaN) })
        val b = DenseMatrix.wrap(n, 2, DoubleArray(n * 2) { 1.0 })

        koblas.trsm(poisoned, b, lower = true, alpha = 0.0)

        assertTrue(b.values.all { it == 0.0 }, "alpha = 0 must empty B without a solve")
    }

    @Test
    fun `a right hand side that does not meet the triangle is rejected on both sides`() {
        val rng = Random(20260824)
        val n = 4
        val (sparse, _) = triangle(n, lower = true, rng)

        assertFailsWith<DimensionMismatch> { koblas.trsm(sparse, randomMatrix(n + 1, 2, rng), lower = true) }
        assertFailsWith<DimensionMismatch> {
            koblas.trsm(
                sparse,
                randomMatrix(2, n + 1, rng),
                lower = true,
                right = true,
            )
        }
    }

    @Test
    fun `the opposite triangle is never read`() {
        val rng = Random(20260813)
        val n = 7
        val (clean, dense) = triangle(n, lower = true, rng)
        // Same lower triangle, plus NaN through the strict upper one, which must not reach the answer.
        val poisoned = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                if (i < j) {
                    column.add(i to Double.NaN)
                } else if (dense[i, j] != 0.0) {
                    column.add(i to dense[i, j])
                }
            }
            poisoned.add(column)
        }
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val fromClean = b.copyOf().also { koblas.trsv(clean, it, lower = true) }
        val fromPoisoned = b.copyOf().also { koblas.trsv(SparseMatrix.ofColumns(n, n, poisoned), it, lower = true) }
        assertClose(fromClean, fromPoisoned, "a NaN in the unread triangle changed the answer", tolerance = 0.0)
    }

    @Test
    fun `unitDiag takes the diagonal as one without reading it`() {
        // A diagonal of 5.0 that must be ignored, and a NaN one that must not even be looked at.
        val lower = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 5.0, 1 to 3.0), listOf(1 to Double.NaN)))
        val x = doubleArrayOf(1.0, 2.0)
        koblas.trsv(lower, x, lower = true, unitDiag = true)
        assertEquals(1.0, x[0])
        assertEquals(-1.0, x[1], "row 1 is 3*x0 + 1*x1 = 2")

        val t = doubleArrayOf(1.0, 2.0)
        koblas.trsv(lower, t, lower = true, transpose = true, unitDiag = true)
        assertEquals(2.0, t[1])
        assertEquals(-5.0, t[0], "column 0 transposed is 1*x0 + 3*x1 = 1")
    }
}
