package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseMatrix
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PortableSparseFactorsTest {
    @Test
    fun `the LU factors reproduce the permuted matrix`() {
        val rng = Random(20260827)
        for (n in intArrayOf(1, 5, 24, 60)) {
            val a = dominant(n, rng)

            assertLuFactorsReproduce(a, F64ReferenceSparseLinearAlgebra.factor(a), "n=$n")
        }
    }

    @Test
    fun `an equilibrating LU reports the scaling its factors are of`() {
        val rng = Random(20260901)
        val a = dominant(30, rng)

        val lu = F64ReferenceSparseDecompositions(equilibrate = true).factor(a)

        assertLuFactorsReproduce(a, lu, "equilibrated")
    }

    @Test
    fun `the Cholesky factor reproduces the matrix`() {
        val rng = Random(20260902)
        for (n in intArrayOf(1, 6, 30)) {
            val a = spdLowerTriangle(n, rng)

            assertCholeskyFactorReproduces(a, F64ReferenceSparseLinearAlgebra.cholesky(a), "n=$n")
        }
    }

    @Test
    fun `the LDL factors reproduce the matrix`() {
        val rng = Random(20260903)
        for (n in intArrayOf(1, 6, 30)) {
            val a = spdLowerTriangle(n, rng)

            assertLdlFactorsReproduce(a, F64ReferenceSparseLinearAlgebra.quasiDefiniteLdl(a), "n=$n")
        }
    }

    @Test
    fun `a singular factorization has no factors to give`() {
        // Column 1 repeats column 0, so no acceptable pivot remains at the second step.
        val a = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 1.0, 1 to 2.0)))

        val lu = F64ReferenceSparseLinearAlgebra.factor(a)

        assertEquals(true, lu.singular)
        assertFailsWith<com.eignex.koblas.SingularMatrix> { lu.l }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { lu.u }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { lu.offDiagonal }
    }

    @Test
    fun `a singular LDL factorization has no factors to give`() {
        val a = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), emptyList()))

        val ldl = F64ReferenceSparseLinearAlgebra.quasiDefiniteLdl(a)

        assertEquals(true, ldl.singular)
        assertFailsWith<com.eignex.koblas.SingularMatrix> { ldl.l }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { ldl.d }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { ldl.order }
    }
}

private fun dominant(n: Int, rng: Random): F64SparseMatrix {
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        for (i in 0 until n) {
            when {
                i == j -> entries.add(i to (n + rng.nextDouble()))
                rng.nextDouble() < 0.15 -> entries.add(i to rng.nextDouble(-1.0, 1.0))
            }
        }
        entries
    }
    return F64SparseMatrix.ofColumns(n, n, columns)
}

private fun spdLowerTriangle(n: Int, rng: Random): F64SparseMatrix {
    val below = List(n) { HashMap<Int, Double>() }
    val weight = DoubleArray(n)
    for (j in 0 until n) {
        for (i in j + 1 until n) {
            if (rng.nextDouble() >= 0.25) continue
            val v = rng.nextDouble(-1.0, 1.0)
            below[j][i] = v
            weight[i] += abs(v)
            weight[j] += abs(v)
        }
    }
    return F64SparseMatrix.ofColumns(
        n,
        n,
        List(n) { j ->
            val column = ArrayList<Pair<Int, Double>>()
            column.add(j to weight[j] + 1.0)
            for (i in j + 1 until n) below[j][i]?.let { column.add(i to it) }
            column
        },
    )
}
