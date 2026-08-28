package com.eignex.koblas.basiclu

import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.sparse.F64ReferenceSparseDecompositions
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.host.basiclu.BasicluFactorization
import com.eignex.koblas.sparse.host.basiclu.BasicluOptions
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu
import com.eignex.koblas.withColumn
import kotlin.test.*

class BundledBasicluTest {
    @Test
    fun `the bundled BASICLU is available`() {
        assertTrue(BundledBasiclu().isAvailable)
    }

    @Test
    fun `the bundled BASICLU solves sparse systems in both directions`() {
        val matrix = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 2.0), listOf(0 to 3.0)))
        val factorization = BundledBasiclu().factor(matrix)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(9.0, 4.0)))
        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(6.0, 6.0), transpose = true))
    }

    @Test
    fun `the bundled BASICLU solves in place with a workspace`() {
        val matrix = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 2.0), listOf(0 to 3.0)))
        val values = doubleArrayOf(9.0, 4.0)

        BundledBasiclu().factor(matrix).solveInto(values, values, workspace = Workspace())

        assertContentEquals(doubleArrayOf(2.0, 3.0), values)
    }

    @Test
    fun `the bundled BASICLU replaces a basis column`() {
        val basis = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
        val entering = F64SparseVector.of(2, intArrayOf(0, 1), doubleArrayOf(1.0, 2.0))
        val expected = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(0 to 1.0, 1 to 2.0)))
        val factorization = BundledBasiclu().factorBasis(basis).replaceColumn(1, entering)

        assertContentEquals(
            F64ReferenceSparseLinearAlgebra.factor(expected).solve(doubleArrayOf(3.0, 4.0)),
            factorization.solve(doubleArrayOf(3.0, 4.0)),
        )
    }

    /**
     * The basis a run of replacements left behind, which the factorization builds when it is asked for
     * rather than at each replacement. A slot replaced twice keeps only what went into it last.
     */
    @Test
    fun `the bundled BASICLU follows a run of replacements to the basis they leave`() {
        val basis = F64SparseMatrix.ofColumns(3, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(2 to 1.0)))
        val first = F64SparseVector.of(3, intArrayOf(0, 1), doubleArrayOf(1.0, 2.0))
        val second = F64SparseVector.of(3, intArrayOf(1, 2), doubleArrayOf(3.0, 1.0))
        val third = F64SparseVector.of(3, intArrayOf(0, 2), doubleArrayOf(4.0, 5.0))

        val factorization = BundledBasiclu()
            .factorBasis(basis)
            .replaceColumn(1, first)
            .replaceColumn(2, second)
            .replaceColumn(1, third)

        assertEquals(basis.withColumn(1, first).withColumn(2, second).withColumn(1, third), factorization.basis)
    }

    /**
     * `nnz` and `rcond` are the only readers of `koblas_basiclu_info`, so without them that downcall is
     * never made and a wrong descriptor for it would ship unnoticed. The operand carries off-diagonal fill,
     * since a permutation factors to no off-diagonal entries at all and would report nothing.
     */
    @Test
    fun `the bundled BASICLU reports its fill and pivot ratio`() {
        val matrix = F64SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)),
        )
        val factorization = BundledBasiclu().factor(matrix)
        assertTrue(factorization.nnz > 0, "expected off-diagonal fill, got ${factorization.nnz}")
        assertTrue(
            factorization.rcond > 0.0 && factorization.rcond <= 1.0,
            "pivot ratio ${factorization.rcond} outside (0, 1]",
        )
    }

    /**
     * The suffix is what discovery reads to leave this provider out when a deployment configured its own
     * library, so it holds whatever the configuration says.
     */
    @Test
    fun `it reports the bundled name whatever is configured`() {
        val previous = System.getProperty("koblas.basiclu.path")
        System.setProperty("koblas.basiclu.path", "/nonexistent/libbasiclu.so.2")
        try {
            val backend = BundledBasiclu()
            assertEquals("basiclu-bundled", backend.name)
            assertTrue(backend.isAvailable, "the bundled library still answers")
        } finally {
            if (previous == null) {
                System.clearProperty("koblas.basiclu.path")
            } else {
                System.setProperty("koblas.basiclu.path", previous)
            }
        }
    }

    @Test
    fun `it offers basis updates through the shared binding`() {
        assertTrue(BundledBasiclu().supportsBasisUpdates)
    }

    /**
     * Rows of 4 and 8 equilibrate by 2^-2 and 2^-3, so the scaling is active and every value stays a power
     * of two, which keeps the solve exact and the comparison with the portable one bit for bit.
     */
    @Test
    fun `an equilibrated factorization solves as the portable one does`() {
        val matrix = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(0 to 2.0, 1 to 8.0)))
        val rhs = doubleArrayOf(8.0, 8.0)
        val factorization = BundledBasiclu(BasicluOptions(equilibrate = true)).factor(matrix)

        assertContentEquals(
            F64ReferenceSparseDecompositions(equilibrate = true).factor(matrix).solve(rhs),
            factorization.solve(rhs),
        )
    }

    @Test
    fun `shared options reach bundled BASICLU routing and equilibration`() {
        val options = BasicluOptions(equilibrate = true)
        val backend = BundledBasiclu(options)
        val matrix = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 8.0)))

        val factorization = backend.factor(matrix)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(8.0, 24.0)))
        assertEquals("true", backend.backendMetadata.options["equilibrate"])
    }

    @Test
    fun `a bundled native factor closes deterministically`() {
        val backend = BundledBasiclu(BasicluOptions())
        val matrix = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 8.0)))
        val factorization = backend.factor(matrix)
        assertIs<BasicluFactorization>(factorization)
        val n = factorization.n

        factorization.close()
        factorization.close()

        assertEquals(n, factorization.n)
        assertFailsWith<IllegalStateException> { factorization.nnz }
        assertFailsWith<IllegalStateException> { factorization.rcond }
        assertFailsWith<IllegalStateException> { factorization.solve(DoubleArray(n)) }
    }

    @Test
    fun `an equilibrated transposed solve agrees with the portable one`() {
        val matrix = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(0 to 2.0, 1 to 8.0)))
        val rhs = doubleArrayOf(8.0, 8.0)
        val factorization = BundledBasiclu(BasicluOptions(equilibrate = true)).factor(matrix)

        assertContentEquals(
            F64ReferenceSparseDecompositions(equilibrate = true).factor(matrix).solve(rhs, transpose = true),
            factorization.solve(rhs, transpose = true),
        )
    }

    @Test
    fun `an equilibrated solve leaves its right-hand side alone`() {
        val matrix = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4e8), listOf(1 to 3e-7)))
        val rhs = doubleArrayOf(1.0, 2.0)

        BundledBasiclu(BasicluOptions(equilibrate = true)).factor(matrix).solve(rhs)

        assertContentEquals(doubleArrayOf(1.0, 2.0), rhs)
    }

    /** A caller reaching this library by name gets its own routines only if the two share a type. */
    @Test
    fun `the bundled BASICLU is the binding rather than a wrapper around it`() {
        assertIs<BasicluSparseLu>(BundledBasiclu(), "factorBasis is on the binding rather than on a seam")
    }
}
