package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SparseTest {
    private fun work(id: String): CaseWork? = sparseArm(Cases.parse(id).single(), koblas)?.work

    @Test
    fun `gather timing variants return the same values across repeated calls`() {
        val resetCase = Cases.parse("spgather+64+sparse-uniform+density=0.25").single()
        val arithmeticCase = Cases.parse("${resetCase.id}+timing=arithmetic").single()
        val reset = assertNotNull(sparseArm(resetCase, BuiltinEngines.scalar)?.work)
        val arithmetic = assertNotNull(sparseArm(arithmeticCase, BuiltinEngines.scalar)?.work)

        repeat(3) {
            reset.run()
            arithmetic.run()
            assertContentEquals(reset.result, arithmetic.result)
        }
    }

    @Test
    fun `a sparse row names the selected engine`() {
        val case = Cases.parse("spdot+4096+sparse-uniform+density=0.25").single()

        val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work)

        assertEquals(ENGINE_COMPARISON, work.comparisonKind)
        assertEquals("${BuiltinEngines.scalar.name}/spdot", work.kernel)
    }

    @Test
    fun `the generic primitives are timed once rather than under each engine label`() {
        val case = Cases.parse("spaccumulate+4096+sparse-uniform+density=0.01").single()

        val scalar = assertNotNull(sparseArm(case, BuiltinEngines.scalar))
        val work = assertNotNull(scalar.work)

        assertEquals("composed", work.comparisonKind)
        assertEquals("primitives/scatterWorkspace plus gatherWorkspace", work.kernel)
        assertTrue(work.run() > 0.0, "the primitive case must gather the entries it scattered")

        val simd = BuiltinEngines.simd ?: return
        val declined = assertNotNull(sparseArm(case, simd))
        assertNull(declined.work)
        assertContains(assertNotNull(declined.reason), "one implementation")
    }

    @Test
    fun `a sparse matrix row names the engine it ran on`() {
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            for (id in listOf(
                "spsymv+65+sparse-triangular+density=0.05+mode=oneshot+uplo=L",
                "spgemm+33x17x21+sparse-uniform+density=0.05+mode=oneshot",
                "spsyrk-sparse+33x17+sparse-uniform+density=0.05+uplo=L",
                "spadd+33x17+sparse-uniform+density=0.05",
            )) {
                val case = Cases.parse(id).single()
                val work = assertNotNull(sparseArm(case, engine)?.work, "$id on ${engine.name}")
                assertEquals("${engine.name}/${case.operation}", work.kernel, id)
            }
        }
    }

    // The generic entry point uses the selected engine, so an arm naming a different one would publish its
    // own label over another engine's work.
    @Test
    fun `the generic entry point runs on every explicitly requested engine`() {
        for (id in listOf(
            "spmm-generic+9x3x7+sparse-uniform+density=0.3+mode=oneshot",
            "spmm-generic+9x3x7+sparse-uniform+density=0.3+mode=oneshot+transA=T",
            "spmm-generic-right+9x3x7+sparse-uniform+density=0.3+mode=oneshot",
            "spmm-generic-right+9x3x7+sparse-uniform+density=0.3+mode=oneshot+transA=T",
            "spgemm-generic+9x5x7+sparse-uniform+density=0.3+mode=oneshot",
        )) {
            val case = Cases.parse(id).single()
            for (engine in listOfNotNull(koblas, BuiltinEngines.scalar, BuiltinEngines.simd).distinct()) {
                val arm = assertNotNull(sparseArm(case, engine), id)
                val work = assertNotNull(arm.work, "$id on ${engine.name}")
                assertEquals(ENGINE_COMPARISON, work.comparisonKind, id)
                assertEquals("oneshot-generic", work.timingMode, id)
                assertEquals("${engine.name}/${case.operation}", work.kernel, id)
                assertTrue(work.run().isFinite(), id)
            }
        }
    }

    /**
     * Every sparse matrix operation, verified against the reference before it would be timed. Building an
     * arm runs its check, so this proves the checks execute for every case shape the file can hold. The
     * fixtures are small on purpose; the case file's own sizes are for measuring.
     */
    @Test
    fun `every sparse matrix operation verifies its result before it is timed`() {
        for (id in listOf(
            "spgemv+9x7+sparse-uniform+density=0.3+mode=oneshot",
            "spgemv+9x7+sparse-uniform+density=0.3+mode=oneshot+transA=T",
            "spmm+9x3x7+sparse-uniform+density=0.3+mode=oneshot",
            "spgemm+9x5x7+sparse-uniform+density=0.3+mode=oneshot",
            "spsymv+9+sparse-triangular+density=0.3+mode=oneshot+uplo=L",
            "spsymv+9+sparse-triangular+density=0.3+mode=oneshot+uplo=U",
            "spsymm+9x3+sparse-triangular+density=0.3+mode=oneshot+side=L+uplo=L",
            "spsymm+3x9+sparse-triangular+density=0.3+mode=oneshot+side=R+uplo=U",
            "sptrsv+9+sparse-triangular+density=0.3+mode=oneshot+uplo=L+transA=N+diag=N",
            "sptrsv+9+sparse-triangular+density=0.3+mode=oneshot+uplo=U+transA=T+diag=U",
            "sptrmv+9+sparse-triangular+density=0.3+mode=oneshot+uplo=U+transA=T+diag=U",
            "sptrsm+9x3+sparse-triangular+density=0.3+mode=oneshot+side=L+uplo=L+transA=N+diag=N",
            "sptrsm+3x9+sparse-triangular+density=0.3+mode=oneshot+side=R+uplo=U+transA=T+diag=U",
            "sptrmm+9x3+sparse-triangular+density=0.3+mode=oneshot+side=L+uplo=U+transA=T+diag=U",
            "sptrmm+3x9+sparse-triangular+density=0.3+mode=oneshot+side=R+uplo=L+transA=N+diag=N",
            "spsyrk-dense+9x5+sparse-uniform+density=0.3+uplo=U",
            "spsyrk-dense+9x5+sparse-uniform+density=0.3+uplo=L",
            "spsyrk-sparse+9x5+sparse-uniform+density=0.3+uplo=U",
            "spsyrk-sparse+9x5+sparse-uniform+density=0.3+uplo=L",
            "spadd+9x5+sparse-uniform+density=0.3",
        )) {
            val case = Cases.parse(id).single()
            val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work, id)
            assertTrue(work.run().isFinite(), id)
        }
    }

    @Test
    fun `an option the generic allocating product cannot apply is rejected at parsing`() {
        assertFailsWith<IllegalArgumentException> {
            Cases.parse("spgemm-generic+9x5x7+sparse-uniform+density=0.3+mode=oneshot+transA=T")
        }
    }

    /**
     * Each dense storage layout is checked against the reference computed from the densified operands as the
     * arm is built. What a transposed row adds is which axis the right-hand sides lie along, which is the
     * axis a panel is cut from.
     */
    @Test
    fun `a transposed dense operand passes numerical preflight`() {
        val base = "spmm+33x16x21+sparse-uniform+density=0.25"

        val columnMajor = assertNotNull(work("$base+mode=oneshot"))
        val rowMajor = assertNotNull(work("$base+mode=oneshot+transB=T"))

        assertContains(assertNotNull(columnMajor.kernel), "spmm")
        assertContains(assertNotNull(rowMajor.kernel), "spmm")
    }
}
