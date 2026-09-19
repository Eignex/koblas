package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SparseTest {
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
    fun `a sparse row names the implementation that ran rather than the engine`() {
        val case = Cases.parse("spdot+4096+sparse-uniform+density=0.25").single()

        val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work)

        assertEquals("direct", work.comparisonKind)
        assertEquals("scalar/dotDense", work.kernel)
    }

    @Test
    fun `an exact arm declines a case its own kernels do not run`() {
        val simd = BuiltinEngines.simd
        if (simd == null) {
            println("SKIPPED: no Vector API engine on this host; delegation enforcement was not verified")
            return
        }
        // Two stored entries sit far below the vector crossover, so the Vector API selection hands the whole
        // call to the scalar kernels. Timing it here would publish the scalar loop as a SIMD measurement.
        val case = Cases.parse("spdot+8+sparse-uniform+density=0.25").single()

        val arm = assertNotNull(sparseArm(case, simd))

        assertNull(arm.work, "a delegated call is not a measurement of the arm that was asked for")
        assertContains(assertNotNull(arm.reason), "simd has no dotDense kernel")
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
    fun `a whole vector reduction names the dense kernel rather than the selection`() {
        val case = Cases.parse("spasum+4096+sparse-uniform+density=0.25").single()

        val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work)

        assertEquals("scalar/asum", work.kernel)
    }

    @Test
    fun `a reduction whose kernel the values decide is not an exact comparison`() {
        val simd = BuiltinEngines.simd
        if (simd == null) {
            println("SKIPPED: no Vector API engine on this host; the norm route was not verified")
            return
        }
        // The vector path computes a square sum and abandons it for the rescaling loop outside the normal
        // range, so at a vectorising width the kernel that finishes the norm depends on the stored values.
        val case = Cases.parse("spnrm2+4096+sparse-uniform+density=0.25").single()

        val arm = assertNotNull(sparseArm(case, simd))

        assertNull(arm.work)
        assertContains(assertNotNull(arm.reason), "on the values")
    }

    @Test
    fun `an operation the vector kernels never implemented is declined at any width`() {
        val simd = BuiltinEngines.simd
        if (simd == null) {
            println("SKIPPED: no Vector API engine on this host; delegation enforcement was not verified")
            return
        }
        val case = Cases.parse("spdot-sparse+65536+sparse-uniform+density=0.25").single()

        val arm = assertNotNull(sparseArm(case, simd))

        assertNull(arm.work)
        assertContains(assertNotNull(arm.reason), "no dotSparse kernel")
    }

    @Test
    fun `a sparse matrix row names the portable scheduling on every engine`() {
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            for (id in listOf(
                "spsymv+65+sparse-triangular+density=0.05+mode=oneshot+uplo=L",
                "spgemm+33x17x21+sparse-uniform+density=0.05+mode=oneshot",
                "spsyrk-sparse+33x17+sparse-uniform+density=0.05+uplo=L",
                "spadd+33x17+sparse-uniform+density=0.05",
            )) {
                val case = Cases.parse(id).single()
                val work = assertNotNull(sparseArm(case, engine)?.work, "$id on ${engine.name}")
                assertEquals("portable-csc/${case.operation}", work.kernel, "$id on ${engine.name}")
            }
        }
    }

    @Test
    fun `a scattered sparse product names the level one leaf its columns reach`() {
        val case = Cases.parse("spgemv+64x32+sparse-uniform+density=0.5+mode=oneshot").single()

        val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work)

        assertEquals("portable-csc+scalar/axpy/spgemv", work.kernel)
    }

    @Test
    fun `the four prepared boundaries are separate timings of different work`() {
        val modes = listOf("oneshot" to "oneshot", "prepared" to "prepared", "setup" to "prepare", "firstuse" to "prepare-and-first-use")
        for ((mode, timing) in modes) {
            val case = Cases.parse("spgemv+64x32+sparse-uniform+density=0.25+mode=$mode").single()

            val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work, mode)

            assertEquals(timing, work.timingMode, mode)
            assertTrue(work.run().isFinite(), "$mode produced no usable result")
        }
    }

    @Test
    fun `the generic entry point is timed with its dispatch on both operand sides`() {
        for (id in listOf(
            "spmm-generic+33x4x21+sparse-uniform+density=0.05+mode=oneshot",
            "spmm-generic-right+33x4x21+sparse-uniform+density=0.05+mode=oneshot",
            "spgemm-generic+33x17x21+sparse-uniform+density=0.05+mode=oneshot",
        )) {
            val case = Cases.parse(id).single()

            val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work, id)

            assertEquals("oneshot-generic", work.timingMode, id)
            assertTrue(work.run().isFinite(), id)
        }
    }
}
