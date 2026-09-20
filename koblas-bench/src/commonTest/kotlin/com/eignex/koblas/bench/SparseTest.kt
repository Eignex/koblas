package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.SparseCall
import com.eignex.koblas.sparse.SparseMatrixOperation
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SparseTest {
    /**
     * The prepared modes measure four different amounts of work, and the row says which: an amortized row is
     * one preparation and all its uses as one batch, so its mode names the count rather than a call, where a
     * prepared row is the steady state with the setup already paid.
     */
    @Test
    fun `prepared modes name the work they time`() {
        val base = "spmm+9x4x7+sparse-uniform+density=0.25"
        val setup = assertNotNull(work("$base+mode=setup"))
        val firstUse = assertNotNull(work("$base+mode=firstuse"))
        val amortized = assertNotNull(work("$base+mode=amortized+reuse=8"))
        val prepared = assertNotNull(work("$base+mode=prepared"))

        assertEquals("prepare", setup.timingMode)
        assertEquals("prepare-and-first-use", firstUse.timingMode)
        assertEquals("prepare-and-8-uses", amortized.timingMode)
        assertEquals("prepared", prepared.timingMode)
        assertContains(assertNotNull(setup.kernel), "spprepare")
    }

    /**
     * A prepared transposed product against a dense block reports the traversal a one-shot call takes, since
     * the snapshot derives no orientation for it and only the structure copy is bought.
     */
    @Test
    fun `a prepared transposed row reports the one shot traversal`() {
        // Enough right-hand sides that the traversal cuts a panel.
        val base = "spmm+33x16x21+sparse-uniform+density=0.25+transA=T"
        val oneShot = assertNotNull(work("$base+mode=oneshot"))
        val prepared = assertNotNull(work("$base+mode=prepared"))
        val firstUse = assertNotNull(work("$base+mode=firstuse"))

        assertContains(assertNotNull(oneShot.kernel), "spmm@1")
        assertEquals(oneShot.kernel, prepared.kernel)
        assertContains(assertNotNull(firstUse.kernel), "spprepare")
    }

    /** A timed body that produces an object keeps it observable, which is what makes the row about it. */
    @Test
    fun `a setup row retains the snapshot it built`() {
        val setup = assertNotNull(work("spmm+9x4x7+sparse-uniform+density=0.25+mode=setup"))

        setup.run()

        assertEquals("PreparedSparseMatrix", Retained.describe())
    }

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
                val kernel = assertNotNull(work.kernel, "$id on ${engine.name}")
                assertTrue(kernel.startsWith("portable-csc"), "$id on ${engine.name} claimed $kernel")
                assertTrue(kernel.endsWith("/${case.operation}"), kernel)
            }
        }
    }

    // The scattered product is the interesting one: its leaf is the engine's indexed selection rather than
    // the scheduling's own arithmetic.
    @Test
    fun `a scattered sparse product names the level one leaf its columns reach`() {
        val case = Cases.parse("spgemv+64x32+sparse-uniform+density=0.5+mode=oneshot").single()
        val engine = BuiltinEngines.scalar

        val work = assertNotNull(sparseArm(case, engine)?.work)

        val expected = engine.routeOf(
            SparseMatrixOperation.Gemv,
            SparseCall(Fixtures.sparse(64, 32, 0.5, 1), alpha = 0.875, beta = -0.25, destinationElements = 64, depth = 32),
        )
        assertEquals("${expected.implementation}/spgemv", work.kernel)
        // A non-unit beta scales the destination through a dense kernel, and the row says so rather than
        // naming only the indexed leaf the columns reach.
        assertEquals("portable-csc+scalar/scale+scalar/axpy/spgemv", work.kernel)
    }

    @Test
    fun `the four prepared boundaries are separate timings of different work`() {
        val modes = listOf(
            "oneshot" to "oneshot",
            "prepared" to "prepared",
            "setup" to "prepare",
            "firstuse" to "prepare-and-first-use",
        )
        for ((mode, timing) in modes) {
            val case = Cases.parse("spgemv+64x32+sparse-uniform+density=0.25+mode=$mode").single()

            val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work, mode)

            assertEquals(timing, work.timingMode, mode)
            assertTrue(work.run().isFinite(), "$mode produced no usable result")
        }
    }

    // Preparing runs no arithmetic kernel, so a setup row naming one attaches its number to a call that
    // never happened.
    @Test
    fun `a setup row names snapshot preparation rather than an arithmetic kernel`() {
        val case = Cases.parse("spgemv+64x32+sparse-uniform+density=0.25+mode=setup").single()

        val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work)

        assertEquals("portable-csc/spprepare", work.kernel)
        assertEquals("prepare", work.timingMode)
    }

    // First use is preparation plus one call, and both ran inside the timed region.
    @Test
    fun `a first-use row names the preparation and the call it pays for`() {
        val case = Cases.parse("spmm+33x4x21+sparse-uniform+density=0.05+mode=firstuse+transA=T").single()

        val work = assertNotNull(sparseArm(case, BuiltinEngines.scalar)?.work)

        assertEquals("composed", work.comparisonKind)
        assertEquals("prepare-and-first-use", work.timingMode)
        val kernel = assertNotNull(work.kernel)
        assertTrue(kernel.startsWith("portable-csc/spprepare then "), kernel)
        // The entry point, and then the grouping the call resolved, which a dense row publishes the same way.
        assertTrue(kernel.substringAfterLast('/').startsWith("spmm@"), kernel)
    }

    // The generic entry point uses the selected engine, so an arm naming a different one would publish its
    // own label over another engine's work.
    @Test
    fun `the generic entry point is timed only on the arm whose engine it actually uses`() {
        for (id in listOf(
            "spmm-generic+33x4x21+sparse-uniform+density=0.05+mode=oneshot",
            "spmm-generic-right+33x4x21+sparse-uniform+density=0.05+mode=oneshot",
            "spgemm-generic+33x17x21+sparse-uniform+density=0.05+mode=oneshot",
        )) {
            val case = Cases.parse(id).single()
            for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
                val arm = assertNotNull(sparseArm(case, engine), id)
                if (engine === koblas) {
                    val work = assertNotNull(arm.work, "$id on the selected engine")
                    assertEquals("default-policy", work.comparisonKind, id)
                    assertEquals("oneshot-generic", work.timingMode, id)
                    assertTrue(assertNotNull(work.kernel).startsWith("portable-csc"), id)
                    assertTrue(work.run().isFinite(), id)
                } else {
                    assertNull(arm.work, "$id was timed under an engine it does not use")
                    assertContains(assertNotNull(arm.reason), "platform-selected engine")
                }
            }
        }
    }

    // The counterpart of the generic case: a snapshot is built by the requested engine rather than the
    // selected one, so its route is that engine's answer.
    @Test
    fun `a prepared row times a snapshot built by the engine whose route it reports`() {
        val case = Cases.parse("spgemv+64x32+sparse-uniform+density=0.25+mode=prepared").single()

        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            val work = assertNotNull(sparseArm(case, engine)?.work, engine.name)
            val expected = engine.routeOf(
                SparseMatrixOperation.Gemv,
                SparseCall(Fixtures.sparse(64, 32, 0.25, 1), alpha = 0.875, beta = -0.25, destinationElements = 64, depth = 32),
            )
            assertEquals("${expected.implementation}/spgemv", work.kernel, engine.name)
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
            "spgemv+9x7+sparse-uniform+density=0.3+mode=prepared",
            "spgemv+9x7+sparse-uniform+density=0.3+mode=firstuse",
            "spmm+9x3x7+sparse-uniform+density=0.3+mode=oneshot",
            "spmm+9x3x7+sparse-uniform+density=0.3+mode=prepared+transA=T",
            "spmm+9x3x7+sparse-uniform+density=0.3+mode=firstuse+transA=T",
            "spgemm+9x5x7+sparse-uniform+density=0.3+mode=oneshot",
            "spgemm+9x5x7+sparse-uniform+density=0.3+mode=prepared+transA=T",
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

    /**
     * The generic pairings on the engine that actually serves them, over a rectangular shape so a transposed
     * operand cannot pass by being square.
     */
    @Test
    fun `every generic pairing verifies its result on the selected engine`() {
        for (id in listOf(
            "spmm-generic+9x3x7+sparse-uniform+density=0.3+mode=oneshot",
            "spmm-generic+9x3x7+sparse-uniform+density=0.3+mode=oneshot+transA=T",
            "spmm-generic-right+9x3x7+sparse-uniform+density=0.3+mode=oneshot",
            "spmm-generic-right+9x3x7+sparse-uniform+density=0.3+mode=oneshot+transA=T",
            "spgemm-generic+9x5x7+sparse-uniform+density=0.3+mode=oneshot",
        )) {
            val case = Cases.parse(id).single()
            val work = assertNotNull(sparseArm(case, koblas)?.work, id)
            assertTrue(work.run().isFinite(), id)
        }
    }

    @Test
    fun `an option the generic allocating product cannot apply is rejected at parsing`() {
        assertFailsWith<IllegalArgumentException> {
            Cases.parse("spgemm-generic+9x5x7+sparse-uniform+density=0.3+mode=oneshot+transA=T")
        }
    }

    // A case stopping before the arithmetic says so rather than publishing the cost of scaling a destination
    // under the name of the product that did not happen.
    @Test
    fun `a case with no arithmetic to do is declined rather than timed`() {
        val empty = Fixtures.sparse(8, 8, 0.25, 1)
        val route = BuiltinEngines.scalar.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(empty, alpha = 0.0, beta = 0.5, destinationElements = 64, depth = 32),
        )

        assertEquals("nowork", route.kind.name.lowercase())
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
        val preparedRowMajor = assertNotNull(work("$base+mode=prepared+transA=T+transB=T"))

        assertContains(assertNotNull(columnMajor.kernel), "spmm")
        assertContains(assertNotNull(rowMajor.kernel), "spmm")
        assertContains(assertNotNull(preparedRowMajor.kernel), "spmm")
    }

    @Test
    fun `a transposed dense operand reports its actual traversal`() {
        val engine = BuiltinEngines.scalar
        val case = Cases.parse("spmm+33x16x21+sparse-uniform+density=0.25+mode=oneshot+transA=T+transB=T")
            .single()
        val a = Fixtures.sparse(21, 33, 0.25, 1)
        fun route(transposeDense: Boolean) = sparseMatrixKernel(
            engine.routeOf(
                SparseMatrixOperation.GemmDense,
                SparseCall(
                    a, alpha = 0.875, beta = -0.25, destinationElements = 33 * 16, depth = 21,
                    rightHandSides = 16, transposeSparse = true, transposeDense = transposeDense,
                ),
            ),
        )
        val expected = route(true)
        val wrongLayout = route(false)

        val actual = assertNotNull(sparseArm(case, engine)?.work).kernel

        assertNotEquals(wrongLayout, expected)
        assertEquals(expected, actual)
    }
}
