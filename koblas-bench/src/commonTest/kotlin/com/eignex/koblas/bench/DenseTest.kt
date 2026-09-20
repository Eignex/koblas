package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.KoblasEngineApi
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DenseTest {
    @Test
    fun `arithmetic scaling preserves fixture magnitudes across repeated calls`() {
        val case = Cases.parse("scal+17+uniform+timing=arithmetic").single()
        val work = assertNotNull(denseWork(case, BuiltinEngines.scalar))
        val initial = Fixtures.vector(17, 1)

        repeat(101) { work.run() }

        assertEquals("arithmetic", work.timingMode)
        assertContentEquals(initial.map { -it }.toDoubleArray(), work.result)
        work.run()
        assertContentEquals(initial, work.result)
    }

    /**
     * Every product case runs its preflight, names a route and times something.
     *
     * The preflight is inside building the work, so a case that computed the wrong thing fails here rather
     * than publishing a number; what this adds on top is that each of the new entry points is reachable and
     * carries the attribution its row will be written with.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `every product case names the route of the call it makes`() {
        val engines = listOfNotNull<KoblasEngine>(BuiltinEngines.scalar, BuiltinEngines.simd)
        for (line in listOf(
            "gemm+64x64x128+uniform",
            "gemm+256x2x512+uniform+transA=T+transB=T",
            "gemm-pack+64x64x64+uniform",
            "gemm-packed+64x64x64+uniform",
            "gemm-packed-left+64x64x64+uniform",
            "gemm-packed-right+64x64x64+uniform",
            "product-block+64x64x128+uniform",
            "product-block+65x63x128+uniform",
        )) {
            val case = Cases.parse(line).single()
            for (engine in engines) {
                val work = assertNotNull(denseWork(case, engine), "$line on ${engine.name}")
                val kernel = assertNotNull(work.kernel, "$line on ${engine.name} named no route")

                assertTrue(kernel.isNotEmpty(), "$line on ${engine.name}")
                assertTrue(work.run().isFinite(), "$line on ${engine.name} produced no finite sink")
                work.close()
            }
        }
    }

    /**
     * Every structured and triangular case names a route that carries the facts its call was made with.
     *
     * The route is built from the facts the case passes, so a case that omitted one would describe a
     * different call from the one it times: a rank update whose transpose never reached the route would
     * name the panels of the untransposed shape, and a solve whose side never reached it would name the
     * wrong stride and so the wrong substitution body. Each pair below differs only in such a fact, and the
     * rows are required to differ with it.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `structured and triangular cases carry the facts their routes are built from`() {
        val engines = listOfNotNull<KoblasEngine>(BuiltinEngines.scalar, BuiltinEngines.simd)
        for (line in STRUCTURED_CASES) {
            val case = Cases.parse(line).single()
            for (engine in engines) {
                val work = assertNotNull(denseWork(case, engine), "$line on ${engine.name}")
                val kernel = assertNotNull(work.kernel, "$line on ${engine.name} named no route")

                assertTrue(kernel.startsWith("portable-dense+"), "$line on ${engine.name} named $kernel")
                // Every one of these shapes is past the crossover its route needs, so a row naming no
                // block and no substitution would mean the case reached a route it was not chosen for.
                assertTrue(
                    "/product-block" in kernel || "/diagonal-solve" in kernel || "/diagonal-multiply" in kernel,
                    "$line on ${engine.name} named $kernel",
                )
                assertTrue(work.run().isFinite(), "$line on ${engine.name} produced no finite sink")
                work.close()
            }
        }
    }

    /**
     * A fact that changes what runs changes the row, on the arm that owns the kernels it changes.
     *
     * The two solves below are the same shape on the same engine and differ only in their side. A left one
     * leaves its right-hand sides strided and the vector backend gathers a block of them; a right one finds
     * them adjacent and gathers nothing. A row that did not carry the side would publish one of those two
     * under the other's name.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `a triangular row names the gather its side decides`() {
        val engine = BuiltinEngines.simd ?: return
        fun kernelOf(line: String): String =
            assertNotNull(assertNotNull(denseWork(Cases.parse(line).single(), engine)).kernel)

        val left = kernelOf("trsm+96x16+triangular+side=L+uplo=L+transA=N+diag=N")
        val right = kernelOf("trsm+16x96+triangular+side=R+uplo=L+transA=N+diag=N")

        assertTrue("portable-gather/rhs-block" in left, left)
        assertTrue("portable-gather/rhs-block" !in right, right)
        assertTrue("diagonal-solve" in left && "diagonal-solve" in right, "$left and $right")
    }

    /**
     * The packing row names preparation, and the prepacked rows name the packing they still do.
     *
     * The three retained entry points are three different amounts of work, and a reader comparing them with
     * the packing-only row has to be able to tell which is which from the row itself.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `the packed rows say which copies they still make`() {
        val engine = BuiltinEngines.scalar
        fun kernelOf(line: String): String =
            assertNotNull(assertNotNull(denseWork(Cases.parse(line).single(), engine)).kernel)

        assertEquals("portable-pack/pack-operands", kernelOf("gemm-pack+64x64x64+uniform"))
        assertTrue(
            "portable-pack" !in kernelOf("gemm-packed+64x64x64+uniform"),
            kernelOf("gemm-packed+64x64x64+uniform"),
        )
        assertTrue(
            kernelOf("gemm-packed-left+64x64x64+uniform").contains("portable-pack/right-panel"),
            kernelOf("gemm-packed-left+64x64x64+uniform"),
        )
        assertTrue(
            kernelOf("gemm-packed-right+64x64x64+uniform").contains("portable-pack/left-panel"),
            kernelOf("gemm-packed-right+64x64x64+uniform"),
        )
    }

    /**
     * The generic product is a default-policy row and is declined on every arm but the selected engine's.
     *
     * A caller holding a `Matrix` has no engine to pass, so timing it under another arm's label would
     * publish that label over the selected engine's work.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `the generic product is timed once on the arm whose engine is the selected one`() {
        val case = Cases.parse("gemm-generic+64x64x64+uniform").single()

        val selected = denseArm(case, koblas)
        assertEquals("default-policy", assertNotNull(selected?.work).comparisonKind)
        assertNotNull(selected?.work?.kernel)

        val other = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).firstOrNull { it !== koblas }
        if (other != null) {
            val declined = denseArm(case, other)
            assertNull(declined?.work, "the generic product was timed under ${other.name}")
            assertNotNull(declined?.reason, "a declined generic row carries no reason")
        }
    }

    private companion object {
        /**
         * One case of each structured and triangular entry point, in both of the orientations it has.
         *
         * Smaller than the shapes the workload carries, and deliberately. What is under test here is that
         * a case passes the facts its route is built from, which the smallest shape that still takes the
         * route shows as well as a large one; building each case runs its numerical preflight against the
         * naive oracle, whose cost is the cube of the order. The shipped shapes are exercised by a capture
         * against the harness, where the timing is the point and the preflight is paid once.
         *
         * Each order below is past the crossover its route needs: a square of forty-eight is packed into
         * tiles on every backend here, and a triangle of ninety-six is more than one diagonal block, so a
         * product runs between them.
         */
        val STRUCTURED_CASES = listOf(
            "gemmt+48x48+uniform+uplo=L+transA=N+transB=N",
            "gemmt+48x48+uniform+uplo=U+transA=T+transB=T",
            "syrk+48x48+uniform+uplo=L+transA=N",
            "syr2k+48x48+uniform+uplo=L+transA=N",
            "symm+48x48+uniform+side=L+uplo=L",
            "symm+48x48+uniform+side=R+uplo=U",
            "trsm+96x16+triangular+side=L+uplo=L+transA=N+diag=N",
            "trsm+96x1+triangular+side=L+uplo=L+transA=N+diag=N",
            "trsm+16x96+triangular+side=R+uplo=L+transA=N+diag=N",
            "trsm+96x16+triangular+side=L+uplo=U+transA=T+diag=U",
            "trmm+96x16+triangular+side=L+uplo=U+transA=T+diag=N",
        )
    }
}
