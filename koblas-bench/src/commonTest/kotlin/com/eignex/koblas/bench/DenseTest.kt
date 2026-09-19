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
}
