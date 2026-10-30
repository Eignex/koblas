package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class DenseTest {
    @Test
    fun `norm cases agree across engines and name the selected engine`() {
        val case = Cases.parse("nrm2+512+uniform").single()
        val expected = assertNotNull(denseWork(case, BuiltinEngines.scalar)).run()
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            val work = assertNotNull(denseWork(case, engine))

            assertEquals(ENGINE_COMPARISON, work.comparisonKind)
            assertEquals("${engine.name}/nrm2", work.kernel)
            assertEquals(expected, work.run(), 1e-12)
        }
    }

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

    // Preflight must reject incorrect work before publishing a timing.
    @Test
    fun `every product case names the engine that runs it`() {
        val engines = listOfNotNull<KoblasEngine>(BuiltinEngines.scalar, BuiltinEngines.simd)
        for (line in listOf(
            "gemm+64x64x128+uniform",
            "gemm+256x2x512+uniform+transA=T+transB=T",
        )) {
            val case = Cases.parse(line).single()
            for (engine in engines) {
                val work = assertNotNull(denseWork(case, engine), "$line on ${engine.name}")
                val kernel = assertNotNull(work.kernel, "$line on ${engine.name} named no engine")

                assertTrue(kernel.isNotEmpty(), "$line on ${engine.name}")
                assertTrue(work.run().isFinite(), "$line on ${engine.name} produced no finite sink")
                work.close()
            }
        }
    }

    @Test
    fun `structured and triangular cases run on every engine and name it`() {
        val engines = listOfNotNull<KoblasEngine>(BuiltinEngines.scalar, BuiltinEngines.simd)
        for (line in STRUCTURED_CASES) {
            val case = Cases.parse(line).single()
            for (engine in engines) {
                val work = assertNotNull(denseWork(case, engine), "$line on ${engine.name}")
                assertEquals("${engine.name}/${case.operation}", work.kernel)
                assertTrue(work.run().isFinite(), "$line on ${engine.name} produced no finite sink")
                work.close()
            }
        }
    }

    @Test
    fun `the generic product runs on every explicitly requested engine`() {
        val case = Cases.parse("gemm-generic+15x7x31+uniform").single()
        for (engine in listOfNotNull(koblas, BuiltinEngines.scalar, BuiltinEngines.simd).distinct()) {
            val work = assertNotNull(denseArm(case, engine)?.work)

            assertEquals(ENGINE_COMPARISON, work.comparisonKind)
            assertNotNull(work.kernel)
            assertTrue(work.run().isFinite())
        }
    }

    @Test
    fun `borrowed product cases verify padding on every requested engine`() {
        for (transpose in listOf("", "+transA=T", "+transB=T")) {
            val case = Cases.parse("gemm+17x9x31+uniform$transpose+workspace=retained+layout=borrowed").single()
            for (engine in listOfNotNull(koblas, BuiltinEngines.scalar, BuiltinEngines.simd).distinct()) {
                val work = assertNotNull(denseWork(case, engine))
                assertTrue(work.run().isFinite())
                assertNotNull(work.kernel)
            }
        }
    }

    private companion object {
        /**
         * One case of each structured and triangular entry point, in both orientations. Deliberately smaller
         * than the shapes the workload carries, since each case's preflight costs the cube of the order.
         *
         * A square of forty-eight is packed into tiles on every backend here, and a triangle of ninety-six is
         * more than one diagonal block, so the blocked paths are what preflight checks.
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
