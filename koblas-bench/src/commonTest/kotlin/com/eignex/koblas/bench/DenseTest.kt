package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class DenseTest {
    @Test
    fun `norm cases agree across engines and name the selected engine`() {
        val case = Cases.parse("nrm2+512+uniform").single()
        val expected = assertNotNull(denseWork(case, KoblasEngine.scalar)).run()
        for (engine in listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd)) {
            val work = assertNotNull(denseWork(case, engine))

            assertEquals(ENGINE_COMPARISON, work.comparisonKind)
            assertEquals("${engine.name}/nrm2", work.kernel)
            assertEquals(expected, work.run(), 1e-12)
        }
    }

    @Test
    fun `arithmetic scaling preserves fixture magnitudes across repeated calls`() {
        val case = Cases.parse("scal+17+uniform+timing=arithmetic").single()
        val work = assertNotNull(denseWork(case, KoblasEngine.scalar))
        val initial = assertNotNull(work.result).copyOf()
        val expected = initial.copyOf()
        KoblasEngine.scalar.scale(expected, 0, -1.0, 17)

        repeat(101) { work.run() }

        assertEquals("arithmetic", work.timingMode)
        assertContentEquals(expected, work.result)
        work.run()
        assertContentEquals(initial, work.result)
    }

    @Test
    fun `arithmetic rank updates return the matrix to its fixture every second call`() {
        for (line in listOf("ger+9x5+uniform", "syr+9+uniform+uplo=L", "syr2+9+uniform+uplo=U")) {
            val work = assertNotNull(denseWork(Cases.parse("$line+timing=arithmetic").single(), KoblasEngine.scalar))
            val initial = Fixtures.matrix(9, if (line.startsWith("ger")) 5 else 9, 1).values[0]

            val first = work.run()
            val second = work.run()

            assertEquals("arithmetic", work.timingMode, line)
            assertTrue(first != initial, line)
            assertEquals(initial, second, 1e-12, line)
        }
    }

    @Test
    fun `every product case runs on every engine and names it`() {
        val engines = listOfNotNull<KoblasEngine>(KoblasEngine.scalar, KoblasEngine.simd)
        for (line in PRODUCT_CASES) {
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
    fun `borrowed product cases verify padding on every requested engine`() {
        for (transpose in listOf("", "+transA=T", "+transB=T")) {
            val case = Cases.parse("gemm+17x9x31+uniform$transpose+workspace=retained+layout=borrowed").single()
            for (engine in listOfNotNull(koblas, KoblasEngine.scalar, KoblasEngine.simd).distinct()) {
                val work = assertNotNull(denseWork(case, engine))
                assertTrue(work.run().isFinite())
                assertNotNull(work.kernel)
            }
        }
    }

    @Test
    fun `strided vector workloads preserve padding and agree with the scalar reference`() {
        for (operation in listOf("dot", "copy", "axpy", "scal", "nrm2", "asum", "sum", "iamax", "swap", "rot")) {
            for (stride in listOf(2, -1, -2, 256, -257)) {
                val case = Cases.parse("$operation+17+uniform+stride=$stride").single()
                for (engine in listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd, koblas)) {
                    val work = assertNotNull(engineArm(case, engine)?.work)
                    assertTrue(work.run().isFinite())
                    assertNotNull(work.provider)
                }
            }
        }
    }

    private companion object {
        /**
         * One case of each general, structured and triangular product, in both orientations. Deliberately
         * smaller than the shapes the workload carries, since each case's preflight costs the cube of the order.
         *
         * A square of forty-eight is packed into tiles on every backend here, and a triangle of ninety-six is
         * more than one diagonal block, so the blocked paths are what preflight checks.
         */
        val PRODUCT_CASES = listOf(
            "gemm+64x64x128+uniform",
            "gemm+256x2x512+uniform+transA=T+transB=T",
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
