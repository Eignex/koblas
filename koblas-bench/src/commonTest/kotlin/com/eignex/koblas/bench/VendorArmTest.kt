package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.openBlas
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The vendor arm has to run the same workload as the arms it will be compared against, and has to say what it
 * ran. Both are checked here: the sentinel a case returns is a function of its output, so agreement with the
 * scalar arm on the same case means the two arms computed the same thing from the same fixtures.
 */
class VendorArmTest {
    private fun vendor(): Blas? = openBlas()

    private val dense = listOf(
        "dot+512+uniform",
        "axpy+512+uniform",
        "scal+512+uniform",
        "nrm2+512+uniform",
        "asum+512+uniform",
        "iamax+512+uniform",
        "swap+512+uniform",
        "rot+512+uniform",
        "gemv+64x48+uniform",
        "gemv+64x48+uniform+transA=T",
        "symv+48+uniform+uplo=L",
        "symv+48+uniform+uplo=U",
        "ger+64x48+uniform",
        "syr+48+uniform+uplo=L",
        "syr2+48+uniform+uplo=U",
        "trsv+48+triangular+uplo=L+transA=N+diag=N",
        "trsv+48+triangular+uplo=U+transA=T+diag=N",
        "trsv+48+triangular+uplo=L+transA=N+diag=U",
        "trmv+48+triangular+uplo=U+transA=T+diag=U",
        "gemm+32x21x48+uniform",
        "gemm+32x21x48+uniform+transA=T",
        "gemm+32x21x48+uniform+transB=T",
        "symm+32x21+uniform+side=L+uplo=L",
        "symm+32x21+uniform+side=R+uplo=U",
        "gemmt+32x21+uniform+uplo=L+transA=N+transB=N",
        "gemmt+32x21+uniform+uplo=U+transA=T+transB=T",
        "syrk+32x21+uniform+uplo=L+transA=N",
        "syrk+32x21+uniform+uplo=U+transA=T",
        "syr2k+32x21+uniform+uplo=U+transA=T",
        "trsm+32x16+triangular+side=L+uplo=L+transA=N+diag=N",
        "trmm+32x16+triangular+side=R+uplo=U+transA=T+diag=N",
    )

    @Test
    fun `every dense vendor case agrees with the scalar arm on the same fixtures`() {
        val blas = vendor()
        if (blas == null) {
            println("SKIPPED: no CBLAS library installed; the vendor arm was not verified on this host")
            return
        }
        for (line in dense) {
            val case = Cases.parse(line).single()
            val arm = vendorArm(case, blas)
            val work = assertNotNull(arm.work, "${case.id} was declined: ${arm.reason}")
            val reference = assertNotNull(denseWork(case, BuiltinEngines.scalar), "no scalar arm for ${case.id}")

            val actual = work.run()
            val expected = reference.run()

            assertEquals(reference.timingMode, work.timingMode, "${case.id} changed its timing boundary")
            assertTrue(
                abs(actual - expected) <= 1e-9 * maxOf(1.0, abs(expected)),
                "${case.id} produced $actual against the scalar arm's $expected",
            )
            work.close()
            reference.close()
        }
    }

    @Test
    fun `a timed vendor case reports the entry point it called`() {
        val blas = vendor()
        if (blas == null) {
            println("SKIPPED: no CBLAS library installed; vendor attribution was not verified on this host")
            return
        }
        val case = Cases.parse("gemm+32x21x48+uniform").single()

        val work = assertNotNull(vendorArm(case, blas).work)

        assertEquals("direct", work.comparisonKind)
        assertEquals(
            "${blas.vendor.vendorName}/cblas_dgemm",
            work.kernel,
            "a vendor arm must carry the entry point of the call it makes",
        )
        work.close()
    }

    /**
     * A built-in product names this library's own traversal, packing and tile, and not an installed library.
     *
     * The shape is whole tiles on both axes for every geometry this machine can resolve, so the row has to
     * name the arithmetic body a full tile reaches and must not name the scalar edge; a SIMD arm that
     * quietly fell back to the portable tile would differ from the scalar arm's row, and both are checked.
     */
    @Test
    fun `built in level three arms name their own packing and tile rather than an installed vendor`() {
        val case = Cases.parse("gemm+64x64x64+uniform").single()
        val scalar = assertNotNull(denseWork(case, BuiltinEngines.scalar))
        val simdEngine = BuiltinEngines.simd

        val scalarKernel = assertNotNull(scalar.kernel)
        assertEquals(expectedProductKernel(BuiltinEngines.scalar), scalarKernel)
        assertTrue("scalar-tile" in scalarKernel, scalarKernel)
        if (simdEngine != null) {
            val simd = assertNotNull(denseWork(case, simdEngine))

            val simdKernel = assertNotNull(simd.kernel)
            assertEquals(expectedProductKernel(simdEngine), simdKernel)
            assertTrue("simd-tile" in simdKernel, "the vector arm fell back to $simdKernel")
            assertTrue("scalar-tile-edge" !in simdKernel, "a whole-tile shape named an edge: $simdKernel")
            assertNotEquals(scalarKernel, simdKernel, "both arms named the same product arithmetic")
            simd.close()
        }
        scalar.close()
    }

    /** What a 64 by 64 by 64 product row must carry on [engine]: shared traversal, both packers, its tile. */
    private fun expectedProductKernel(engine: KoblasEngine): String {
        val tile = engine.productKernels
        val bodies = tile.implementationsFor(tile.tileRows, tile.tileColumns, 64)
        assertEquals(1, bodies.size, "a whole tile reached more than one body on ${engine.name}")
        return "portable-dense+portable-pack/right-panel+portable-pack/left-panel+" +
            "${bodies.single()}/product-block/gemm"
    }

    @Test
    fun `an operation with no vendor entry point is declined with a reason rather than timed`() {
        val blas = vendor()
        if (blas == null) {
            println("SKIPPED: no CBLAS library installed; vendor declines were not verified on this host")
            return
        }
        // The Level 1 extensions outside standard CBLAS and everything sparse are Kotlin-only.
        for (line in listOf("sum+512+uniform", "spdot+512+sparse-uniform+density=0.01")) {
            val case = Cases.parse(line).single()

            val arm = vendorArm(case, blas)

            assertNull(arm.work, "${case.id} was timed through a vendor arm that cannot run it")
            assertTrue("no vendor entry point" in assertNotNull(arm.reason), "unexpected reason: ${arm.reason}")
        }
    }

    @Test
    fun `vendor modes name a vendor and other modes do not`() {
        assertEquals(Vendor.OneMkl, vendorFromMode("jvm-vendor-onemkl"))
        assertEquals(Vendor.OneMkl, vendorFromMode("native-vendor-onemkl"))
        assertEquals(Vendor.OpenBlas, vendorFromMode("jvm-vendor-openblas"))
        assertNull(vendorFromMode("jvm-simd"))
        assertNull(vendorFromMode("native"))
        assertNull(vendorFromMode("jvm-c-raw-avx2"))
    }
}
