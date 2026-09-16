package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.openBlas
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The vendor arm has to run the same workload as the arms it will be compared against, and has to say what it
 * ran. Both are checked here: the sentinel a case returns is a function of its output, so agreement with the
 * scalar arm on the same case means the two arms computed the same thing from the same fixtures.
 */
class VendorArmTest {
    private fun vendor(): Blas? = openBlas() ?: openBlas(Vendor.OpenBlas)

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

    @Test
    fun `an operation with no vendor entry point is declined with a reason rather than timed`() {
        val blas = vendor()
        if (blas == null) {
            println("SKIPPED: no CBLAS library installed; vendor declines were not verified on this host")
            return
        }
        // The Level 1 extensions outside standard CBLAS and everything sparse are Kotlin-only.
        for (line in listOf("sum+512+uniform", "compensated-sum+512+uniform", "spdot+512+sparse-uniform+density=0.01")) {
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
