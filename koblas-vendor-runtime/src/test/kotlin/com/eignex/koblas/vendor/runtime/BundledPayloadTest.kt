package com.eignex.koblas.vendor.runtime

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.ThreadEvidence
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.openBlas
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Opens each vendor out of the payload artifacts this host builds, and computes with it.
 *
 * This runs against the jars this module produces, on the test classpath, with `user.home` pointing at an empty
 * directory so that no installed library can answer instead. Nothing here reads the packaging build's own
 * account of what it did: the evidence is a library that opened from an extracted path and returned the right
 * numbers on one thread.
 */
class BundledPayloadTest {
    private val vendors: List<Vendor> = System.getProperty("koblas.payload.vendors", "")
        .split(",")
        .filter { it.isNotEmpty() }
        .map { name -> Vendor.entries.first { it.name.lowercase() == name } }

    @TestFactory
    fun `each packaged runtime opens and multiplies`(): List<DynamicTest> = vendors.map { vendor ->
        DynamicTest.dynamicTest(vendor.vendorName) {
            val blas = opened(vendor)

            // Small integers, so every product and sum is exact and the check is equality, not a tolerance.
            val a = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 4.0)))
            val b = DenseMatrix.ofRows(arrayOf(doubleArrayOf(5.0, 7.0), doubleArrayOf(6.0, 8.0)))
            val c = DenseMatrix.zero(2)
            blas.gemm(1.0, a, false, b, false, 0.0, c)

            assertEquals(listOf(23.0, 31.0, 34.0, 46.0), listOf(c[0, 0], c[0, 1], c[1, 0], c[1, 1]))
        }
    }

    @TestFactory
    fun `each library that answered came out of the payload`(): List<DynamicTest> = vendors.map { vendor ->
        DynamicTest.dynamicTest(vendor.vendorName) {
            val blas = opened(vendor)

            // The directory the JVM loader extracts into. An installed library would report a path under a
            // vendor prefix instead, which is what this run's empty home directory rules out.
            assertTrue(
                "koblas-vendor-${vendor.name.lowercase()}" in blas.libraryPath,
                "resolved ${blas.libraryPath}, which is not an extracted payload",
            )
        }
    }

    @TestFactory
    fun `each packaged runtime holds to one compute thread`(): List<DynamicTest> = vendors.map { vendor ->
        DynamicTest.dynamicTest(vendor.vendorName) {
            // Every bundled vendor exports a thread count, so unconfirmed here would mean the payload is not
            // the sequential build it is supposed to be rather than a vendor that cannot be asked.
            assertEquals(ThreadEvidence.Confirmed, opened(vendor).threadEvidence)
        }
    }

    private fun opened(vendor: Vendor): Blas =
        openBlas(vendor) ?: error("no $vendor resolved from its payload")
}
