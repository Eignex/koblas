package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * What a library has to satisfy before a caller reaches it, and what survives a caller's mistake.
 *
 * These run on every target, so the Native binding's pinning and cleanup are covered as well as the JVM's
 * transfers. The rules that do not need a library to exercise are tested against the rule itself, because a
 * host that happens to have a library missing exactly the right symbol is not something a test can rely on.
 */
class VendorLoadingTest {
    private fun installed(): VendorBlas? = openVendorBlas() ?: openVendorBlas(Vendor.OpenBlas)

    @Test
    fun `a library missing any required entry point is rejected rather than half bound`() {
        val complete = missingRequiredSymbols { true }
        val withoutGemm = missingRequiredSymbols { it != "cblas_dgemm" }
        val withoutTrsm = missingRequiredSymbols { it != "cblas_dtrsm" }

        assertEquals(emptyList(), complete)
        assertEquals(listOf("cblas_dgemm"), withoutGemm)
        assertEquals(listOf("cblas_dtrsm"), withoutTrsm)
    }

    @Test
    fun `an operation outside the required surface does not make a library partial`() {
        // gemmt is the one operation a supported vendor may legitimately lack; it composes instead.
        val withoutGemmt = missingRequiredSymbols { it != "cblas_dgemmt" }

        assertEquals(emptyList(), withoutGemmt)
    }

    @Test
    fun `a build advertising 64 bit integers is not the abi koblas binds`() {
        assertTrue(declaresWideIntegers("OpenBLAS 0.3.32 USE64BITINT DYNAMIC_ARCH"))
        assertTrue(declaresWideIntegers("BLIS 0.9.0 ilp64"))
        assertTrue(declaresWideIntegers("something INT64 something"))
        assertTrue(!declaresWideIntegers("OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH Haswell MAX_THREADS=128"))
        assertTrue(!declaresWideIntegers("Intel(R) oneAPI Math Kernel Library Version 2026.1-Product Build"))
    }

    @Test
    fun `the known answer probe is exactly representable`() {
        // The probe compares for equality, which is only legitimate because these values round-trip exactly.
        val dot = AbiProbe.x.indices.sumOf { AbiProbe.x[it] * AbiProbe.y[it] }

        assertEquals(AbiProbe.DOT, dot)
        assertEquals(4, AbiProbe.identity.size)
        assertEquals(4, AbiProbe.operand.size)
    }

    @Test
    fun `a bundled payload has one path that both runtimes read the same way`() {
        val linux = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Intel)
        val arm = HostPlatform(OperatingSystem.Linux, Architecture.Arm64, CpuVendor.Unknown)

        assertEquals("com/eignex/koblas/vendor/linux-x86_64/libmkl_rt.so.3", Bundle.path(Vendor.OneMkl, linux))
        assertEquals("com/eignex/koblas/vendor/linux-arm64/libarmpl_lp64.so", Bundle.path(Vendor.ArmPl, arm))
        assertEquals("com/eignex/koblas/vendor/linux-x86_64/libblis-mt.so.4", Bundle.path(Vendor.Aocl, linux))
    }

    @Test
    fun `what is never bundled has no bundled path`() {
        val macos = HostPlatform(OperatingSystem.MacOs, Architecture.Arm64, CpuVendor.Unknown)
        val unsupported = HostPlatform(OperatingSystem.Other, Architecture.Other, CpuVendor.Unknown)

        // Accelerate is part of the system and OpenBLAS is a bench reference, so neither ships a payload.
        assertEquals(null, Bundle.path(Vendor.Accelerate, macos))
        assertEquals(null, Bundle.path(Vendor.OpenBlas, macos))
        assertEquals(null, Bundle.path(Vendor.OneMkl, unsupported))
        assertEquals(null, Bundle.platform(unsupported))
    }

    @Test
    fun `an installed library is preferred over a bundled one`() {
        val blas = installed() ?: return skipped("bundled precedence")

        // Nothing bundles a payload into this artifact, so a resolved library is necessarily an installed one,
        // and its path is a real file rather than an extraction of a packaged resource.
        assertTrue(blas.libraryPath.startsWith("/"), "resolved ${blas.libraryPath}")
        assertTrue(
            !blas.libraryPath.contains("koblas-vendor-"),
            "an extracted payload was preferred over an installed library at ${blas.libraryPath}",
        )
    }

    @Test
    fun `two instances of one vendor are independent of each other`() {
        val first = installed() ?: return skipped("independent instances")
        val second = assertNotNull(openVendorBlas(first.vendor), "the same vendor failed to open twice")

        assertTrue(first !== second, "opening a vendor twice returned one shared instance")
        assertEquals(first.vendor, second.vendor)
        assertEquals(first.libraryPath, second.libraryPath)
        assertEquals(first.directlyImplemented, second.directlyImplemented)

        // Using one must not disturb the other: both compute the same answer from the same operands.
        val x = VectorWindow(doubleArrayOf(1.0, 2.0, 3.0, 4.0), 4)
        assertEquals(first.dot(x, x), second.dot(x, x))
    }

    @Test
    fun `an opened library confirmed its single compute thread or cannot be asked`() {
        val blas = installed() ?: return skipped("thread evidence")

        // A library that reported more than one thread never became a binding, so reaching here is the check.
        assertTrue(blas.threadEvidence in ThreadEvidence.entries, "no thread evidence recorded")
        assertTrue(blas.threadEvidence.label.startsWith("1-thread-"), "unexpected label")
    }

    @Test
    fun `a rejected call leaves the binding usable and its storage untouched`() {
        val blas = installed() ?: return skipped("failure cleanup")
        val values = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val x = VectorWindow(values, 4)
        val shorter = VectorWindow(doubleArrayOf(1.0, 2.0), 2)
        val expected = blas.dot(x, x)

        repeat(200) {
            assertFailsWith<IllegalArgumentException> { blas.dot(x, shorter) }
            assertFailsWith<IllegalArgumentException> { blas.axpy(1.0, x, shorter) }
        }

        assertEquals(expected, blas.dot(x, x), "a run of rejected calls changed what a good one returns")
        assertEquals(listOf(1.0, 2.0, 3.0, 4.0), values.toList(), "a rejected call wrote to its operand")
    }

    @Test
    fun `many failed and successful calls interleave without disturbing each other`() {
        val blas = installed() ?: return skipped("interleaved failures")
        val a = MatrixWindow(DoubleArray(9) { it + 1.0 }, 3, 3)
        val b = MatrixWindow(DoubleArray(9) { 9.0 - it }, 3, 3)
        val destination = DoubleArray(9)
        val c = MatrixWindow(destination, 3, 3)
        val mismatched = MatrixWindow(DoubleArray(4), 2, 2)
        blas.gemm(1.0, a, b, 0.0, c)
        val expected = destination.copyOf()

        repeat(100) {
            assertFailsWith<IllegalArgumentException> { blas.gemm(1.0, a, mismatched, 0.0, c) }
            destination.fill(0.0)
            blas.gemm(1.0, a, b, 0.0, c)
        }

        for (index in expected.indices) {
            assertTrue(
                abs(expected[index] - destination[index]) <= 1e-12 * maxOf(1.0, abs(expected[index])),
                "entry $index drifted across interleaved failures",
            )
        }
    }

    private fun skipped(what: String) {
        println("SKIPPED: no CBLAS library installed; $what was not verified on this host")
    }
}
