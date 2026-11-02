package com.eignex.koblas.vendor

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.configuration.environmentVariableOrNull
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * What a library has to satisfy before a caller reaches it, and what survives a caller's mistake. The rules
 * that do not need a library are tested against the rule itself, since a host that happens to have one
 * missing exactly the right symbol is not something a test can rely on.
 */
class VendorLoadingTest {
    private fun installed(): KoblasEngine? = Vendor.entries.firstNotNullOfOrNull {
        BuiltinEngines.vendor(
            it,
        )
    }

    @Test
    fun `a parallel armpl build with hidden controls is never accepted as serial`() {
        for (candidate in listOf("libarmpl_lp64.so", "armpl_lp64.dll", "C:\\parallel_mp\\armpl_lp64.dll")) {
            assertTrue(ThreadControl.OpenMp.permitsAbsentControl(candidate))
        }
        for (candidate in listOf("libarmpl_lp64_mp.so", "armpl_lp64_mp.dll", "C:\\ArmPL\\ARMPL_LP64_MP.DLL")) {
            assertTrue(!ThreadControl.OpenMp.permitsAbsentControl(candidate))
        }
        for (control in ThreadControl.entries.filterNot { it == ThreadControl.OpenMp }) {
            assertTrue(!control.permitsAbsentControl("library.dll"))
        }
    }

    @Test
    fun `explicitly required vendor libraries must pass the loader checks`() {
        val requested = environmentVariableOrNull("KOBLAS_REQUIRE_VENDORS") ?: return
        for (name in requested.split(',')) {
            val vendor = assertNotNull(
                Vendor.entries.singleOrNull { it.name.equals(name.trim(), ignoreCase = true) },
                "Unknown required vendor: $name",
            )

            assertNotNull(
                BuiltinEngines.vendor(vendor),
                "Required ${vendor.vendorName} failed to load or pass ABI and thread checks",
            )
        }
    }

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
    fun `a build advertising sixty four bit integers is not the abi koblas binds`() {
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
    fun `two instances of one vendor are independent of each other`() {
        val first = installed() ?: return skipped("independent instances")
        val second = assertNotNull(
            BuiltinEngines.vendor(requireNotNull(first.vendor)),
            "the same vendor failed to open twice",
        )

        assertTrue(first !== second, "opening a vendor twice returned one shared instance")
        assertEquals(first.vendor, second.vendor)
        assertEquals(first.libraryPath, second.libraryPath)
        assertEquals(first.implementations, second.implementations)

        // Using one must not disturb the other: both compute the same answer from the same operands.
        val x = DenseVector.wrap(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
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
        val x = DenseVector.wrap(values)
        val shorter = DenseVector.wrap(doubleArrayOf(1.0, 2.0))
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
        val a = DenseMatrix.wrap(3, 3, DoubleArray(9) { it + 1.0 })
        val b = DenseMatrix.wrap(3, 3, DoubleArray(9) { 9.0 - it })
        val destination = DoubleArray(9)
        val c = DenseMatrix.wrap(3, 3, destination)
        val mismatched = DenseMatrix.wrap(2, 2, DoubleArray(4))
        blas.gemm(1.0, a, false, b, false, 0.0, c)
        val expected = destination.copyOf()

        repeat(100) {
            assertFailsWith<IllegalArgumentException> { blas.gemm(1.0, a, false, mismatched, false, 0.0, c) }
            destination.fill(0.0)
            blas.gemm(1.0, a, false, b, false, 0.0, c)
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
