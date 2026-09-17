package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.StridedVector
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
    private fun installed(): Blas? = openBlas()

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
    fun `a call that reaches no vendor work is not described as reaching it`() {
        val blas = installed() ?: return skipped("no-work routing")
        val empty = DenseVector.wrap(DoubleArray(0))
        val present = DenseVector.wrap(doubleArrayOf(1.0, 2.0))
        val emptyMatrix = DenseMatrix.wrap(0, 0, DoubleArray(0))
        val matrix = DenseMatrix.wrap(2, 2, DoubleArray(4) { it + 1.0 })

        val emptyDot = blas.routeOf(BlasOperation.Dot, emptyList(), listOf(empty, empty))
        val realDot = blas.routeOf(BlasOperation.Dot, emptyList(), listOf(present, present))
        val emptyGemm = blas.routeOf(BlasOperation.Gemm, listOf(matrix, matrix, emptyMatrix))

        assertEquals(RouteKind.NoWork, emptyDot.kind, "an empty dot performs no vendor work")
        assertEquals(null, emptyDot.entryPoint, "a call that never reached BLAS resolved no entry point")
        assertTrue(!emptyDot.exactlyMeasurable, "an empty call is not a measurement of the vendor")
        assertEquals(RouteKind.NoWork, emptyGemm.kind, "an empty destination performs no vendor work")
        assertEquals(RouteKind.Direct, realDot.kind, "a dot with entries does reach the vendor")
    }

    @Test
    fun `the no work rule the route reports is the one the call acts on`() {
        val blas = installed() ?: return skipped("no-work agreement")
        val untouched = doubleArrayOf(7.0, 8.0, 9.0)
        val destination = StridedVector(untouched, 0, 0)
        val source = DenseVector.wrap(DoubleArray(0))

        val route = blas.routeOf(BlasOperation.Axpy, emptyList(), listOf(source, destination))
        blas.axpy(2.0, source, destination)

        // The description said nothing ran; the call must agree by leaving the storage alone.
        assertEquals(RouteKind.NoWork, route.kind)
        assertEquals(listOf(7.0, 8.0, 9.0), untouched.toList())
    }

    @Test
    fun `two instances of one vendor are independent of each other`() {
        val first = installed() ?: return skipped("independent instances")
        val second = assertNotNull(openBlas(first.vendor), "the same vendor failed to open twice")

        assertTrue(first !== second, "opening a vendor twice returned one shared instance")
        assertEquals(first.vendor, second.vendor)
        assertEquals(first.libraryPath, second.libraryPath)
        assertEquals(first.directlyImplemented, second.directlyImplemented)

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
