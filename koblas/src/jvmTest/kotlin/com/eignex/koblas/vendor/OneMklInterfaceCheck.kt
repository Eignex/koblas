package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.copyOf
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.internal.jvm25.JvmVendorBlas
import com.eignex.koblas.internal.jvm25.JvmVendorLibrary
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.JAVA_INT
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * Checks oneMKL initialization in a fresh process, where no other test can select its interface first.
 *
 * The Gradle checks launch with an ILP64 environment. The inherited case must override it and compute
 * through LP64; the initialized case must reject an interface another caller already fixed to ILP64.
 * Symbol lookup establishes availability without calling oneMKL before the loader gets its turn.
 * CI sets `koblas.requireOneMkl` so an absent runtime fails instead of leaving these checks unverified.
 */
internal object OneMklInterfaceCheck {
    @JvmStatic
    fun main(args: Array<String>) {
        assertEquals("ILP64", System.getenv("MKL_INTERFACE_LAYER"))
        val runtimes = installedRuntimes()
        if (runtimes.isEmpty()) {
            check(System.getProperty("koblas.requireOneMkl") != "true") {
                "oneMKL interface verification requires a complete installed runtime"
            }
            println("SKIPPED: no complete oneMKL runtime installed; interface selection was not verified")
            return
        }
        when (args.single()) {
            "inherited" -> checkInheritedInterface()
            "initialized" -> checkInitializedInterface(runtimes)
            else -> error("unknown interface check")
        }
    }

    private fun checkInheritedInterface() {
        val library = assertNotNull(JvmVendorLibrary.open(Vendor.OneMkl), "installed oneMKL failed LP64 loading")
        val interfaceLayer = library.handle("MKL_Set_Interface_Layer", FunctionDescriptor.of(JAVA_INT, JAVA_INT))
        assertEquals(
            MKL_LP64,
            interfaceLayer.invokeExact(-1) as Int,
            "the inherited ILP64 interface was not overridden",
        )
        assertEquals(ThreadEvidence.Confirmed, library.threadEvidence)
        val blas = JvmVendorBlas(library)
        val x = DenseVector.wrap(vendorValues(7, 31))
        val y = DenseVector.wrap(vendorValues(7, 32))
        val expectedDot = ScalarVectorKernels.dot(x.values, 0, y.values, 0, x.size)
        val a = DenseMatrix.wrap(3, 4, vendorValues(12, 33))
        val b = DenseMatrix.wrap(4, 2, vendorValues(8, 34))
        val c = DenseMatrix.wrap(3, 2, vendorValues(6, 35))
        val expected = c.copyOf()
        ReferenceBlas.gemm(0.5, a, false, b, false, 0.25, expected)

        val dot = blas.dot(x, y)
        blas.gemm(0.5, a, false, b, false, 0.25, c)

        assertAgreesWithReference(expectedDot, dot, "oneMKL LP64 dot")
        assertAgreesWithReference(expected.values, c.values, "oneMKL LP64 gemm")
        println("oneMKL inherited ILP64 overridden: ${library.version}; LP64 dot and gemm passed")
    }

    private fun checkInitializedInterface(runtimes: List<SymbolLookup>) {
        // A second installed version is a valid fallback unless it too has already been fixed to ILP64.
        runtimes.forEach(::initializeRuntime)

        val library = JvmVendorLibrary.open(Vendor.OneMkl)

        assertNull(library, "a runtime already initialized with ILP64 must be rejected before arithmetic")
        for (lookup in runtimes) assertEquals(ILP64, selectedInterface(lookup))
        println("oneMKL already initialized with ILP64 was safely rejected")
    }

    private fun initializeRuntime(lookup: SymbolLookup) {
        val linker = Linker.nativeLinker()
        val layer = linker.downcallHandle(
            lookup.find("MKL_Set_Threading_Layer").orElseThrow(),
            FunctionDescriptor.of(JAVA_INT, JAVA_INT),
        )
        layer.invokeExact(1) as Int // Avoid an optional OpenMP dependency when initializing the runtime.
        val threads = linker.downcallHandle(
            lookup.find("MKL_Get_Max_Threads").orElseThrow(),
            FunctionDescriptor.of(JAVA_INT),
        )
        threads.invokeExact() as Int // Fix the inherited interface before Koblas opens the same runtime.
        assertEquals(ILP64, selectedInterface(lookup))
    }

    /** An invalid request queries an already selected layer without changing it. */
    private fun selectedInterface(lookup: SymbolLookup): Int {
        val handle = Linker.nativeLinker().downcallHandle(
            lookup.find("MKL_Set_Interface_Layer").orElseThrow(),
            FunctionDescriptor.of(JAVA_INT, JAVA_INT),
        )
        return handle.invokeExact(-1) as Int
    }

    private fun installedRuntimes(): List<SymbolLookup> {
        val runtimes = mutableListOf<SymbolLookup>()
        for (candidate in Vendor.OneMkl.resolvedCandidates(System.getProperty("user.home"))) {
            val lookup = try {
                SymbolLookup.libraryLookup(candidate, Arena.global())
            } catch (_: IllegalArgumentException) {
                continue
            } catch (_: UnsatisfiedLinkError) {
                continue
            }
            if (missingRequiredSymbols { lookup.find(it).isPresent }.isEmpty()) runtimes.add(lookup)
        }
        return runtimes
    }

    private const val ILP64 = 1
}
