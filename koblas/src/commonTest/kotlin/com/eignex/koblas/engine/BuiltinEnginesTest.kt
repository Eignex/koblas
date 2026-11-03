package com.eignex.koblas.engine

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.host.environmentVariableOrNull
import com.eignex.koblas.internal.portable.PortableEngine
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Checks the exact engines, the default Level 1 arm and the vendors a caller can open by name.
 */
class BuiltinEnginesTest {

    @Test
    fun `exact built in engines do not resolve a host library`() {
        assertEquals(null, BuiltinEngines.scalar.vendor)
        assertEquals(null, BuiltinEngines.simd?.vendor)
    }

    @Test
    fun `exact SIMD uses supplied kernels from their intrinsic minimum length`() {
        val simd = BuiltinEngines.simd ?: return
        val leaf = simd.providers.first { it.id == "jvm-simd" && EngineImplementation.Iamax in it.implementations }
            as EngineKernels

        assertEquals(
            leaf.name,
            simd.implementationFor(EngineImplementation.Iamax, leaf.minimumLength(EngineImplementation.Iamax)),
        )
    }

    // Kotlin/Native can still compose whole host calls over the owned matrix backends, which is checked
    // separately; the desktop JVM default with the Vector API is checked by JvmBackendTest.
    @Test
    fun `without the Vector API the owned matrix backends are portable`() {
        if (BuiltinEngines.simd != null) return

        assertEquals(PortableEngine.name, koblas.implementationFor(PanelWork.RankUpdate, WIDE, 4))
    }

    /** Below any arm's threshold the call overhead outweighs the arithmetic, so the portable loop runs. */
    @Test
    fun `a single entry call stays with the portable kernels`() {
        assertEquals(PortableEngine.name, koblas.implementationFor(EngineImplementation.Dot, 1))
    }

    @Test
    fun `profiles use only their explicit owned preferences`() {
        val owned = provider("jvm-simd", setOf(EngineImplementation.Sum))
        val bounded = EngineRule("jvm-simd", setOf(EngineImplementation.Sum), rows = 32..64)
        for (rules in listOf(emptyList(), listOf(bounded))) {
            val engine = configuredEngine(EngineProfile("explicit", rules), owned) { error("Unexpected vendor open") }

            for (length in listOf(16, 32, 64, 128)) {
                assertEquals(
                    if (rules.isNotEmpty() && length in 32..64) owned.name else PortableEngine.name,
                    engine.implementationFor(EngineImplementation.Sum, length),
                )
            }
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
    fun `exact vendor providers do not supply portable only operations`() {
        val engine = installed() ?: return skipped("exact vendor separation")

        assertEquals(setOf(engine.id), engine.providers.map { it.id }.toSet())
        assertTrue(EngineImplementation.Sum !in engine.implementations)
        assertFailsWith<UnsupportedOperationException> { engine.sum(doubleArrayOf(1.0), 0, 1) }
        assertEquals(null, engine.implementationFor(EngineImplementation.Sum, 1))
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

    private fun installed(): KoblasEngine? = Vendor.entries.firstNotNullOfOrNull { BuiltinEngines.vendor(it) }

    private fun skipped(what: String) {
        println("SKIPPED: no CBLAS library installed; $what was not verified on this host")
    }

    private fun provider(id: String, operations: Set<EngineImplementation>): KoblasEngine = object : EngineKernels {
        override val name = id
        override val implementations = operations
    }

    @Test
    fun `the default level one kernels agree with the scalar loops`() = assertLevel1KernelsAgreeWithReference(koblas)

    @Test
    fun `the default scaling agrees with the reference across windows`() = assertScaleAgreesWithReference(koblas)

    @Test
    fun `the default reductions agree with the scalar loops`() = assertReductionsAgreeWithReference(koblas)

    @Test
    fun `the default norm agrees with the scalar oracle at extreme magnitudes`() =
        assertNormExtremesAgreeWithReference(koblas)

    @Test
    fun `the default norm agrees with the scalar oracle across offsets and lengths`() =
        assertNormWindowsAgreeWithReference(koblas)

    @Test
    fun `the default iamax honours its contract`() = assertIamaxHonoursItsContract(koblas)

    @Test
    fun `the default swap agrees with the scalar loop`() = assertSwapAgreesWithReference(koblas)

    @Test
    fun `the default rot kernel agrees with the portable one`() = assertRotKernelAgreesWithReference(koblas)

    // The KoblasEngine.axpy contract: a zero alpha returns without evaluating the products, so infinity
    // times zero never reaches the destination.
    @Test
    fun `a zero alpha axpy evaluates no product`() {
        val x = DoubleArray(64) { Double.POSITIVE_INFINITY }
        val y = DoubleArray(64)

        koblas.axpy(y, 0, 0.0, x, 0, 64)

        assertContentEquals(DoubleArray(64), y, "zero axpy evaluated infinity times zero")
    }

    /**
     * A caller walking a matrix reaches its last row or column with an empty tail, so every routine here is
     * called with a zero length. Each must read nothing and return the identity.
     */
    @Test
    fun `every default kernel accepts a zero length run`() {
        val v = doubleArrayOf(1.0, 2.0, 3.0)

        val dot = koblas.dot(v, 3, v, 3, 0)
        val nrm2 = koblas.nrm2(v, 3, 0)
        val asum = koblas.asum(v, 3, 0)
        val sum = koblas.sum(v, 3, 0)
        val iamax = koblas.iamax(v, 3, 0)
        koblas.axpy(v, 3, 2.0, v, 3, 0)
        koblas.scale(v, 3, 2.0, 0)
        koblas.swap(v, 3, v, 3, 0)

        assertEquals(0.0, dot, "dot over nothing")
        assertEquals(0.0, nrm2, "nrm2 over nothing")
        assertEquals(0.0, asum, "asum over nothing")
        assertEquals(0.0, sum, "sum over nothing")
        assertEquals(-1, iamax, "iamax over nothing")
        assertEquals(listOf(1.0, 2.0, 3.0), v.toList(), "a zero-length write touched the vector")
    }

    /** Equal runs are staged before the vendor call, without changing arithmetic providers. */
    @Test
    fun `an aliased rotation stays on the selected vendor`() {
        val arm = koblas.implementationFor(EngineImplementation.Rot, LONG_COLUMN)
        if (koblas.providers.none { it.vendor != null && it.name == arm }) {
            println("SKIPPED: no measured host profile routes rotation to a binding on this host")
            return
        }
        val values = DoubleArray(LONG_COLUMN) { (it - 17) / 13.0 }
        val expected = values.copyOf()
        BuiltinEngines.scalar.rot(expected, 0, expected, 0, LONG_COLUMN, 0.75, -0.25)

        koblas.rot(values, 0, values, 0, LONG_COLUMN, 0.75, -0.25)

        assertEquals(arm, koblas.implementationFor(EngineImplementation.Rot, LONG_COLUMN))
        assertClose(expected, values, "aliased vendor rotation")
    }

    private companion object {
        /** Wider than any arm's crossover, so the accelerated one is reached if there is one. */
        const val WIDE = 1 shl 16

        /** Past the Native Level 1 crossover, so a contiguous rotation could reach the library. */
        const val LONG_COLUMN = 256
    }
}
