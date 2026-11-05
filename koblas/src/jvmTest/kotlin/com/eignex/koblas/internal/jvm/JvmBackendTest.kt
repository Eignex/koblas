package com.eignex.koblas.internal.jvm

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.productProvider
import org.junit.Assume
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class JvmBackendTest {
    @Test
    fun `runtime loads the implementation from its packaged section`() {
        val resource = JvmBackend::class.java.getResource("JvmBackend.class").toString()
        val section = if (Runtime.version().feature() >= 25) "META-INF/versions/25/" else ""

        assertTrue(resource.contains(".jar!/${section}com/eignex/koblas/internal/jvm/JvmBackend.class"), resource)
    }

    @Test
    fun `baseline runtime selects scalar without opening a host library`() {
        Assume.assumeTrue("the desktop section replaces the baseline from Java 25", Runtime.version().feature() < 25)

        assertSame(BuiltinEngines.scalar, koblas)
        assertNull(BuiltinEngines.simd)
        assertNull(Vendor.entries.firstNotNullOfOrNull { BuiltinEngines.vendor(it) })
    }

    // An installed host takes whole products from the default, which must not change the tile the owned
    // product kernels block around.
    @Test
    fun `the default keeps the owned product tile with an installed host`() {
        val exact = BuiltinEngines.simd ?: BuiltinEngines.scalar

        assertEquals(exact.productProvider.tileRows, koblas.productProvider.tileRows)
        assertEquals(exact.productProvider.tileColumns, koblas.productProvider.tileColumns)
    }

    @Test
    fun `an eligible default product preserves an aliased input buffer`() {
        val order = 128
        val a = DenseMatrix.wrap(order, order, DoubleArray(order * order) { (it % 19 - 9) / 17.0 })
        val b = DenseMatrix.wrap(order, order, DoubleArray(order * order) { (it % 13 - 6) / 11.0 })
        val expected = DenseMatrix.wrap(order, order, a.values.copyOf())
        ReferenceBlas.gemm(0.875, a, false, b, true, -0.25, expected)

        koblas.gemm(0.875, a, false, b, true, -0.25, a, scratch = Scratch())

        assertClose(expected.values, a.values, "aliased default host product")
    }
}
