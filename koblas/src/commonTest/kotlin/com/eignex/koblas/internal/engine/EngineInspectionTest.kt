package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.dense.RecordingBlas
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.test.*

class EngineInspectionTest {
    @Test
    fun `host attribution follows execution across the cutoff and borrowed layouts`() {
        for (order in listOf(7, 8)) {
            for (borrowed in listOf(false, true)) {
                val host = RecordingBlas()
                val tuning = EngineTuning(
                    EngineRuntime.Native,
                    HostPlatform(OperatingSystem.Linux, Architecture.X86_64),
                    host.vendor,
                )
                val engine = BuiltinEngine("test", host = host, tuning = tuning)
                val a = if (borrowed) {
                    DenseMatrix.zero(order + 2, order + 2).view(1, 1, order, order)
                } else {
                    DenseMatrix.zero(order, order)
                }
                val b = DenseMatrix.zero(order, order)
                val c = DenseMatrix.zero(order, order)
                val expected = host.takeIf { order == 8 && !borrowed }

                engine.gemm(1.0, a, false, b, false, 0.0, c)

                assertSame(expected, engine.wholeCallFor(Gemm, order, order, order, borrowed))
                assertSame(expected, engine.routedWholeCall(Gemm, order, order, order, borrowed))
                assertEquals(expected?.id ?: "scheduled", engine.providerFor(Gemm, order, order, order, true, borrowed))
                assertEquals(if (expected == null) 0 else 1, host.calls.size)
            }
        }
    }

    @Test
    fun `exact engines are attributed without routing back into themselves`() {
        val host = RecordingBlas()
        val a = DenseMatrix.zero(3, 3)

        host.gemm(1.0, a, false, a, false, 0.0, DenseMatrix.zero(3, 3))

        assertSame(host, host.wholeCallFor(Gemm, 3, 3, 3))
        assertNull(host.routedWholeCall(Gemm, 3, 3, 3, false))
        assertEquals(host.id, host.providerFor(Gemm, 3, 3, 3, true, false))
        assertEquals(1, host.calls.size)
        assertNull(ScalarEngine.wholeCallFor(Gemm, 3, 3, 3))
        assertNull(ScalarEngine.routedWholeCall(Gemm, 3, 3, 3, false))
        assertEquals("scheduled", ScalarEngine.providerFor(Gemm, 3, 3, 3, true, false))
    }

    @Test
    fun `missing exports report the schedule that executes the call`() {
        val host = RecordingBlas(implementations = setOf(Dot))
        val engine = BuiltinEngine("test", host = host)
        val a = DenseMatrix.zero(3, 3)

        engine.gemm(1.0, a, false, a, false, 0.0, DenseMatrix.zero(3, 3))

        assertNull(engine.wholeCallFor(Gemm, 3, 3, 3))
        assertNull(engine.routedWholeCall(Gemm, 3, 3, 3, false))
        assertEquals("scheduled", engine.providerFor(Gemm, 3, 3, 3, true, false))
        assertTrue(host.calls.isEmpty())
    }
}
