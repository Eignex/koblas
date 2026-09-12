package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

class DenseDispatchRuntimeTest {
    @Test
    fun `warmed selection and mixed vector execution allocate no plans`() {
        val variant = BuiltinEngines.nativeVariants.firstOrNull() ?: return
        val runtime = BuiltinEngines.simd ?: BuiltinEngines.scalar
        val profile = DenseProfiles.resolve(
            ProfileOverrides { key ->
                if (key == "jvm.simd.c.dot.crossover") "128" else null
            },
        )
        val engine = densePolicyEngine(runtime, BuiltinEngines.exactC(variant), RuntimeCompetitor.JvmVector, profile)
        val data = DoubleArray(256) { 0.25 }
        val selection = bytesPerIteration(2000, warmup = 3000) {
            engine.dispatch!!.usesNative(DenseOperation.Dot, 256)
        }
        val runtimeBytes = bytesPerIteration(1000, warmup = 3000) { engine.vectorKernels.dot(data, 0, data, 0, 64) }
        val nativeBytes = bytesPerIteration(1000, warmup = 3000) { engine.vectorKernels.dot(data, 0, data, 0, 256) }
        assertTrue(selection <= 16.0, "selection allocated $selection bytes")
        assertTrue(runtimeBytes <= 32.0, "runtime mixed path allocated $runtimeBytes bytes")
        assertTrue(nativeBytes <= 32.0, "native mixed path allocated $nativeBytes bytes")
    }
}
