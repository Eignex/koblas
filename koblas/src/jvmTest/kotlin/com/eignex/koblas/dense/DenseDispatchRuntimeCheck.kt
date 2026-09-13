package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.testutil.allocation.bytesPerIteration

/** Uninstrumented C2 allocation checks for prebound selection and mixed runtime/native vector execution. */
internal object DenseDispatchRuntimeCheck {
    @JvmStatic
    fun main(args: Array<String>) {
        val variant = checkNotNull(BuiltinEngines.nativeVariants.firstOrNull())
        val runtime = BuiltinEngines.simd ?: BuiltinEngines.scalar
        val profile = DenseProfiles.resolve(
            ProfileOverrides { key ->
                if (key == "jvm.simd.c.dot.crossover") "128" else null
            },
        )
        val engine = densePolicyEngine(runtime, BuiltinEngines.exactC(variant), RuntimeCompetitor.JvmVector, profile)
        check(!engine.explain(DenseOperation.Dot, 64).startsWith("native id="))
        check(engine.explain(DenseOperation.Dot, 256).startsWith("native id="))
        val data = DoubleArray(256) { 0.25 }
        val selection = bytesPerIteration(2000, warmup = 20_000) {
            engine.dispatch!!.usesNative(DenseOperation.Dot, 256)
        }
        val runtimeBytes = bytesPerIteration(1000, warmup = 20_000) { engine.vectorKernels.dot(data, 0, data, 0, 64) }
        val nativeBytes = bytesPerIteration(1000, warmup = 20_000) { engine.vectorKernels.dot(data, 0, data, 0, 256) }
        check(selection <= 16.0) { "selection allocated $selection bytes" }
        check(runtimeBytes <= 32.0) { "runtime mixed path allocated $runtimeBytes bytes" }
        check(nativeBytes <= 32.0) { "native mixed path allocated $nativeBytes bytes" }
        println("selection=$selection runtime=$runtimeBytes native=$nativeBytes bytes per call (${runtime.name})")
    }
}
