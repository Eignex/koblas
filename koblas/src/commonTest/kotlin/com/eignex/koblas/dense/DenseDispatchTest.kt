package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DenseDispatchTest {
    @Test
    fun `execution and inspection use the same operation specific choice`() {
        val variant = BuiltinEngines.nativeVariants.firstOrNull() ?: return
        val raw = BuiltinEngines.exactC(variant)
        var nativeCalls = 0
        val counted = object : DenseVectorKernels by raw.vectorKernels {
            override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
                nativeCalls++
                return raw.vectorKernels.dot(a, aOff, b, bOff, len)
            }
        }
        val native = KoblasEngine(
            counted,
            raw.panelKernels,
            raw.packedKernels,
            raw.sparseKernels,
            raw.indexedSparseKernels,
            variant,
        )
        val profile = DenseProfiles.resolve(
            ProfileOverrides { key ->
            when (key) {
                "jvm.c.dot.crossover" -> "3"
                "jvm.c.sum.crossover" -> "never"
                else -> null
            }
        }
        )
        val engine = densePolicyEngine(BuiltinEngines.scalar, native, RuntimeCompetitor.JvmScalar, profile)
        val input = doubleArrayOf(1.0, 2.0, 3.0)

        engine.vectorKernels.dot(input, 0, input, 0, 2)
        assertEquals(0, nativeCalls)
        assertFalse(engine.explain(DenseOperation.Dot, 2).startsWith("native id="))
        engine.vectorKernels.dot(input, 0, input, 0, 3)
        assertEquals(1, nativeCalls)
        assertTrue(engine.explain(DenseOperation.Dot, 3).startsWith("native id="))
        assertFalse(engine.explain(DenseOperation.Sum, Int.MAX_VALUE).startsWith("native id="))
    }

    @Test
    fun `scalar and vector competitors can select different components`() {
        val variant = BuiltinEngines.nativeVariants.firstOrNull() ?: return
        val raw = BuiltinEngines.exactC(variant)
        val profile = DenseProfiles.resolve(
            ProfileOverrides { key ->
            when (key) {
                "jvm.c.dot.crossover" -> "never"
                "jvm.simd.c.dot.crossover" -> "always"
                else -> null
            }
        }
        )
        val scalar = densePolicyEngine(BuiltinEngines.scalar, raw, RuntimeCompetitor.JvmScalar, profile)
        val vector = densePolicyEngine(BuiltinEngines.scalar, raw, RuntimeCompetitor.JvmVector, profile)

        assertFalse(scalar.explain(DenseOperation.Dot, 128).startsWith("native id="))
        assertTrue(vector.explain(DenseOperation.Dot, 128).startsWith("native id="))
        assertLevel1KernelsAgreeWithReference(vector.vectorKernels)
    }

    @Test
    fun `incompatible packed geometry cannot be forced by a performance override`() {
        val variant = BuiltinEngines.nativeVariants.firstOrNull() ?: return
        val raw = BuiltinEngines.exactC(variant)
        val runtime = BuiltinEngines.scalar
        val otherShape = object : PackedKernels by runtime.packedKernels {
            override val gemmTileRows: Int = 8
        }
        val alternate = KoblasEngine(
            runtime.vectorKernels,
            runtime.panelKernels,
            otherShape,
            runtime.sparseKernels,
            runtime.indexedSparseKernels,
        )
        val profile = DenseProfiles.resolve(ProfileOverrides { "always" })
        val engine = densePolicyEngine(alternate, raw, RuntimeCompetitor.JvmVector, profile)

        assertEquals(8, engine.packedKernels.gemmTileRows)
        assertTrue(engine.explain(DenseOperation.GemmTile, 128).contains("incompatible packed geometry"))
    }

    @Test
    fun `exact native engine has no hidden small input crossover`() {
        val variant = BuiltinEngines.nativeVariants.firstOrNull() ?: return
        val engine = BuiltinEngines.exactC(variant)
        assertTrue(engine.explain(DenseOperation.Dot, 1).startsWith("native id="))
        assertTrue(engine.explain(DenseOperation.TrsmTile, 1).startsWith("native id="))
        assertLevel1KernelsAgreeWithReference(engine.vectorKernels)
    }
}
