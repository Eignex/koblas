package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SimdComponentDescriptionTest {
    @Test
    fun `packed descriptions distinguish the portable solve stage`() {
        val engine = BuiltinEngines.simd ?: return
        assertTrue(engine.explain(DenseOperation.TrsmTile, 4).startsWith("portable-solve;"))
        assertTrue(engine.explain(DenseOperation.GemmTrsmTile, 4).contains("update + portable-solve"))
        assertTrue(engine.explain(DenseOperation.GemmTrsmTile, 0).startsWith("portable-solve;"))
    }

    @Test
    fun `short runtime operations identify scalar arithmetic`() {
        val engine = BuiltinEngines.simd ?: return
        for (operation in DenseOperation.entries.filter { !it.packed }) {
            assertEquals("scalar runtime (short input)", engine.explain(operation, 1), operation.name)
        }
        if (DenseTuning.simdIamaxCrossover > SimdOps.lanes()) {
            assertEquals("scalar runtime (short input)", engine.explain(DenseOperation.Iamax, SimdOps.lanes()))
        }
    }

    @Test
    fun `data and stride dependent fallbacks remain explicit`() {
        val engine = BuiltinEngines.simd ?: return
        val length = SimdOps.lanes()
        assertTrue(engine.explain(DenseOperation.Nrm2, length).contains("depends on values"))
        assertTrue(engine.explain(DenseOperation.DotAxpy, length).contains("depends on values"))
        assertTrue(engine.explain(DenseOperation.Rotm, length).contains("nonunit strides"))
    }
}
