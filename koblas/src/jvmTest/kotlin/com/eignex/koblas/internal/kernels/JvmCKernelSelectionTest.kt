package com.eignex.koblas.internal.kernels

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmCKernelSelectionTest {
    @Test
    fun `a jvm without the vector module selects c kernels`() {
        if (simdAvailable) {
            assertTrue(koblas.vectorKernels.name.startsWith("simd"))
            assertEquals("simd-sparse", koblas.sparseKernels.name)
        } else if (BuiltinEngines.c != null) {
            assertEquals(BuiltinEngines.c!!.vectorKernels.name, koblas.vectorKernels.name)
            assertEquals("c-scalar-indexed-policy", koblas.sparseKernels.name)
        } else {
            assertEquals("scalar", koblas.vectorKernels.name)
            assertEquals("scalar", koblas.sparseKernels.name)
        }
    }
}
