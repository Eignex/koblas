package com.eignex.koblas.internal.kernels

import com.eignex.koblas.dense.cKernelsAvailable
import com.eignex.koblas.dense.platformDenseKernelFamilies
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.sparse.platformSparseKernelFamilies
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmCKernelSelectionTest {
    @Test
    fun `a jvm without the vector module selects c kernels`() {
        if (simdAvailable) {
            assertTrue(platformDenseKernelFamilies.vector.name.startsWith("simd"))
            assertEquals("simd-sparse", platformSparseKernelFamilies.vector.name)
        } else if (cKernelsAvailable) {
            assertEquals("c", platformDenseKernelFamilies.vector.name)
            assertEquals("c-sparse", platformSparseKernelFamilies.vector.name)
        } else {
            assertEquals("scalar", platformDenseKernelFamilies.vector.name)
            assertEquals("scalar", platformSparseKernelFamilies.vector.name)
        }
    }
}
