package com.eignex.koblas.internal.kernels

import com.eignex.koblas.dense.PlatformVectorKernels
import com.eignex.koblas.dense.cKernelsAvailable
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.configuration.ImplementationNames
import com.eignex.koblas.sparse.platformSparseKernelFamilies
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmCKernelSelectionTest {
    @Test
    fun `a jvm without the vector module selects c kernels`() {
        if (simdAvailable) {
            assertTrue(PlatformVectorKernels.name.startsWith(ImplementationNames.SIMD))
            assertEquals(ImplementationNames.SIMD_SPARSE, platformSparseKernelFamilies.vector.name)
        } else if (cKernelsAvailable) {
            assertEquals(ImplementationNames.C, PlatformVectorKernels.name)
            assertEquals(ImplementationNames.C_SPARSE, platformSparseKernelFamilies.vector.name)
        } else {
            assertEquals(ImplementationNames.SCALAR, PlatformVectorKernels.name)
            assertEquals(ImplementationNames.SCALAR, platformSparseKernelFamilies.vector.name)
        }
    }
}
