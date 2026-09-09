package com.eignex.koblas.internal.kernels

import com.eignex.koblas.dense.PlatformKernels
import com.eignex.koblas.dense.cKernelsAvailable
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.configuration.ImplementationNames
import com.eignex.koblas.sparse.PlatformSparseKernels
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmCKernelSelectionTest {
    @Test
    fun `a jvm without the vector module selects c kernels`() {
        if (simdAvailable) {
            assertTrue(PlatformKernels.name.startsWith(ImplementationNames.SIMD))
            assertEquals(ImplementationNames.SIMD_SPARSE, PlatformSparseKernels.name)
        } else if (cKernelsAvailable) {
            assertEquals(ImplementationNames.C, PlatformKernels.name)
            assertEquals(ImplementationNames.C_SPARSE, PlatformSparseKernels.name)
        } else {
            assertEquals(ImplementationNames.SCALAR, PlatformKernels.name)
            assertEquals(ImplementationNames.SCALAR, PlatformSparseKernels.name)
        }
    }
}
