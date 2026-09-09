package com.eignex.koblas.internal.kernels

import com.eignex.koblas.dense.PlatformKernels
import com.eignex.koblas.dense.cKernelsAvailable
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.PlatformSparseKernels
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmCKernelSelectionTest {
    @Test
    fun `a jvm without the vector module selects c kernels`() {
        if (simdAvailable) {
            assertTrue(PlatformKernels.name.startsWith(BackendNames.SIMD))
            assertEquals(BackendNames.SIMD_SPARSE, PlatformSparseKernels.name)
        } else if (cKernelsAvailable) {
            assertEquals(BackendNames.C, PlatformKernels.name)
            assertEquals(BackendNames.C_SPARSE, PlatformSparseKernels.name)
        } else {
            assertEquals(BackendNames.SCALAR, PlatformKernels.name)
            assertEquals(BackendNames.REFERENCE, PlatformSparseKernels.name)
        }
    }
}
