package com.eignex.koblas.internal.kernels

import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.F64PlatformSparseKernels
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmCKernelSelectionTest {
    @Test
    fun `a jvm without the vector module selects c kernels`() {
        if (simdAvailable) {
            assertTrue(F64PlatformKernels.name.startsWith(BackendNames.SIMD))
            assertEquals(BackendNames.SIMD_SPARSE, F64PlatformSparseKernels.name)
        } else {
            assertEquals(BackendNames.C, F64PlatformKernels.name)
            assertEquals(BackendNames.C_SPARSE, F64PlatformSparseKernels.name)
        }
    }
}
