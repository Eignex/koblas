package com.eignex.koblas

import com.eignex.koblas.dense.assertLevel1KernelsAgreeWithReference
import com.eignex.koblas.dense.assertReductionsAgreeWithReference
import com.eignex.koblas.dense.assertSwapAgreesWithReference
import kotlin.test.Test

class JvmBuiltinKernelsTest {
    @Test
    fun `available explicit providers agree with scalar kernels`() {
        listOfNotNull(BuiltinKernels.c, BuiltinKernels.simd).forEach { provider ->
            assertLevel1KernelsAgreeWithReference(provider.vectorKernels)
            assertReductionsAgreeWithReference(provider.vectorKernels)
            assertSwapAgreesWithReference(provider.vectorKernels)
        }
    }
}
