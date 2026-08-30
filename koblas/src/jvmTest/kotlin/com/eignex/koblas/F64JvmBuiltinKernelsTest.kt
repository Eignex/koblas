package com.eignex.koblas

import com.eignex.koblas.dense.assertLevel1KernelsAgreeWithScalar
import com.eignex.koblas.dense.assertReductionsAgreeWithScalar
import com.eignex.koblas.dense.assertSwapAgreesWithScalar
import kotlin.test.Test

@OptIn(ExperimentalKoblasApi::class)
class F64JvmBuiltinKernelsTest {
    @Test
    fun `available explicit providers agree with scalar kernels`() {
        listOfNotNull(F64BuiltinKernels.c, F64BuiltinKernels.simd).forEach { provider ->
            assertLevel1KernelsAgreeWithScalar(provider.kernels)
            assertReductionsAgreeWithScalar(provider.kernels)
            assertSwapAgreesWithScalar(provider.kernels)
        }
    }
}
