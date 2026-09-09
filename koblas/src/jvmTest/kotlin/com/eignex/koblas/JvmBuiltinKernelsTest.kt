package com.eignex.koblas

import com.eignex.koblas.dense.CPackedKernels
import com.eignex.koblas.dense.CPanelKernels
import com.eignex.koblas.dense.PortablePackedKernels
import com.eignex.koblas.dense.ScalarPanelKernels
import com.eignex.koblas.dense.SimdPackedKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.dense.assertLevel1KernelsAgreeWithScalar
import com.eignex.koblas.dense.assertReductionsAgreeWithScalar
import com.eignex.koblas.dense.assertSwapAgreesWithScalar
import kotlin.test.Test
import kotlin.test.assertSame

@OptIn(ExperimentalKoblasApi::class)
class JvmBuiltinKernelsTest {
    @Test
    fun `available explicit providers agree with scalar kernels`() {
        listOfNotNull(BuiltinKernels.c, BuiltinKernels.simd).forEach { provider ->
            assertLevel1KernelsAgreeWithScalar(provider.vectorKernels)
            assertReductionsAgreeWithScalar(provider.vectorKernels)
            assertSwapAgreesWithScalar(provider.vectorKernels)
        }
    }

    @Test
    fun `explicit providers retain their cohesive family implementations`() {
        assertSame(ScalarPanelKernels, BuiltinKernels.scalar.denseKernelFamilies.panel)
        assertSame(PortablePackedKernels, BuiltinKernels.scalar.denseKernelFamilies.packed)
        BuiltinKernels.c?.denseKernelFamilies?.let { families ->
            assertSame(CPanelKernels, families.panel)
            assertSame(CPackedKernels, families.packed)
        }
        BuiltinKernels.simd?.denseKernelFamilies?.let { families ->
            assertSame(SimdPanelKernels, families.panel)
            assertSame(SimdPackedKernels, families.packed)
        }
    }
}
