package com.eignex.koblas

import com.eignex.koblas.dense.assertLevel1KernelsAgreeWithReference
import com.eignex.koblas.dense.assertReductionsAgreeWithReference
import com.eignex.koblas.dense.assertSwapAgreesWithReference
import kotlin.test.Test

class JvmBuiltinEnginesTest {
    @Test
    fun `available explicit engines agree with scalar kernels`() {
        listOfNotNull(BuiltinEngines.c, BuiltinEngines.simd).forEach { engine ->
            assertLevel1KernelsAgreeWithReference(engine.vectorKernels)
            assertReductionsAgreeWithReference(engine.vectorKernels)
            assertSwapAgreesWithReference(engine.vectorKernels)
        }
    }
}
