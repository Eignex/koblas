package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertSame

@OptIn(ExperimentalKoblasApi::class)
class F64BuiltinKernelsTest {
    @Test
    fun `the scalar provider resolves an exact context`() {
        val provider = F64BuiltinKernels.scalar
        val context = F64ContextBuilder()
            .withBuiltinKernels(provider)
            .resolve()

        assertSame(provider.kernels, context.kernels)
        assertSame(provider.sparseKernels, context.sparseKernels)
    }
}
