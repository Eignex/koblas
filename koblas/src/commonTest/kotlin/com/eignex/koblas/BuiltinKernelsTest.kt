package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertSame

@OptIn(ExperimentalKoblasApi::class)
class BuiltinKernelsTest {
    @Test
    fun `the scalar provider resolves an exact context`() {
        val provider = BuiltinKernels.scalar
        val context = ContextBuilder()
            .withBuiltinKernels(provider)
            .resolve()

        assertSame(provider.kernels, context.kernels)
        assertSame(provider.sparseKernels, context.sparseKernels)
    }
}
