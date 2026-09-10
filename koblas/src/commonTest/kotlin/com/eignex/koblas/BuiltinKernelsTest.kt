package com.eignex.koblas

import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import kotlin.test.Test
import kotlin.test.assertSame

@OptIn(ExperimentalKoblasApi::class)
class BuiltinKernelsTest {
    @Test
    fun `the scalar provider resolves an exact context`() {
        val provider = BuiltinKernels.scalar
        val context = provider.engine()

        assertSame(provider.vectorKernels, context.vectorKernels)
        assertSame(provider.sparseKernels, context.sparseKernels)
        assertSame(ScalarIndexedSparseKernels, context.sparseKernelFamilies.indexed)
    }
}
