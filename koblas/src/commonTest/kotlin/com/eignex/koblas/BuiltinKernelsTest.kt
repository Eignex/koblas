package com.eignex.koblas

import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import kotlin.test.Test
import kotlin.test.assertSame

class BuiltinKernelsTest {
    @Test
    fun `the scalar engine uses the scalar families`() {
        val engine = BuiltinKernels.scalar

        assertSame(ScalarIndexedSparseKernels, engine.sparseKernelFamilies.indexed)
    }
}
