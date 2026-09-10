package com.eignex.koblas

import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import kotlin.test.Test
import kotlin.test.assertSame

class BuiltinEnginesTest {
    @Test
    fun `the scalar engine uses the scalar families`() {
        val engine = BuiltinEngines.scalar

        assertSame(ScalarIndexedSparseKernels, engine.indexedSparseKernels)
    }
}
