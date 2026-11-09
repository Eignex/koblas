package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.dense.withEngines
import com.eignex.koblas.sparse.ReferenceSparseBlas
import kotlin.test.Test
import kotlin.test.assertEquals

class SparseMatrixVectorOperationsTest {
    @Test
    fun `products sharing all coefficient buffers agree with the reference`() = withEngines { engine ->
        for (operation in operations) {
            for (xStride in intArrayOf(1, -1)) {
                for (yStride in intArrayOf(1, -1)) {
                    assertSharedInputsAgreeWithReference(engine, operation, xStride, yStride, Scratch())
                }
            }
        }
    }

    @Test
    fun `repeated sparse products reuse their scratch`() = withEngines { engine ->
        for (operation in operations) {
            val scratch = Scratch()

            repeat(3) { assertSharedInputsAgreeWithReference(engine, operation, 1, 1, scratch) }

            assertEquals(2, scratch.available(4), "$operation needs one snapshot per input")
            assertEquals(1, scratch.idleLengths(), "$operation changed scratch lengths")
        }
    }

    private fun assertSharedInputsAgreeWithReference(
        engine: KoblasEngine,
        operation: String,
        xStride: Int,
        yStride: Int,
        scratch: Scratch,
    ) {
        val a = fixture(doubleArrayOf(1.5, 2.0, 0.5, 3.0))
        val x = DenseVector.wrap(a.values, if (xStride < 0) 3 else 0, 4, xStride)
        val y = DenseVector.wrap(a.values, if (yStride < 0) 3 else 0, 4, yStride)
        val reference = fixture(a.values.copyOf())
        val expected = y.toDoubleArray()
        val symmetric = operation.startsWith("symv")
        val flag = operation.endsWith("transposed") || operation.endsWith("lower")

        if (symmetric) {
            ReferenceSparseBlas.symv(0.5, reference, x.toDoubleArray(), -0.25, expected, flag)
            engine.symv(0.5, a, x, -0.25, y, flag, scratch)
        } else {
            ReferenceSparseBlas.gemv(0.5, reference, x.toDoubleArray(), -0.25, expected, flag)
            engine.gemv(0.5, a, x, -0.25, y, flag, scratch)
        }

        assertClose(expected, y.toDoubleArray(), "${engine.name} $operation strides $xStride $yStride")
    }

    private fun fixture(values: DoubleArray): SparseMatrix = SparseMatrix.wrap(
        4,
        4,
        intArrayOf(0, 1, 2, 3, 4),
        intArrayOf(1, 0, 3, 2),
        values,
    )

    private val operations = listOf("gemv", "gemv transposed", "symv lower", "symv upper")
}
