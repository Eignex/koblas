package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.dense.withEngines
import com.eignex.koblas.internal.kernels.KernelProvider
import com.eignex.koblas.sparse.ReferenceSparseBlas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

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

    @Test
    fun `packed sparse products reuse dirty scratch and preserve padding`() = withEngines { engine ->
        for (operation in operations) {
            for (beta in doubleArrayOf(0.0, -0.25)) {
                val scratch = Scratch()
                scratch.borrow(4) { first ->
                    first.fill(Double.NaN)
                    scratch.borrow(4) { it.fill(Double.NaN) }
                }

                val a = fixture(doubleArrayOf(1.5, 2.0, 0.5, 3.0))
                val x = DenseVector.wrap(DoubleArray(9) { it * 0.25 }, 7, 4, -2)
                val y = DenseVector.wrap(DoubleArray(9) { -77.0 }, 1, 4, 2)
                repeat(3) { assertProductAgreesWithReference(engine, operation, a, x, y, beta, scratch) }

                assertEquals(2, scratch.available(4), "$operation did not reuse its packed vectors")
                assertEquals(1, scratch.idleLengths())
            }
        }
    }

    @Test
    fun `packed triangular vectors stage shared coefficients until write back`() = withEngines { engine ->
        for (solve in booleanArrayOf(false, true)) {
            val scratch = Scratch()
            repeat(3) { assertPackedTriangleAgreesWithReference(engine, solve, scratch) }
            assertEquals(1, scratch.available(3))
        }
    }

    @Test
    fun `packed vector loans return when sparse arithmetic throws`() {
        val engine = object : KernelProvider {
            override val name = "throwing"
            override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
                error("arithmetic failed")
            }
        }
        val a = fixture(doubleArrayOf(1.5, 2.0, 0.5, 3.0))
        val x = DenseVector.wrap(DoubleArray(8), 0, 4, 2)
        val y = DenseVector.wrap(DoubleArray(8), 1, 4, 2)
        val scratch = Scratch()

        assertFailsWith<IllegalStateException> { engine.gemv(0.5, a, x, 0.5, y, scratch = scratch) }

        assertEquals(2, scratch.available(4))
    }

    private fun assertPackedTriangleAgreesWithReference(engine: KoblasEngine, solve: Boolean, scratch: Scratch) {
        val a = SparseMatrix.wrap(
            3,
            3,
            intArrayOf(0, 3, 5, 6),
            intArrayOf(0, 1, 2, 1, 2, 2),
            doubleArrayOf(2.0, 1.0, -1.0, 3.0, 2.0, 4.0),
        )
        val x = DenseVector.wrap(a.values, 2, 3, -1)
        val dense = DenseMatrix.ofRows(a.toArray())
        val expected = DenseVector.wrap(x.toDoubleArray())
        val backing = a.values.copyOf()
        if (solve) {
            KoblasEngine.scalar.trsv(dense, expected, true)
        } else {
            KoblasEngine.scalar.trmv(dense, expected, true)
        }
        for (i in 0 until x.size) backing[x.offset + i * x.stride] = expected[i]

        if (solve) {
            engine.trsv(a, x, true, scratch = scratch)
        } else {
            engine.trmv(a, x, true, scratch = scratch)
        }

        assertClose(backing, a.values, "${engine.name} packed triangle solve $solve")
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
        assertProductAgreesWithReference(engine, operation, a, x, y, -0.25, scratch)
    }

    private fun assertProductAgreesWithReference(
        engine: KoblasEngine,
        operation: String,
        a: SparseMatrix,
        x: DenseVector,
        y: DenseVector,
        beta: Double,
        scratch: Scratch,
    ) {
        val expected = y.toDoubleArray()
        val backing = y.values.copyOf()
        val symmetric = operation.startsWith("symv")
        val flag = operation.endsWith("transposed") || operation.endsWith("lower")
        if (symmetric) {
            ReferenceSparseBlas.symv(0.5, a, x.toDoubleArray(), beta, expected, flag)
        } else {
            ReferenceSparseBlas.gemv(0.5, a, x.toDoubleArray(), beta, expected, flag)
        }
        for (i in expected.indices) backing[y.offset + i * y.stride] = expected[i]
        if (beta == 0.0) for (i in 0 until y.size) y[i] = Double.NaN

        if (symmetric) {
            engine.symv(0.5, a, x, beta, y, flag, scratch)
        } else {
            engine.gemv(0.5, a, x, beta, y, flag, scratch)
        }

        assertClose(backing, y.values, "${engine.name} $operation beta $beta")
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
