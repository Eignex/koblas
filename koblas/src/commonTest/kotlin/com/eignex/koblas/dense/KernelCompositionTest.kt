package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertSame

class KernelCompositionTest {
    @Test
    fun `scalar composition binds separate responsibility families`() {
        val families = DenseKernelFamilies(
            ScalarKernels,
            ScalarPanelKernels,
            PortablePackedKernels,
        )

        assertSame(ScalarKernels, families.vector)
        assertSame(ScalarPanelKernels, families.panel)
        assertSame(PortablePackedKernels, families.packed)
    }

    @Test
    fun `matrix algorithms use explicit panel and packed dependencies`() {
        var panelCalls = 0
        val panel = object : DensePanelKernels by ScalarPanelKernels {
            override fun axpy4(
                y: DoubleArray,
                yOff: Int,
                a: DoubleArray,
                aOff: Int,
                stride: Int,
                c0: Double,
                c1: Double,
                c2: Double,
                c3: Double,
                len: Int,
            ) {
                panelCalls++
                ScalarPanelKernels.axpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
            }
        }
        var packedCalls = 0
        val packed = object : PackedKernels by PortablePackedKernels {
            override fun gemmTile(
                depth: Int,
                packedA: DoubleArray,
                aOff: Int,
                packedB: DoubleArray,
                bOff: Int,
                c: DoubleArray,
                cOff: Int,
                ldc: Int,
            ) {
                packedCalls++
                PortablePackedKernels.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
            }
        }
        val blas = BuiltinBlas(DenseKernelFamilies(ScalarKernels, panel, packed))

        val matrix = DenseMatrix.wrap(2, 4, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        val result = DoubleArray(2)
        blas.gemv(1.0, matrix, doubleArrayOf(1.0, 1.0, 1.0, 1.0), 0.0, result)
        val product = DenseMatrix.zero(2, 2)
        val identity = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 0.0, 0.0, 1.0))
        blas.gemm(1.0, identity, false, identity, false, 0.0, product)

        assertContentEquals(doubleArrayOf(16.0, 20.0), result)
        assertContentEquals(identity.data, product.data)
        assertEquals(1, panelCalls)
        assertEquals(1, packedCalls)
    }
}
