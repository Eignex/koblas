package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.*
import kotlin.test.*

class StridedBlasTest {
    @Test
    fun `norm2 over a strided view propagates a NaN entry`() {
        // A NaN never raises the running scale, so reading the zero case off the scale alone would report a
        // clean norm for a corrupt vector. The dense path and dnrm2 both answer NaN.
        val allNaN = F64StridedVectorView(doubleArrayOf(Double.NaN), 0, 1)
        val zerosAndNaN = F64StridedVectorView(doubleArrayOf(0.0, Double.NaN, 0.0), 0, 3)

        assertTrue(allNaN.norm2().isNaN(), "an all-NaN view reported ${allNaN.norm2()}")
        assertTrue(zerosAndNaN.norm2().isNaN(), "zeros with a NaN reported ${zerosAndNaN.norm2()}")
        assertEquals(0.0, F64StridedVectorView(doubleArrayOf(0.0, 0.0), 0, 2).norm2(), "an all-zero view")
    }

    @Test
    fun `strided gemv forms zero products from nonzero alpha`() {
        // The dense reference routes this through axpy so 0 times Infinity stays NaN, and
        // BlasConformanceTest pins it there. The strided default has to agree, not least because the host
        // adapter falls back to it for negative strides.
        val a = F64DenseMatrix(2, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0, 2.0, 3.0)).asView()
        val x = F64StridedVectorView(doubleArrayOf(0.0, 1.0), 0, 2)
        val storage = doubleArrayOf(0.0, 0.0)
        val y = F64StridedVectorView(storage, 0, 2)

        koblas.blas.gemv(1.0, a, x, 0.0, y)

        assertTrue(storage[0].isNaN(), "the infinite coefficient was skipped, giving ${storage[0]}")
    }

    @Test
    fun `strided gemm scales without forming products when alpha is zero`() {
        val a = F64DenseMatrix(1, 1, doubleArrayOf(Double.POSITIVE_INFINITY)).asView()
        val b = F64DenseMatrix(1, 1, doubleArrayOf(1.0)).asView()
        val storage = doubleArrayOf(5.0)
        val c = F64DenseMatrix.wrap(1, 1, storage).asView()

        koblas.blas.gemm(0.0, a, false, b, false, 0.0, c)

        assertEquals(0.0, storage[0], "alpha zero let an infinite operand reach the destination")
    }

    @Test
    fun `gemv reads a panel and strided vectors without copies`() {
        val storage = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(-1.0, 1.0, 2.0, -1.0),
                doubleArrayOf(-1.0, 3.0, 4.0, -1.0),
                doubleArrayOf(-1.0, 5.0, 6.0, -1.0),
                doubleArrayOf(-1.0, -1.0, -1.0, -1.0),
            ),
        )
        val a = storage.view(0, 3, 1, 2)
        val xBuffer = doubleArrayOf(2.0, -99.0, 3.0)
        val yBuffer = doubleArrayOf(10.0, -99.0, 20.0, -99.0, 30.0)
        val x = F64StridedVectorView(xBuffer, 0, 2, 2)
        val y = F64StridedVectorView(yBuffer, 0, 3, 2)

        F64ReferenceLinearAlgebra.gemv(2.0, a, x, 0.5, y)

        assertContentEquals(doubleArrayOf(21.0, -99.0, 46.0, -99.0, 71.0), yBuffer)
    }

    @Test
    fun `strided gemv leaves its destination alone for an empty matrix`() {
        val yBuffer = DoubleArray(3) { Double.NaN }

        F64ReferenceLinearAlgebra.gemv(
            1.0,
            F64StridedMatrixView(0, 3, DoubleArray(0)),
            F64StridedVectorView(DoubleArray(0), 0, 0),
            0.0,
            F64StridedVectorView(yBuffer, 2, 3, -1),
            transpose = true,
        )

        assertTrue(yBuffer.all(Double::isNaN), "strided quick return changed ${yBuffer.toList()}")
    }

    @Test
    fun `gemm writes a panel while preserving its surrounding buffer`() {
        val aOwner = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 99.0),
                doubleArrayOf(3.0, 4.0, 99.0),
                doubleArrayOf(99.0, 99.0, 99.0),
            ),
        )
        val bOwner = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(5.0, 6.0),
                doubleArrayOf(7.0, 8.0),
                doubleArrayOf(99.0, 99.0),
            ),
        )
        val output = F64DenseMatrix.of(Array(4) { DoubleArray(4) { -1.0 } })

        F64ReferenceLinearAlgebra.gemm(
            1.0,
            aOwner.view(0, 2, 0, 2),
            false,
            bOwner.view(0, 2, 0, 2),
            false,
            0.0,
            output.view(1, 2, 1, 2),
        )

        assertEquals(19.0, output[1, 1])
        assertEquals(22.0, output[1, 2])
        assertEquals(43.0, output[2, 1])
        assertEquals(50.0, output[2, 2])
        assertEquals(-1.0, output[0, 0])
        assertEquals(-1.0, output[3, 3])
    }

    @Test
    fun `view BLAS rejects an overlapping destination`() {
        val owner = F64DenseMatrix.diagonal(3)
        val matrix = owner.asView()
        val x = matrix.column(0)
        val y = matrix.column(1)

        assertFailsWith<IllegalArgumentException> {
            F64ReferenceLinearAlgebra.gemv(1.0, matrix, x, 0.0, y)
        }
        assertFailsWith<IllegalArgumentException> {
            F64ReferenceLinearAlgebra.gemm(1.0, matrix, false, matrix, false, 0.0, matrix)
        }
    }

    @Test
    fun `vector operations mutate borrowed rows and slices`() {
        val owner = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)),
        )
        val row = owner.asView().row(1)
        val increment = F64StridedVectorView(doubleArrayOf(1.0, 9.0, 1.0, 9.0, 1.0), 0, 3, 2)

        row.axpy(2.0, increment)
        row.scale(0.5)

        assertContentEquals(doubleArrayOf(3.0, 3.5, 4.0), row.toDoubleArray())
        assertEquals(kotlin.math.sqrt(37.25), row.norm2(), 1e-12)
    }
}
