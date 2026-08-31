package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64PlatformKernels
import kotlin.test.Test
import kotlin.test.assertTrue

class SparseLevelOneReuseTest {

    private class RecordingKernels : F64Kernels by F64PlatformKernels {
        var axpys = 0
        var scales = 0

        override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
            axpys++
            F64PlatformKernels.axpy(y, yOff, alpha, x, xOff, len)
        }

        override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
            scales++
            F64PlatformKernels.scale(v, vOff, alpha, len)
        }
    }

    private val lower = F64SparseMatrix.ofColumns(
        3,
        3,
        listOf(
            listOf(0 to 2.0, 1 to -0.5, 2 to 0.25),
            listOf(1 to 3.0, 2 to 0.75),
            listOf(2 to 4.0),
        ),
    )

    @Test
    fun `right product sends contiguous dense columns through axpy`() {
        val kernels = RecordingKernels()
        val backend = F64ReferenceSparseBackend(kernels)
        val b = F64DenseMatrix(5, 3, DoubleArray(15) { (it + 1).toDouble() })

        backend.gemm(0.75, lower, false, b, false, 0.0, F64DenseMatrix.zero(5, 3), right = true)

        assertTrue(kernels.axpys > 0, "right gemm did not use dense axpy")
    }

    @Test
    fun `right solve sends dense column work through level one kernels`() {
        val kernels = RecordingKernels()
        val backend = F64ReferenceSparseBackend(kernels)
        val b = F64DenseMatrix(5, 3, DoubleArray(15) { (it + 1).toDouble() })

        backend.trsm(lower, b, lower = true, transpose = false, unitDiag = false, right = true, alpha = 1.0)

        assertTrue(kernels.axpys > 0, "right trsm did not use dense axpy")
        assertTrue(kernels.scales > 0, "right trsm did not use dense scale")
    }
}
