@file:Suppress("LongParameterList") // Preserve the existing primitive window interface.

package com.eignex.koblas.internal.host

import com.eignex.koblas.StridedVector
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.vendor.Blas

/** Adapts public bindings, including their window adaptation costs. Native supplies this interface directly. */
internal class BindingVectorKernels(private val binding: Blas) : DenseVectorKernels {
    override val name: String = "${binding.vendor.vendorName.lowercase()}-level1"
    private fun window(v: DoubleArray, offset: Int, length: Int, stride: Int = 1) =
        StridedVector(v, offset, length, stride)

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = binding.dot(window(a, aOff, len, aStride), window(b, bOff, len, bStride))
    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = binding.nrm2(
        window(v, vOff, len, vStride),
    )
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int = binding.iamax(
        window(v, vOff, len, vStride),
    )
    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = binding.asum(
        window(v, vOff, len, vStride),
    )
    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) =
        binding.scal(alpha, window(v, vOff, len, vStride))
    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) = binding.axpy(alpha, window(x, xOff, len, xStride), window(y, yOff, len, yStride))
    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        binding.swap(window(a, aOff, len, aStride), window(b, bOff, len, bStride))
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        binding.rot(window(x, xOff, len), window(y, yOff, len), c, s)
}
