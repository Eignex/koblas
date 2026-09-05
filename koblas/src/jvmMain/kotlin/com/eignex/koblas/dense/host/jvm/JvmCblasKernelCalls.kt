package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.CblasKernelCalls
import java.lang.foreign.MemorySegment

/**
 * [CblasKernelCalls] over `java.lang.foreign`. Each array goes across as a segment over the on-heap array
 * itself, sliced to the element offset, which `Linker.Option.critical` lets the call read and write in place
 * rather than copying.
 */
@Suppress("LongParameterList") // the CBLAS signatures
internal class JvmCblasKernelCalls(private val calls: HostBlasCalls) : CblasKernelCalls {
    private fun seg(values: DoubleArray, offset: Int) =
        MemorySegment.ofArray(values).asSlice(offset.toLong() * Double.SIZE_BYTES)

    override fun ddot(n: Int, x: DoubleArray, xOffset: Int, incx: Int, y: DoubleArray, yOffset: Int, incy: Int) =
        calls.ddot.invokeExact(n, seg(x, xOffset), incx, seg(y, yOffset), incy) as Double

    override fun daxpy(
        n: Int,
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
    ) {
        calls.daxpy.invokeExact(n, alpha, seg(x, xOffset), incx, seg(y, yOffset), incy) as Unit
    }

    override fun dswap(n: Int, x: DoubleArray, xOffset: Int, incx: Int, y: DoubleArray, yOffset: Int, incy: Int) {
        calls.dswap.invokeExact(n, seg(x, xOffset), incx, seg(y, yOffset), incy) as Unit
    }

    override fun dscal(n: Int, alpha: Double, x: DoubleArray, xOffset: Int, incx: Int) {
        calls.dscal.invokeExact(n, alpha, seg(x, xOffset), incx) as Unit
    }

    override fun dnrm2(n: Int, x: DoubleArray, xOffset: Int, incx: Int) =
        calls.dnrm2.invokeExact(n, seg(x, xOffset), incx) as Double

    override fun dasum(n: Int, x: DoubleArray, xOffset: Int, incx: Int) =
        calls.dasum.invokeExact(n, seg(x, xOffset), incx) as Double

    override fun drotm(
        n: Int,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
        parameters: DoubleArray,
    ) {
        calls.drotm.invokeExact(
            n,
            seg(x, xOffset),
            incx,
            seg(y, yOffset),
            incy,
            seg(parameters, 0),
        ) as Unit
    }

    override fun drot(
        n: Int,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
        c: Double,
        s: Double,
    ) {
        calls.drot.invokeExact(n, seg(x, xOffset), incx, seg(y, yOffset), incy, c, s) as Unit
    }
}
