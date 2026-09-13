@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.NativeVariant
import kotlinx.cinterop.*

/** Native marshalling for the same typed exact-ID ABI used by the JVM. */
internal class NativeCKernelBindings(val variant: NativeVariant) {
    init {
        NativeCatalog.requireVariant(variant)
    }

    @Suppress("LongParameterList")
    fun denseDot(a: CPointer<DoubleVar>, aOff: Int, b: CPointer<DoubleVar>, bOff: Int, len: Int): Double = memScoped {
        val result = alloc<DoubleVar>()
        val status = koblas_dense_dot_v1((1 * 16 + variant.id).toUInt(), a, aOff, b, bOff, len, result.ptr)
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun denseSsqd(a: CPointer<DoubleVar>, aOff: Int, b: CPointer<DoubleVar>, bOff: Int, len: Int): Double = memScoped {
        val result = alloc<DoubleVar>()
        val status = koblas_dense_ssqd_v1((2 * 16 + variant.id).toUInt(), a, aOff, b, bOff, len, result.ptr)
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun denseAxpy(y: CPointer<DoubleVar>, yOff: Int, alpha: Double, x: CPointer<DoubleVar>, xOff: Int, len: Int) {
        val status = koblas_dense_axpy_v1((3 * 16 + variant.id).toUInt(), y, yOff, alpha, x, xOff, len)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseAxpyArithmetic(
        y: CPointer<DoubleVar>,
        yOff: Int,
        alpha: Double,
        x: CPointer<DoubleVar>,
        xOff: Int,
        len: Int,
    ) {
        val status = koblas_dense_axpy_arithmetic_v1((4 * 16 + variant.id).toUInt(), y, yOff, alpha, x, xOff, len)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseScale(v: CPointer<DoubleVar>, vOff: Int, alpha: Double, len: Int) {
        val status = koblas_dense_scale_v1((5 * 16 + variant.id).toUInt(), v, vOff, alpha, len)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseNrm2(v: CPointer<DoubleVar>, vOff: Int, len: Int): Double = memScoped {
        val result = alloc<DoubleVar>()
        val status = koblas_dense_nrm2_v1((6 * 16 + variant.id).toUInt(), v, vOff, len, result.ptr)
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun denseSum(v: CPointer<DoubleVar>, vOff: Int, len: Int): Double = memScoped {
        val result = alloc<DoubleVar>()
        val status = koblas_dense_sum_v1((7 * 16 + variant.id).toUInt(), v, vOff, len, result.ptr)
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun denseAsum(v: CPointer<DoubleVar>, vOff: Int, len: Int): Double = memScoped {
        val result = alloc<DoubleVar>()
        val status = koblas_dense_asum_v1((8 * 16 + variant.id).toUInt(), v, vOff, len, result.ptr)
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun denseIamax(v: CPointer<DoubleVar>, vOff: Int, len: Int): Int = memScoped {
        val result = alloc<IntVar>()
        val status = koblas_dense_iamax_v1((9 * 16 + variant.id).toUInt(), v, vOff, len, result.ptr)
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun denseGemmTile(
        depth: Int,
        packedA: CPointer<DoubleVar>,
        aOff: Int,
        packedB: CPointer<DoubleVar>,
        bOff: Int,
        c: CPointer<DoubleVar>,
        cOff: Int,
        ldc: Int,
    ) {
        val status =
            koblas_dense_gemm_tile_v1(
                (10 * 16 + variant.id).toUInt(),
                depth,
                packedA,
                aOff,
                packedB,
                bOff,
                c,
                cOff,
                ldc,
            )
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseSwap(a: CPointer<DoubleVar>, aOff: Int, b: CPointer<DoubleVar>, bOff: Int, len: Int) {
        val status = koblas_dense_swap_v1((11 * 16 + variant.id).toUInt(), a, aOff, b, bOff, len)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseDot4(
        a: CPointer<DoubleVar>,
        aOff: Int,
        stride: Int,
        b: CPointer<DoubleVar>,
        bOff: Int,
        len: Int,
        out: CPointer<DoubleVar>,
        outOff: Int,
    ) {
        val status = koblas_dense_dot4_v1((12 * 16 + variant.id).toUInt(), a, aOff, stride, b, bOff, len, out, outOff)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseAxpy4(
        y: CPointer<DoubleVar>,
        yOff: Int,
        a: CPointer<DoubleVar>,
        aOff: Int,
        stride: Int,
        c0: Double,
        c1: Double,
        c2: Double,
        c3: Double,
        len: Int,
    ) {
        val status =
            koblas_dense_axpy4_v1((13 * 16 + variant.id).toUInt(), y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseDotAxpy(
        y: CPointer<DoubleVar>,
        yOff: Int,
        alpha: Double,
        a: CPointer<DoubleVar>,
        aOff: Int,
        x: CPointer<DoubleVar>,
        xOff: Int,
        len: Int,
    ): Double = memScoped {
        val result = alloc<DoubleVar>()
        val status =
            koblas_dense_dot_axpy_v1((14 * 16 + variant.id).toUInt(), y, yOff, alpha, a, aOff, x, xOff, len, result.ptr)
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun denseRotm(
        x: CPointer<DoubleVar>,
        xOff: Int,
        xStride: Int,
        y: CPointer<DoubleVar>,
        yOff: Int,
        yStride: Int,
        len: Int,
        h11: Double,
        h12: Double,
        h21: Double,
        h22: Double,
    ) {
        val status =
            koblas_dense_rotm_v1(
                (15 * 16 + variant.id).toUInt(),
                x,
                xOff,
                xStride,
                y,
                yOff,
                yStride,
                len,
                h11,
                h12,
                h21,
                h22,
            )
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun sparseDotDense(
        indices: CPointer<IntVar>,
        indexOff: Int,
        values: CPointer<DoubleVar>,
        valueOff: Int,
        len: Int,
        dense: CPointer<DoubleVar>,
    ): Double = memScoped {
        val result = alloc<DoubleVar>()
        val status = koblas_sparse_dot_dense_v1(
            (16 * 16 + 1).toUInt(),
            indices,
            indexOff,
            values,
            valueOff,
            len,
            dense,
            result.ptr,
        )
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun sparseDotSparse(
        aIndices: CPointer<IntVar>,
        aValues: CPointer<DoubleVar>,
        aLen: Int,
        bIndices: CPointer<IntVar>,
        bValues: CPointer<DoubleVar>,
        bLen: Int,
    ): Double = memScoped {
        val result = alloc<DoubleVar>()
        val status = koblas_sparse_dot_sparse_v1(
            (17 * 16 + 1).toUInt(),
            aIndices,
            aValues,
            aLen,
            bIndices,
            bValues,
            bLen,
            result.ptr,
        )
        check(status == 0) { "native execution rejected: $status" }
        result.value
    }

    @Suppress("LongParameterList")
    fun sparseAxpy(
        indices: CPointer<IntVar>,
        indexOff: Int,
        values: CPointer<DoubleVar>,
        valueOff: Int,
        len: Int,
        alpha: Double,
        dense: CPointer<DoubleVar>,
    ) {
        val status = koblas_sparse_axpy_v1(
            (18 * 16 + 1).toUInt(),
            indices,
            indexOff,
            values,
            valueOff,
            len,
            alpha,
            dense,
        )
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun sparseScatter(
        indices: CPointer<IntVar>,
        indexOff: Int,
        values: CPointer<DoubleVar>,
        valueOff: Int,
        len: Int,
        dense: CPointer<DoubleVar>,
    ) {
        val status = koblas_sparse_scatter_v1((19 * 16 + 1).toUInt(), indices, indexOff, values, valueOff, len, dense)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun sparseNrm2(indices: CPointer<IntVar>, indexOff: Int, len: Int, values: CPointer<DoubleVar>): Double =
        memScoped {
            val result = alloc<DoubleVar>()
            val status = koblas_sparse_nrm2_v1((20 * 16 + 1).toUInt(), indices, indexOff, len, values, result.ptr)
            check(status == 0) { "native execution rejected: $status" }
            result.value
        }

    @Suppress("LongParameterList")
    fun sparseGather(indices: CPointer<IntVar>, values: CPointer<DoubleVar>, len: Int, dense: CPointer<DoubleVar>) {
        val status = koblas_sparse_gather_v1((21 * 16 + 1).toUInt(), indices, values, len, dense)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun sparseGatherZero(
        indices: CPointer<IntVar>,
        values: CPointer<DoubleVar>,
        len: Int,
        dense: CPointer<DoubleVar>,
    ) {
        val status = koblas_sparse_gather_zero_v1((22 * 16 + 1).toUInt(), indices, values, len, dense)
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseTrsmTile(
        validRows: Int,
        order: Int,
        packedTriangle: CPointer<DoubleVar>,
        triangleOff: Int,
        lower: Int,
        unitDiag: Int,
        x: CPointer<DoubleVar>,
        xOff: Int,
    ) {
        val status =
            koblas_dense_trsm_tile_v1(
                (23 * 16 + variant.id).toUInt(),
                validRows,
                order,
                packedTriangle,
                triangleOff,
                lower,
                unitDiag,
                x,
                xOff,
            )
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseGemmTrsmTile(
        depth: Int,
        validRows: Int,
        order: Int,
        packedA: CPointer<DoubleVar>,
        aOff: Int,
        packedB: CPointer<DoubleVar>,
        bOff: Int,
        packedTriangle: CPointer<DoubleVar>,
        triangleOff: Int,
        lower: Int,
        unitDiag: Int,
        x: CPointer<DoubleVar>,
        xOff: Int,
    ) {
        val status =
            koblas_dense_gemm_trsm_tile_v1(
                (24 * 16 + variant.id).toUInt(),
                depth,
                validRows,
                order,
                packedA,
                aOff,
                packedB,
                bOff,
                packedTriangle,
                triangleOff,
                lower,
                unitDiag,
                x,
                xOff,
            )
        check(status == 0) { "native execution rejected: $status" }
    }
}
