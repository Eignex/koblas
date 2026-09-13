package com.eignex.koblas.internal.kernels

import com.eignex.koblas.NativeVariant
import java.lang.foreign.ValueLayout.*

/** Typed, exact-ID FFM calls. A binding fixes its native variant before arithmetic begins. */
internal class JvmCKernelBindings(val variant: NativeVariant) {
    init {
        NativeCatalog.requireVariant(variant)
    }

    @Suppress("LongParameterList")
    fun denseDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        val result = results.get()
        val status = denseDotHandle.invokeExact(
            1 * 16 + variant.id,
            JvmArraySegments.of(a),
            aOff,
            JvmArraySegments.of(b),
            bOff,
            len,
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun denseSsqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        val result = results.get()
        val status = denseSsqdHandle.invokeExact(
            2 * 16 + variant.id,
            JvmArraySegments.of(a),
            aOff,
            JvmArraySegments.of(b),
            bOff,
            len,
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun denseAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        val status = denseAxpyHandle.invokeExact(
            3 * 16 + variant.id,
            JvmArraySegments.of(y),
            yOff,
            alpha,
            JvmArraySegments.of(x),
            xOff,
            len,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseAxpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        val status = denseAxpyArithmeticHandle.invokeExact(
            4 * 16 + variant.id,
            JvmArraySegments.of(y),
            yOff,
            alpha,
            JvmArraySegments.of(x),
            xOff,
            len,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        val status = denseScaleHandle.invokeExact(5 * 16 + variant.id, JvmArraySegments.of(v), vOff, alpha, len) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseNrm2(v: DoubleArray, vOff: Int, len: Int): Double {
        val result = results.get()
        val status = denseNrm2Handle.invokeExact(
            6 * 16 + variant.id,
            JvmArraySegments.of(v),
            vOff,
            len,
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun denseSum(v: DoubleArray, vOff: Int, len: Int): Double {
        val result = results.get()
        val status = denseSumHandle.invokeExact(
            7 * 16 + variant.id,
            JvmArraySegments.of(v),
            vOff,
            len,
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun denseAsum(v: DoubleArray, vOff: Int, len: Int): Double {
        val result = results.get()
        val status = denseAsumHandle.invokeExact(
            8 * 16 + variant.id,
            JvmArraySegments.of(v),
            vOff,
            len,
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun denseIamax(v: DoubleArray, vOff: Int, len: Int): Int {
        val result = indices.get()
        val status = denseIamaxHandle.invokeExact(
            9 * 16 + variant.id,
            JvmArraySegments.of(v),
            vOff,
            len,
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun denseGemmTile(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
    ) {
        val status = denseGemmTileHandle.invokeExact(
            10 * 16 + variant.id, depth,
            JvmArraySegments.of(
                packedA,
            ),
            aOff, JvmArraySegments.of(packedB), bOff, JvmArraySegments.of(c), cOff, ldc,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseSwap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        val status = denseSwapHandle.invokeExact(
            11 * 16 + variant.id,
            JvmArraySegments.of(a),
            aOff,
            JvmArraySegments.of(b),
            bOff,
            len,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseDot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        val status = denseDot4Handle.invokeExact(
            12 * 16 + variant.id,
            JvmArraySegments.of(
                a,
            ),
            aOff, stride, JvmArraySegments.of(b), bOff, len, JvmArraySegments.of(out), outOff,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseAxpy4(
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
        val status = denseAxpy4Handle.invokeExact(
            13 * 16 + variant.id,
            JvmArraySegments.of(
                y,
            ),
            yOff, JvmArraySegments.of(a), aOff, stride, c0, c1, c2, c3, len,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseDotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double {
        val result = results.get()
        val status = denseDotAxpyHandle.invokeExact(
            14 * 16 + variant.id,
            JvmArraySegments.of(
                y,
            ),
            yOff, alpha,
            JvmArraySegments.of(
                a,
            ),
            aOff, JvmArraySegments.of(x), xOff, len, JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun denseRotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        h11: Double,
        h12: Double,
        h21: Double,
        h22: Double,
    ) {
        val status = denseRotmHandle.invokeExact(
            15 * 16 + variant.id,
            JvmArraySegments.of(
                x,
            ),
            xOff, xStride, JvmArraySegments.of(y), yOff, yStride, len, h11, h12, h21, h22,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun sparseDotDense(
        indices: IntArray,
        indexOff: Int,
        values: DoubleArray,
        valueOff: Int,
        len: Int,
        dense: DoubleArray,
    ): Double {
        val result = results.get()
        val status = sparseDotDenseHandle.invokeExact(
            16 * 16 + 1,
            JvmArraySegments.of(indices),
            indexOff,
            JvmArraySegments.of(values),
            valueOff,
            len,
            JvmArraySegments.of(dense),
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun sparseDotSparse(
        aIndices: IntArray,
        aValues: DoubleArray,
        aLen: Int,
        bIndices: IntArray,
        bValues: DoubleArray,
        bLen: Int,
    ): Double {
        val result = results.get()
        val status = sparseDotSparseHandle.invokeExact(
            17 * 16 + 1,
            JvmArraySegments.of(aIndices),
            JvmArraySegments.of(aValues),
            aLen,
            JvmArraySegments.of(bIndices),
            JvmArraySegments.of(bValues),
            bLen,
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun sparseAxpy(
        indices: IntArray,
        indexOff: Int,
        values: DoubleArray,
        valueOff: Int,
        len: Int,
        alpha: Double,
        dense: DoubleArray,
    ) {
        val status = sparseAxpyHandle.invokeExact(
            18 * 16 + 1,
            JvmArraySegments.of(indices),
            indexOff,
            JvmArraySegments.of(values),
            valueOff,
            len,
            alpha,
            JvmArraySegments.of(dense),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun sparseScatter(
        indices: IntArray,
        indexOff: Int,
        values: DoubleArray,
        valueOff: Int,
        len: Int,
        dense: DoubleArray,
    ) {
        val status = sparseScatterHandle.invokeExact(
            19 * 16 + 1,
            JvmArraySegments.of(indices),
            indexOff,
            JvmArraySegments.of(values),
            valueOff,
            len,
            JvmArraySegments.of(dense),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun sparseNrm2(indices: IntArray, indexOff: Int, len: Int, values: DoubleArray): Double {
        val result = results.get()
        val status = sparseNrm2Handle.invokeExact(
            20 * 16 + 1,
            JvmArraySegments.of(indices),
            indexOff,
            len,
            JvmArraySegments.of(values),
            JvmArraySegments.of(result),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
        return result[0]
    }

    @Suppress("LongParameterList")
    fun sparseGather(indices: IntArray, values: DoubleArray, len: Int, dense: DoubleArray) {
        val status = sparseGatherHandle.invokeExact(
            21 * 16 + 1,
            JvmArraySegments.of(indices),
            JvmArraySegments.of(values),
            len,
            JvmArraySegments.of(dense),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun sparseGatherZero(indices: IntArray, values: DoubleArray, len: Int, dense: DoubleArray) {
        val status = sparseGatherZeroHandle.invokeExact(
            22 * 16 + 1,
            JvmArraySegments.of(indices),
            JvmArraySegments.of(values),
            len,
            JvmArraySegments.of(dense),
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseTrsmTile(
        validRows: Int,
        order: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Int,
        unitDiag: Int,
        x: DoubleArray,
        xOff: Int,
    ) {
        val status = denseTrsmTileHandle.invokeExact(
            23 * 16 + variant.id, validRows, order,
            JvmArraySegments.of(
                packedTriangle,
            ),
            triangleOff, lower, unitDiag, JvmArraySegments.of(x), xOff,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    @Suppress("LongParameterList")
    fun denseGemmTrsmTile(
        depth: Int,
        validRows: Int,
        order: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Int,
        unitDiag: Int,
        x: DoubleArray,
        xOff: Int,
    ) {
        val status = denseGemmTrsmTileHandle.invokeExact(
            24 * 16 + variant.id, depth, validRows, order,
            JvmArraySegments.of(
                packedA,
            ),
            aOff,
            JvmArraySegments.of(
                packedB,
            ),
            bOff, JvmArraySegments.of(packedTriangle), triangleOff, lower, unitDiag, JvmArraySegments.of(x), xOff,
        ) as Int
        check(status == 0) { "native execution rejected: $status" }
    }

    // All instances call the same typed ABI symbols; static final handles let HotSpot inline the downcall.
    private companion object {
        private val library = checkNotNull(JvmNativeLibrary.library)

        private val results = ThreadLocal.withInitial { DoubleArray(1) }

        private val indices = ThreadLocal.withInitial { IntArray(1) }

        private val denseDotHandle = library.handle(
            "koblas_dense_dot_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS),
        )

        private val denseSsqdHandle = library.handle(
            "koblas_dense_ssqd_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS),
        )

        private val denseAxpyHandle = library.handle(
            "koblas_dense_axpy_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, JAVA_INT),
        )

        private val denseAxpyArithmeticHandle = library.handle(
            "koblas_dense_axpy_arithmetic_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, JAVA_INT),
        )

        private val denseScaleHandle = library.handle(
            "koblas_dense_scale_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_INT),
        )

        private val denseNrm2Handle = library.handle(
            "koblas_dense_nrm2_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS),
        )

        private val denseSumHandle = library.handle(
            "koblas_dense_sum_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS),
        )

        private val denseAsumHandle = library.handle(
            "koblas_dense_asum_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS),
        )

        private val denseIamaxHandle = library.handle(
            "koblas_dense_iamax_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS),
        )

        private val denseGemmTileHandle = library.handle(
            "koblas_dense_gemm_tile_v1",
            FfmLibrary.intOf(JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT),
        )

        private val denseSwapHandle = library.handle(
            "koblas_dense_swap_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT),
        )

        private val denseDot4Handle = library.handle(
            "koblas_dense_dot4_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT),
        )

        private val denseAxpy4Handle = library.handle(
            "koblas_dense_axpy4_v1",
            FfmLibrary.intOf(
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
                JAVA_INT,
            ),
        )

        private val denseDotAxpyHandle = library.handle(
            "koblas_dense_dot_axpy_v1",
            FfmLibrary.intOf(
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                JAVA_DOUBLE,
                ADDRESS,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                ADDRESS,
            ),
        )

        private val denseRotmHandle = library.handle(
            "koblas_dense_rotm_v1",
            FfmLibrary.intOf(
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
            ),
        )

        private val sparseDotDenseHandle = library.handle(
            "koblas_sparse_dot_dense_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS),
        )

        private val sparseDotSparseHandle = library.handle(
            "koblas_sparse_dot_sparse_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
        )

        private val sparseAxpyHandle = library.handle(
            "koblas_sparse_axpy_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS),
        )

        private val sparseScatterHandle = library.handle(
            "koblas_sparse_scatter_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS),
        )

        private val sparseNrm2Handle = library.handle(
            "koblas_sparse_nrm2_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS),
        )

        private val sparseGatherHandle = library.handle(
            "koblas_sparse_gather_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
        )

        private val sparseGatherZeroHandle = library.handle(
            "koblas_sparse_gather_zero_v1",
            FfmLibrary.intOf(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
        )

        private val denseTrsmTileHandle = library.handle(
            "koblas_dense_trsm_tile_v1",
            FfmLibrary.intOf(JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT),
        )

        private val denseGemmTrsmTileHandle = library.handle(
            "koblas_dense_gemm_trsm_tile_v1",
            FfmLibrary.intOf(
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
            ),
        )
    }
}
