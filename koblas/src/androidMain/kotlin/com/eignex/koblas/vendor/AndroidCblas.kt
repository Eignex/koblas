@file:Suppress("LongParameterList", "TooManyFunctions") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import dalvik.annotation.optimization.FastNative

/**
 * The JNI entry points of the bundled OpenBLAS, one per CBLAS routine with that routine's own arguments.
 *
 * ART has no foreign function interface, so these are the only way to reach native code from it. Each
 * vector operand is an array, the offset of the lowest entry the call touches, and the increment; each
 * matrix operand is the whole of its array and a leading dimension. The layout is always column-major and
 * is fixed on the native side rather than passed.
 *
 * Arrays are pinned for the call rather than copied, and released as soon as it returns. Nothing here
 * validates: every caller has already checked its operands against the call they describe.
 *
 * The functions are public members of an internal object so that their JVM names stay unmangled, which is
 * what the native symbol names are derived from.
 */
internal object AndroidCblas {
    /** Whether the bundled library loaded, settled once; a process that could not load it never retries. */
    val loaded: Boolean = try {
        System.loadLibrary(LIBRARY)
        true
    } catch (_: UnsatisfiedLinkError) {
        false
    }

    /** `cblas_ddot`. */
    @FastNative @JvmStatic
    external fun ddot(n: Int, x: DoubleArray, xOff: Int, incX: Int, y: DoubleArray, yOff: Int, incY: Int): Double

    /** `cblas_dnrm2`. */
    @FastNative @JvmStatic
    external fun dnrm2(n: Int, x: DoubleArray, xOff: Int, incX: Int): Double

    /** `cblas_dasum`. */
    @FastNative @JvmStatic
    external fun dasum(n: Int, x: DoubleArray, xOff: Int, incX: Int): Double

    /** `cblas_idamax`. */
    @FastNative @JvmStatic
    external fun idamax(n: Int, x: DoubleArray, xOff: Int, incX: Int): Int

    /** `cblas_daxpy`. */
    @FastNative @JvmStatic
    external fun daxpy(
        n: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        incX: Int,
        y: DoubleArray,
        yOff: Int,
        incY: Int,
    )

    /** `cblas_dscal`. */
    @FastNative @JvmStatic
    external fun dscal(n: Int, alpha: Double, x: DoubleArray, xOff: Int, incX: Int)

    /** `cblas_dcopy`. */
    @FastNative @JvmStatic
    external fun dcopy(n: Int, x: DoubleArray, xOff: Int, incX: Int, y: DoubleArray, yOff: Int, incY: Int)

    /** `cblas_dswap`. */
    @FastNative @JvmStatic
    external fun dswap(n: Int, x: DoubleArray, xOff: Int, incX: Int, y: DoubleArray, yOff: Int, incY: Int)

    /** `cblas_drot`. */
    @FastNative @JvmStatic
    external fun drot(
        n: Int,
        x: DoubleArray,
        xOff: Int,
        incX: Int,
        y: DoubleArray,
        yOff: Int,
        incY: Int,
        c: Double,
        s: Double,
    )

    /** `cblas_dgemv`. */
    @FastNative @JvmStatic
    external fun dgemv(
        trans: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        x: DoubleArray,
        xOff: Int,
        incX: Int,
        beta: Double,
        y: DoubleArray,
        yOff: Int,
        incY: Int,
    )

    /** `cblas_dsymv`. */
    @FastNative @JvmStatic
    external fun dsymv(
        uplo: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        x: DoubleArray,
        xOff: Int,
        incX: Int,
        beta: Double,
        y: DoubleArray,
        yOff: Int,
        incY: Int,
    )

    /** `cblas_dger`. */
    @FastNative @JvmStatic
    external fun dger(
        m: Int,
        n: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        incX: Int,
        y: DoubleArray,
        yOff: Int,
        incY: Int,
        a: DoubleArray,
        lda: Int,
    )

    /** `cblas_dsyr`. */
    @FastNative @JvmStatic
    external fun dsyr(uplo: Int, n: Int, alpha: Double, x: DoubleArray, xOff: Int, incX: Int, a: DoubleArray, lda: Int)

    /** `cblas_dsyr2`. */
    @FastNative @JvmStatic
    external fun dsyr2(
        uplo: Int,
        n: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        incX: Int,
        y: DoubleArray,
        yOff: Int,
        incY: Int,
        a: DoubleArray,
        lda: Int,
    )

    /** `cblas_dtrsv`. */
    @FastNative @JvmStatic
    external fun dtrsv(
        uplo: Int,
        trans: Int,
        diag: Int,
        n: Int,
        a: DoubleArray,
        lda: Int,
        x: DoubleArray,
        xOff: Int,
        incX: Int,
    )

    /** `cblas_dtrmv`. */
    @FastNative @JvmStatic
    external fun dtrmv(
        uplo: Int,
        trans: Int,
        diag: Int,
        n: Int,
        a: DoubleArray,
        lda: Int,
        x: DoubleArray,
        xOff: Int,
        incX: Int,
    )

    /** `cblas_dgemm`. */
    @FastNative @JvmStatic
    external fun dgemm(
        transA: Int,
        transB: Int,
        m: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    )

    /** `cblas_dsymm`. */
    @FastNative @JvmStatic
    external fun dsymm(
        side: Int,
        uplo: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    )

    /** `cblas_dsyrk`. */
    @FastNative @JvmStatic
    external fun dsyrk(
        uplo: Int,
        trans: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    )

    /** `cblas_dsyr2k`. */
    @FastNative @JvmStatic
    external fun dsyr2k(
        uplo: Int,
        trans: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    )

    /** `cblas_dtrmm`. */
    @FastNative @JvmStatic
    external fun dtrmm(
        side: Int,
        uplo: Int,
        trans: Int,
        diag: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
    )

    /** `cblas_dtrsm`. */
    @FastNative @JvmStatic
    external fun dtrsm(
        side: Int,
        uplo: Int,
        trans: Int,
        diag: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
    )

    /** `cblas_dgemmt`. */
    @FastNative @JvmStatic
    external fun dgemmt(
        uplo: Int,
        transA: Int,
        transB: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    )

    /** `openblas_get_config`, the build's description of itself. */
    @FastNative @JvmStatic
    external fun config(): String

    /** `openblas_get_corename`, the core whose kernels the dispatch chose when the library loaded. */
    @FastNative @JvmStatic
    external fun coreName(): String

    /** `openblas_get_num_threads`. */
    @FastNative @JvmStatic
    external fun threads(): Int

    /** `openblas_set_num_threads(1)`. */
    @FastNative @JvmStatic
    external fun holdToOneThread()

    /** The file the library was loaded from, as the dynamic loader reports it, or null if it cannot say. */
    @FastNative @JvmStatic
    external fun libraryPath(): String?

    /** The library's base name, which the AAR carries as `lib<name>.so` for each ABI. */
    const val LIBRARY: String = "koblas_openblas"
}
