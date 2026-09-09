@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.internal.configuration.ImplementationNames
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.numeric.scalarAxpy
import com.eignex.koblas.internal.numeric.scalarAxpy4
import com.eignex.koblas.internal.numeric.scalarAxpyArithmetic
import com.eignex.koblas.internal.numeric.scalarDot
import com.eignex.koblas.internal.numeric.scalarDotAxpy
import com.eignex.koblas.internal.numeric.scalarScale
import com.eignex.koblas.portableRotmg
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/**
 * The shortest run for which crossing into the C kernels pays on Kotlin/Native. The measurement behind it
 * is on [DenseTuning.nativeCMinLength]. The JVM half gates the same way at its own measured length.
 */
private val C_HOST_MIN_LENGTH = DenseTuning.nativeCMinLength

/** The C level-1 kernels compiled into each Kotlin/Native host artifact. */
internal actual object PlatformKernels : Kernels, ArithmeticKernels {
    actual override val name: String get() = ImplementationNames.C

    /**
     * The matrix-product tile, in C for the same reason every other kernel here is: a loop written in
     * Kotlin keeps a safepoint poll and two bounds checks per element and vectorises to nothing on this
     * target, where the same loop in C compiles to packed arithmetic.
     *
     * Unlike the level-1 kernels this needs no short-run guard. One call carries the whole depth of a
     * block, so it is thousands of multiply-adds behind a single crossing, and the pinning that dominates a
     * short vector call disappears into it.
     */
    @Suppress("LongParameterList")
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
        if (depth == 0) return
        packedA.usePinned { ap ->
            packedB.usePinned { bp ->
                c.usePinned { cp ->
                    koblas_dense_gemm_tile(
                        depth,
                        ap.addressOf(0),
                        aOff,
                        bp.addressOf(0),
                        bOff,
                        cp.addressOf(0),
                        cOff,
                        ldc,
                    )
                }
            }
        }
    }

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (
        len < C_HOST_MIN_LENGTH
    ) {
        scalarDot(a, aOff, b, bOff, len)
    } else {
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_dot(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        if (len < C_HOST_MIN_LENGTH) return scalarAxpy(y, yOff, alpha, x, xOff, len)
        y.usePinned { yp ->
            x.usePinned { xp -> koblas_dense_axpy(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len) }
        }
    }

    actual override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len < C_HOST_MIN_LENGTH) return scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        y.usePinned { yp ->
            x.usePinned { xp -> koblas_dense_axpy_arithmetic(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len) }
        }
    }

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (alpha == 1.0) return
        if (len < C_HOST_MIN_LENGTH) return scalarScale(v, vOff, alpha, len)
        v.usePinned { vp -> koblas_dense_scale(vp.addressOf(0), vOff, alpha, len) }
    }

    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> koblas_dense_nrm2(vp.addressOf(0), vOff, len) }

    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> koblas_dense_asum(vp.addressOf(0), vOff, len) }

    actual override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens =
        portableRotmg(d1, d2, x1, y1)

    @Suppress("LongParameterList")
    actual override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: ModifiedGivens,
    ) {
        if (transformation.flag == -2.0 || len == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                koblas_dense_rotm(
                    xp.addressOf(0),
                    xOff,
                    xStride,
                    yp.addressOf(0),
                    yOff,
                    yStride,
                    len,
                    transformation.h11,
                    transformation.h12,
                    transformation.h21,
                    transformation.h22,
                )
            }
        }
    }

    // A plane rotation is the modified Givens transformation (c, s, -s, c), so it goes to the same kernel.
    @Suppress("LongParameterList")
    actual override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (len == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                koblas_dense_rotm(xp.addressOf(0), xOff, 1, yp.addressOf(0), yOff, 1, len, c, s, -s, c)
            }
        }
    }

    actual override fun sum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { p -> koblas_dense_sum(p.addressOf(0), vOff, len) }

    actual override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (len == 0) {
        0.0
    } else {
        a.usePinned { ap ->
            b.usePinned { bp ->
                koblas_dense_ssqd(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len)
            }
        }
    }

    actual override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (len == 0) return
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_swap(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    @Suppress("LongParameterList")
    actual override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        if (len == 0) {
            for (r in 0 until 4) out[outOff + r] = 0.0
            return
        }
        a.usePinned { ap ->
            b.usePinned { bp ->
                out.usePinned { op ->
                    koblas_dense_dot4(
                        ap.addressOf(0),
                        aOff,
                        stride,
                        bp.addressOf(0),
                        bOff,
                        len,
                        op.addressOf(0),
                        outOff,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList")
    actual override fun axpy4(
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
        if (len < C_HOST_MIN_LENGTH) {
            return scalarAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
        }
        y.usePinned { yp ->
            a.usePinned { ap ->
                koblas_dense_axpy4(
                    yp.addressOf(0), yOff, ap.addressOf(0), aOff, stride, c0, c1, c2, c3, len,
                )
            }
        }
    }

    @Suppress("LongParameterList")
    actual override fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double {
        if (len < C_HOST_MIN_LENGTH) return scalarDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
        return y.usePinned { yp ->
            a.usePinned { ap ->
                x.usePinned { xp ->
                    koblas_dense_dot_axpy(
                        yp.addressOf(0),
                        yOff,
                        alpha,
                        ap.addressOf(0),
                        aOff,
                        xp.addressOf(0),
                        xOff,
                        len,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList")
    actual override fun trsmTile(
        validRows: Int,
        order: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOff: Int,
    ) {
        if (validRows == 0 || order == 0) return
        packedTriangle.usePinned { triangle ->
            x.usePinned { result ->
                koblas_dense_trsm_tile(
                    validRows,
                    order,
                    triangle.addressOf(0),
                    triangleOff,
                    if (lower) 1 else 0,
                    if (unitDiag) 1 else 0,
                    result.addressOf(0),
                    xOff,
                )
            }
        }
    }

    @Suppress("LongParameterList")
    actual override fun gemmTrsmTile(
        depth: Int,
        validRows: Int,
        order: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOff: Int,
    ) {
        if (validRows == 0 || order == 0) return
        if (depth == 0) {
            trsmTile(validRows, order, packedTriangle, triangleOff, lower, unitDiag, x, xOff)
            return
        }
        packedA.usePinned { left ->
            packedB.usePinned { right ->
                packedTriangle.usePinned { triangle ->
                    x.usePinned { result ->
                        koblas_dense_gemm_trsm_tile(
                            depth,
                            validRows,
                            order,
                            left.addressOf(0),
                            aOff,
                            right.addressOf(0),
                            bOff,
                            triangle.addressOf(0),
                            triangleOff,
                            if (lower) 1 else 0,
                            if (unitDiag) 1 else 0,
                            result.addressOf(0),
                            xOff,
                        )
                    }
                }
            }
        }
    }
}
