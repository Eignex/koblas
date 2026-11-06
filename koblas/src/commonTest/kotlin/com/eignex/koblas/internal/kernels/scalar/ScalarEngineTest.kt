package com.eignex.koblas.internal.kernels.scalar

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.engine.EngineImplementation.*
import com.eignex.koblas.internal.kernels.*
import kotlin.random.Random
import kotlin.test.Test

class ScalarEngineTest {
    @Test
    fun `whole level three callbacks agree with the reference on addressed outputs`() {
        val rng = Random(20261103)
        val a = wellConditioned(4, rng)
        val b = randomMatrix(4, 4, rng)
        val poison = DenseMatrix.wrap(4, 4, DoubleArray(16) { Double.NaN })
        for (operation in listOf(Gemm, Gemmt, Symm, Syrk, Syr2k, Trmm, Trsm)) {
            // Each operation varies only the flags it reads: a general product has no triangle, a symmetric
            // product no transpose, and only the symmetric and triangular products have a side.
            val lowers = if (operation == Gemm) listOf(false) else listOf(false, true)
            val transposes = if (operation == Symm) listOf(false) else listOf(false, true)
            val rights = if (operation in listOf(Symm, Trmm, Trsm)) listOf(false, true) else listOf(false)
            for (lower in lowers) {
                for (transpose in transposes) {
                    for (right in rights) {
                        for (alpha in listOf(0.0, 0.75)) {
                            val expected = randomMatrix(6, 6, rng)
                            val actual = expected.copyOf()
                            val e = expected.view(1, 1, 4, 4)
                            val c = actual.view(1, 1, 4, 4)
                            val input = if (alpha == 0.0) poison else a
                            when (operation) {
                                Gemm -> {
                                    ReferenceBlas.gemm(alpha, a, transpose, b, !transpose, -0.5, e)
                                    ScalarEngine.gemm(alpha, input, transpose, b, !transpose, -0.5, c)
                                }

                                Gemmt -> {
                                    ReferenceBlas.gemmt(alpha, a, transpose, b, !transpose, -0.5, e, lower)
                                    ScalarEngine.gemmt(
                                        alpha,
                                        input,
                                        transpose,
                                        b,
                                        !transpose,
                                        -0.5,
                                        c,
                                        lower,
                                    )
                                }

                                Symm -> {
                                    ReferenceBlas.symm(alpha, a, b, -0.5, e, lower, right)
                                    ScalarEngine.symm(alpha, input, b, -0.5, c, lower, right)
                                }

                                Syrk -> {
                                    ReferenceBlas.syrk(alpha, a, transpose, -0.5, e, lower)
                                    ScalarEngine.syrk(alpha, input, transpose, -0.5, c, lower)
                                }

                                Syr2k -> {
                                    ReferenceBlas.syr2k(alpha, a, b, transpose, -0.5, e, lower)
                                    ScalarEngine.syr2k(alpha, input, b, transpose, -0.5, c, lower)
                                }

                                Trmm -> {
                                    ReferenceBlas.trmm(a, e, lower, transpose, right = right, alpha = alpha)
                                    ScalarEngine.trmm(input, c, lower, transpose, right = right, alpha = alpha)
                                }

                                Trsm -> {
                                    ReferenceBlas.trsm(a, e, lower, transpose, right = right, alpha = alpha)
                                    ScalarEngine.trsm(input, c, lower, transpose, right = right, alpha = alpha)
                                }

                                else -> error("Unexpected operation $operation")
                            }
                            assertCallbacksAgreeWithReference(
                                expected.values,
                                actual.values,
                                "$operation $lower $transpose $right $alpha",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `whole level two callbacks agree with the reference`() {
        val rng = Random(20261104)
        val a = wellConditioned(4, rng)
        val x = DenseVector.wrap(randomVector(4, rng))
        val y = DenseVector.wrap(randomVector(4, rng))
        for (lower in listOf(false, true)) {
            for (transpose in listOf(false, true)) {
                for (operation in listOf(Gemv, Symv, Trsv, Trmv)) {
                    val expected = y.toDoubleArray()
                    val actual = DenseVector.wrap(expected.copyOf())
                    when (operation) {
                        Gemv -> {
                            ReferenceBlas.gemv(0.75, a, x.values, -0.5, expected, transpose)
                            ScalarEngine.gemv(
                                0.75,
                                a,
                                x.values,
                                x.offset,
                                x.stride,
                                -0.5,
                                actual.values,
                                actual.offset,
                                actual.stride,
                                transpose,
                            )
                        }

                        Symv -> {
                            ReferenceBlas.symv(0.75, a, x.values, -0.5, expected, lower)
                            ScalarEngine.symv(
                                0.75,
                                a,
                                x.values,
                                x.offset,
                                x.stride,
                                -0.5,
                                actual.values,
                                actual.offset,
                                actual.stride,
                                lower,
                            )
                        }

                        Trsv -> {
                            ReferenceBlas.trsv(a, expected, lower, transpose)
                            ScalarEngine.trsv(a, actual.values, actual.offset, actual.stride, lower, transpose)
                        }

                        Trmv -> {
                            ReferenceBlas.trmv(a, expected, lower, transpose)
                            ScalarEngine.trmv(a, actual.values, actual.offset, actual.stride, lower, transpose)
                        }

                        else -> error("Unexpected operation $operation")
                    }
                    assertCallbacksAgreeWithReference(expected, actual.values, "$operation $lower $transpose")
                }
            }
            for (operation in listOf(Ger, Syr, Syr2)) {
                val expected = a.copyOf()
                val actual = a.copyOf()
                when (operation) {
                    Ger -> {
                        ReferenceBlas.ger(0.75, x.values, y.values, expected)
                        ScalarEngine.ger(0.75, x.values, x.offset, x.stride, y.values, y.offset, y.stride, actual)
                    }

                    Syr -> {
                        ReferenceBlas.syr(0.75, x, expected, lower)
                        ScalarEngine.syr(0.75, x.values, x.offset, x.stride, actual, lower)
                    }

                    Syr2 -> {
                        ReferenceBlas.syr2(0.75, x, y, expected, lower)
                        ScalarEngine.syr2(
                            0.75,
                            x.values,
                            x.offset,
                            x.stride,
                            y.values,
                            y.offset,
                            y.stride,
                            actual,
                            lower,
                        )
                    }

                    else -> error("Unexpected operation $operation")
                }
                assertCallbacksAgreeWithReference(expected.values, actual.values, "$operation $lower")
            }
        }
    }

    private fun assertCallbacksAgreeWithReference(expected: DoubleArray, actual: DoubleArray, context: String) =
        assertClose(expected, actual, context)
}
