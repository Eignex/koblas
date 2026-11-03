package com.eignex.koblas.internal.portable

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.EngineImplementation.*
import kotlin.random.Random
import kotlin.test.Test

class PortableEngineTest {
    @Test
    fun `whole level three callbacks agree with the reference on addressed outputs`() {
        val rng = Random(20261103)
        val a = wellConditioned(4, rng)
        val b = randomMatrix(4, 4, rng)
        val poison = DenseMatrix.wrap(4, 4, DoubleArray(16) { Double.NaN })
        for (operation in listOf(Gemm, Gemmt, Symm, Syrk, Syr2k, Trmm, Trsm)) {
            for (lower in listOf(false, true)) {
                for (transpose in listOf(false, true)) {
                    for (right in listOf(false, true)) {
                        for (alpha in listOf(0.0, 0.75)) {
                            val expected = randomMatrix(6, 6, rng)
                            val actual = expected.copyOf()
                            val e = expected.view(1, 1, 4, 4)
                            val c = actual.view(1, 1, 4, 4)
                            val structure = symmetricStructure(lower)
                            val triangle = triangle(lower, unitDiag = false)
                            val input = if (alpha == 0.0) poison else a
                            when (operation) {
                                Gemm -> {
                                    ReferenceBlas.gemm(alpha, a, transpose, b, !transpose, -0.5, e)
                                    PortableEngine.computeGemm(alpha, input, transpose, b, !transpose, -0.5, c)
                                }

                                Gemmt -> {
                                    ReferenceBlas.gemmt(alpha, a, transpose, b, !transpose, -0.5, e, lower)
                                    PortableEngine.computeGemmt(
                                        alpha,
                                        input,
                                        transpose,
                                        b,
                                        !transpose,
                                        -0.5,
                                        c,
                                        structure,
                                    )
                                }

                                Symm -> {
                                    ReferenceBlas.symm(alpha, a, b, -0.5, e, lower, right)
                                    PortableEngine.computeSymm(alpha, input, structure, b, -0.5, c, right)
                                }

                                Syrk -> {
                                    ReferenceBlas.syrk(alpha, a, transpose, -0.5, e, lower)
                                    PortableEngine.computeSyrk(alpha, input, transpose, -0.5, c, structure)
                                }

                                Syr2k -> {
                                    ReferenceBlas.syr2k(alpha, a, b, transpose, -0.5, e, lower)
                                    PortableEngine.computeSyr2k(alpha, input, b, transpose, -0.5, c, structure)
                                }

                                Trmm -> {
                                    ReferenceBlas.trmm(a, e, lower, transpose, right = right, alpha = alpha)
                                    PortableEngine.computeTrmm(alpha, input, triangle, transpose, c, right)
                                }

                                Trsm -> {
                                    ReferenceBlas.trsm(a, e, lower, transpose, right = right, alpha = alpha)
                                    PortableEngine.computeTrsm(alpha, input, triangle, transpose, c, right)
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
            val structure = symmetricStructure(lower)
            val triangle = triangle(lower, unitDiag = false)
            for (transpose in listOf(false, true)) {
                for (operation in listOf(Gemv, Symv, Trsv, Trmv)) {
                    val expected = y.toDoubleArray()
                    val actual = DenseVector.wrap(expected.copyOf())
                    when (operation) {
                        Gemv -> {
                            ReferenceBlas.gemv(0.75, a, x.values, -0.5, expected, transpose)
                            PortableEngine.computeGemv(0.75, a, transpose, x, -0.5, actual)
                        }

                        Symv -> {
                            ReferenceBlas.symv(0.75, a, x.values, -0.5, expected, lower)
                            PortableEngine.computeSymv(0.75, a, structure, x, -0.5, actual)
                        }

                        Trsv -> {
                            ReferenceBlas.trsv(a, expected, lower, transpose)
                            PortableEngine.computeTrsv(a, triangle, transpose, actual)
                        }

                        Trmv -> {
                            ReferenceBlas.trmv(a, expected, lower, transpose)
                            PortableEngine.computeTrmv(a, triangle, transpose, actual)
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
                        PortableEngine.computeGer(0.75, x, y, actual)
                    }

                    Syr -> {
                        ReferenceBlas.syr(0.75, x, expected, lower)
                        PortableEngine.computeSyr(0.75, x, actual, structure)
                    }

                    Syr2 -> {
                        ReferenceBlas.syr2(0.75, x, y, expected, lower)
                        PortableEngine.computeSyr2(0.75, x, y, actual, structure)
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
