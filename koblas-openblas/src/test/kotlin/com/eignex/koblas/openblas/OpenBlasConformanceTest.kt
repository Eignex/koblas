package com.eignex.koblas.openblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.determinant
import com.eignex.koblas.koblas
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class OpenBlasConformanceTest {

    private val openblas = OpenBlasLinearAlgebra()
    private val reference = ReferenceLinearAlgebra

    private fun randomMatrix(rng: Random, rows: Int, cols: Int) =
        DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, tol: Double = 1e-12, context: String = "") {
        assertEquals(expected.size, actual.size, context)
        for (i in expected.indices) {
            val scale = maxOf(1.0, abs(expected[i]))
            assertTrue(
                abs(expected[i] - actual[i]) <= tol * scale,
                "$context index $i: expected ${expected[i]} actual ${actual[i]}",
            )
        }
    }

    @Test
    fun `serviceloader activates the openblas backend`() {
        assertEquals("openblas", koblas.name)
    }

    @Test
    fun `gemv matches reference across transpose and alpha beta combos`() {
        val rng = Random(20260727)
        for (transpose in booleanArrayOf(false, true)) {
            for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val a = randomMatrix(rng, 7, 5)
                    val x = DoubleArray(if (transpose) 7 else 5) { rng.nextDouble(-1.0, 1.0) }
                    val yLen = if (transpose) 5 else 7
                    val y0 = DoubleArray(yLen) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val yRef = y0.copyOf()
                    val yOpen = y0.copyOf()
                    reference.gemv(alpha, a, x, beta, yRef, transpose)
                    openblas.gemv(alpha, a, x, beta, yOpen, transpose)
                    assertClose(yRef, yOpen, context = "gemv t=$transpose a=$alpha b=$beta")
                }
            }
        }
    }

    @Test
    fun `gemm matches reference across transpose and alpha beta combos`() {
        val rng = Random(20260728)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val m = 6
                        val k = 4
                        val n = 5
                        val a = if (transposeA) randomMatrix(rng, k, m) else randomMatrix(rng, m, k)
                        val b = if (transposeB) randomMatrix(rng, n, k) else randomMatrix(rng, k, n)
                        val c0 = DoubleArray(m * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val cRef = DenseMatrix.wrap(m, n, c0.copyOf())
                        val cOpen = DenseMatrix.wrap(m, n, c0.copyOf())
                        reference.gemm(alpha, a, transposeA, b, transposeB, beta, cRef)
                        openblas.gemm(alpha, a, transposeA, b, transposeB, beta, cOpen)
                        assertClose(
                            cRef.data,
                            cOpen.data,
                            context = "gemm tA=$transposeA tB=$transposeB a=$alpha b=$beta",
                        )
                    }
                }
            }
        }
    }

    @Test
    fun `syrk matches reference and is exactly symmetric`() {
        val rng = Random(20260729)
        for (transpose in booleanArrayOf(false, true)) {
            for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val a = randomMatrix(rng, 6, 4)
                    val n = if (transpose) 4 else 6
                    val c0 = DoubleArray(n * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val cRef = DenseMatrix.wrap(n, n, c0.copyOf())
                    val cOpen = DenseMatrix.wrap(n, n, c0.copyOf())
                    reference.syrk(alpha, a, transpose, beta, cRef)
                    openblas.syrk(alpha, a, transpose, beta, cOpen)
                    assertClose(cRef.data, cOpen.data, context = "syrk t=$transpose a=$alpha b=$beta")
                    if (beta == 0.0) {
                        // With no prior C the result is the alpha term alone, which must be bit-symmetric.
                        for (i in 0 until n) {
                            for (j in 0 until i) {
                                assertEquals(cOpen.data[i * n + j], cOpen.data[j * n + i], "asymmetric at ($i;$j)")
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `factor and solve match reference in both directions`() {
        val rng = Random(20260730)
        for (n in intArrayOf(1, 3, 8, 33)) {
            val a = randomMatrix(rng, n, n)
            for (i in 0 until n) a[i, i] = a[i, i] + n
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val luRef = reference.factor(a)
            val luOpen = openblas.factor(a)
            for (transpose in booleanArrayOf(false, true)) {
                val xRef = reference.solve(luRef, b, transpose)
                val xOpen = openblas.solve(luOpen, b, transpose)
                assertClose(xRef, xOpen, tol = 1e-10, context = "solve n=$n t=$transpose")
            }
            assertTrue(
                abs(luRef.determinant() - luOpen.determinant()) <= 1e-9 * maxOf(1.0, abs(luRef.determinant())),
                "determinant n=$n: ${luRef.determinant()} vs ${luOpen.determinant()}",
            )
        }
    }

    @Test
    fun `factorizations interchange between backends`() {
        val rng = Random(20260731)
        val n = 12
        val a = randomMatrix(rng, n, n)
        for (i in 0 until n) a[i, i] = a[i, i] + n
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        for (transpose in booleanArrayOf(false, true)) {
            val viaOpenFactor = reference.solve(openblas.factor(a), b, transpose)
            val viaRefFactor = openblas.solve(reference.factor(a), b, transpose)
            assertClose(viaOpenFactor, viaRefFactor, tol = 1e-10, context = "interchange t=$transpose")
        }
    }

    @Test
    fun `singular matrix sets the flag and zero determinant`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        val lu = openblas.factor(a)
        assertTrue(lu.singular)
        assertEquals(0.0, lu.determinant())
    }

    @Test
    fun `degenerate shapes honor the beta conventions`() {
        // 0x0 factor/solve round-trips.
        val empty = openblas.factor(DenseMatrix(0, 0))
        assertEquals(0, openblas.solve(empty, DoubleArray(0)).size)
        // gemv over a 0x3 matrix transposed: the sum is empty, so beta == 0 must still zero y.
        val y = DoubleArray(3) { Double.NaN }
        openblas.gemv(1.0, DenseMatrix(0, 3), DoubleArray(0), 0.0, y, transpose = true)
        assertTrue(y.all { it == 0.0 }, "gemv beta=0 with empty sum left ${y.toList()}")
        // gemm with k == 0 and beta == 0 zeroes C without reading it.
        val c = DenseMatrix.wrap(2, 2, DoubleArray(4) { Double.NaN })
        openblas.gemm(1.0, DenseMatrix(2, 0), false, DenseMatrix(0, 2), false, 0.0, c)
        assertTrue(c.data.all { it == 0.0 }, "gemm k=0 beta=0 left ${c.data.toList()}")
        // syrk with alpha == 0 reduces to the beta scale.
        val s = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        openblas.syrk(0.0, DenseMatrix(2, 5), transpose = false, beta = 2.0, c = s)
        assertClose(doubleArrayOf(2.0, 4.0, 6.0, 8.0), s.data, context = "syrk alpha=0")
    }
}
