package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.Uplo
import com.eignex.koblas.determinant
import com.eignex.koblas.installLinearAlgebra
import com.eignex.koblas.koblas
import com.eignex.koblas.norm1
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CblasConformanceTest {

    private val cblas = CblasLinearAlgebra()
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
    fun `the backend installs itself eagerly and install overrides either way`() {
        assertTrue(CblasLinearAlgebra.isAvailable(), "host OpenBLAS expected in the test environment")
        // Eager initialization ran before main: depending on the artifact alone activated it.
        assertEquals("cblas", koblas.name)
        try {
            installLinearAlgebra(null)
            assertEquals("reference", koblas.name)
        } finally {
            installLinearAlgebra(CblasLinearAlgebra())
        }
        assertEquals("cblas", koblas.name)
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
                    val yCblas = y0.copyOf()
                    reference.gemv(alpha, a, x, beta, yRef, transpose)
                    cblas.gemv(alpha, a, x, beta, yCblas, transpose)
                    assertClose(yRef, yCblas, context = "gemv t=$transpose a=$alpha b=$beta")
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
                        val cCblas = DenseMatrix.wrap(m, n, c0.copyOf())
                        reference.gemm(alpha, a, transposeA, b, transposeB, beta, cRef)
                        cblas.gemm(alpha, a, transposeA, b, transposeB, beta, cCblas)
                        assertClose(
                            cRef.data,
                            cCblas.data,
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
                    val cCblas = DenseMatrix.wrap(n, n, c0.copyOf())
                    reference.syrk(alpha, a, transpose, beta, cRef)
                    cblas.syrk(alpha, a, transpose, beta, cCblas)
                    assertClose(cRef.data, cCblas.data, context = "syrk t=$transpose a=$alpha b=$beta")
                    if (beta == 0.0) {
                        // With no prior C the result is the alpha term alone, which must be bit-symmetric.
                        for (i in 0 until n) {
                            for (j in 0 until i) {
                                assertEquals(cCblas.data[i * n + j], cCblas.data[j * n + i], "asymmetric at ($i;$j)")
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `syrk triangle modes match reference and leave the other triangle untouched`() {
        val rng = Random(20260934)
        for (uplo in listOf(Uplo.LOWER, Uplo.UPPER)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (beta in doubleArrayOf(0.0, -0.5)) {
                    val a = randomMatrix(rng, 6, 4)
                    val n = if (transpose) 4 else 6
                    val c0 = DoubleArray(n * n) { idx ->
                        val i = idx / n
                        val j = idx % n
                        val selected = if (uplo == Uplo.LOWER) j <= i else j >= i
                        if (!selected || beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                    }
                    val cRef = DenseMatrix.wrap(n, n, c0.copyOf())
                    val cCblas = DenseMatrix.wrap(n, n, c0.copyOf())
                    reference.syrk(0.75, a, transpose, beta, cRef, uplo)
                    cblas.syrk(0.75, a, transpose, beta, cCblas, uplo)
                    for (i in 0 until n) {
                        for (j in 0 until n) {
                            val selected = if (uplo == Uplo.LOWER) j <= i else j >= i
                            val ctx = "syrk $uplo t=$transpose b=$beta ($i;$j)"
                            if (selected) {
                                val e = cRef.data[i * n + j]
                                val v = cCblas.data[i * n + j]
                                assertTrue(abs(e - v) <= 1e-12 * maxOf(1.0, abs(e)), "$ctx: $e vs $v")
                            } else {
                                assertTrue(cCblas.data[i * n + j].isNaN(), "$ctx: untouched triangle written")
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `symv and symm match reference with the unselected triangle poisoned`() {
        val rng = Random(20260912)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 6, 13)) {
                val a = DenseMatrix(n, n)
                for (i in 0 until n) {
                    for (j in 0..i) {
                        a[if (lower) i else j, if (lower) j else i] = rng.nextDouble(-1.0, 1.0)
                        if (j != i) a[if (lower) j else i, if (lower) i else j] = Double.NaN
                    }
                }
                val x = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
                val yRef = DoubleArray(n) { Double.NaN }
                val yCblas = DoubleArray(n) { Double.NaN }
                reference.symv(0.75, a, x, 0.0, yRef, lower)
                cblas.symv(0.75, a, x, 0.0, yCblas, lower)
                assertClose(yRef, yCblas, context = "symv n=$n lower=$lower")
                val p = 3
                val b = DenseMatrix(n, p)
                for (i in 0 until n) for (j in 0 until p) b[i, j] = rng.nextDouble(-1.0, 1.0)
                val cRef = DenseMatrix(n, p)
                val cCblas = DenseMatrix(n, p)
                reference.symm(0.75, a, b, 0.0, cRef, lower)
                cblas.symm(0.75, a, b, 0.0, cCblas, lower)
                assertClose(cRef.data, cCblas.data, context = "symm n=$n lower=$lower")
                val br = DenseMatrix(p, n)
                for (i in 0 until p) for (j in 0 until n) br[i, j] = rng.nextDouble(-1.0, 1.0)
                val crRef = DenseMatrix(p, n)
                val crCblas = DenseMatrix(p, n)
                reference.symm(0.75, a, br, 0.0, crRef, lower, right = true)
                cblas.symm(0.75, a, br, 0.0, crCblas, lower, right = true)
                assertClose(crRef.data, crCblas.data, context = "symm right n=$n lower=$lower")
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
            val luCblas = cblas.factor(a)
            for (transpose in booleanArrayOf(false, true)) {
                val xRef = reference.solve(luRef, b, transpose)
                val xCblas = cblas.solve(luCblas, b, transpose)
                assertClose(xRef, xCblas, tol = 1e-10, context = "solve n=$n t=$transpose")
            }
            assertTrue(
                abs(luRef.determinant() - luCblas.determinant()) <= 1e-9 * maxOf(1.0, abs(luRef.determinant())),
                "determinant n=$n: ${luRef.determinant()} vs ${luCblas.determinant()}",
            )
        }
    }

    @Test
    fun `block solves match reference for lu and ldl`() {
        val rng = Random(20260952)
        val n = 9
        val nrhs = 4
        val a = randomMatrix(rng, n, n)
        for (i in 0 until n) a[i, i] = a[i, i] + n
        val b = randomMatrix(rng, n, nrhs)
        for (transpose in booleanArrayOf(false, true)) {
            val xRef = reference.solve(reference.factor(a), b, transpose)
            val xCblas = cblas.solve(cblas.factor(a), b, transpose)
            assertClose(xRef.data, xCblas.data, tol = 1e-10, context = "lu block t=$transpose")
        }
        val sym = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var v = rng.nextDouble(-1.0, 1.0)
                if (i == j) v += if (i % 2 == 0) 2.0 else -2.0
                sym[i, j] = v
                if (j != i) sym[j, i] = Double.NaN
            }
        }
        val xRef = reference.solve(reference.ldl(sym), b)
        val xCblas = cblas.solve(cblas.ldl(sym), b)
        assertClose(xRef.data, xCblas.data, tol = 1e-9, context = "ldl block")
    }

    @Test
    fun `factorizations interchange between backends`() {
        val rng = Random(20260731)
        val n = 12
        val a = randomMatrix(rng, n, n)
        for (i in 0 until n) a[i, i] = a[i, i] + n
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        for (transpose in booleanArrayOf(false, true)) {
            val viaCblasFactor = reference.solve(cblas.factor(a), b, transpose)
            val viaRefFactor = cblas.solve(reference.factor(a), b, transpose)
            assertClose(viaCblasFactor, viaRefFactor, tol = 1e-10, context = "interchange t=$transpose")
        }
    }

    @Test
    fun `rcond agrees with the reference estimator in magnitude`() {
        val rng = Random(20260905)
        for (n in intArrayOf(1, 6, 24)) {
            val a = randomMatrix(rng, n, n)
            for (i in 0 until n) a[i, i] = a[i, i] + n
            val anorm = norm1(a)
            val ref = reference.rcond(reference.factor(a), anorm)
            val est = cblas.rcond(cblas.factor(a), anorm)
            assertTrue(est in ref / 10.0..ref * 10.0, "n=$n: cblas $est vs reference $ref")
        }
        // Conventions match the contract exactly.
        val singular = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        assertEquals(0.0, cblas.rcond(cblas.factor(singular), norm1(singular)))
        assertEquals(1.0, cblas.rcond(cblas.factor(DenseMatrix(0, 0)), 0.0))
    }

    @Test
    fun `qr factorizations interchange between backends`() {
        val rng = Random(20260925)
        for ((m, n) in listOf(6 to 6, 10 to 4)) {
            val a = randomMatrix(rng, m, n)
            val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
            val fRef = reference.qr(a)
            val fCblas = cblas.qr(a)
            // Same b through either factorization on either backend gives the same least-squares x.
            val xs = listOf(
                reference.solveLeastSquares(fRef, b),
                reference.solveLeastSquares(fCblas, b),
                cblas.solveLeastSquares(fRef, b),
                cblas.solveLeastSquares(fCblas, b),
            )
            for ((i, x) in xs.withIndex()) {
                assertClose(xs[0], x, tol = 1e-10, context = "qr interchange ${m}x$n variant $i")
            }
            // Q from one backend's packed form applies identically on the other.
            val y = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
            assertClose(
                reference.applyQ(fCblas, y),
                cblas.applyQ(fCblas, y),
                tol = 1e-11,
                context = "applyQ on cblas factors ${m}x$n",
            )
            assertClose(
                reference.applyQ(fRef, y),
                cblas.applyQ(fRef, y),
                tol = 1e-11,
                context = "applyQ on reference factors ${m}x$n",
            )
            // The same tall factorization drives the minimum-norm solve of the wide transpose.
            val bWide = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            assertClose(
                reference.solveMinimumNorm(fRef, bWide),
                cblas.solveMinimumNorm(fCblas, bWide),
                tol = 1e-10,
                context = "solveMinimumNorm ${n}x$m",
            )
        }
    }

    @Test
    fun `ldl factorizations match and interchange between backends`() {
        val rng = Random(20260932)
        for (n in intArrayOf(1, 2, 5, 14, 33)) {
            val a = DenseMatrix(n, n)
            for (i in 0 until n) {
                for (j in 0..i) {
                    var v = rng.nextDouble(-1.0, 1.0)
                    if (i == j) v += if (i % 2 == 0) 2.0 else -2.0
                    a[i, j] = v
                    if (j != i) a[j, i] = Double.NaN
                }
            }
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val fRef = reference.ldl(a)
            val fCblas = cblas.ldl(a)
            val xs = listOf(
                reference.solve(fRef, b),
                reference.solve(fCblas, b),
                cblas.solve(fRef, b),
                cblas.solve(fCblas, b),
            )
            for ((i, x) in xs.withIndex()) {
                assertClose(xs[0], x, tol = 1e-9, context = "ldl interchange n=$n variant $i")
            }
        }
        // Conventions.
        assertTrue(cblas.ldl(DenseMatrix(3, 3)).singular)
        assertTrue(cblas.solve(cblas.ldl(DenseMatrix(0, 0)), DoubleArray(0)).isEmpty())
    }

    @Test
    fun `singular matrix sets the flag and zero determinant`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        val lu = cblas.factor(a)
        assertTrue(lu.singular)
        assertEquals(0.0, lu.determinant())
    }

    @Test
    fun `degenerate shapes honor the beta conventions`() {
        // 0x0 factor/solve round-trips.
        val empty = cblas.factor(DenseMatrix(0, 0))
        assertEquals(0, cblas.solve(empty, DoubleArray(0)).size)
        // gemv over a 0x3 matrix transposed: the sum is empty, so beta == 0 must still zero y.
        val y = DoubleArray(3) { Double.NaN }
        cblas.gemv(1.0, DenseMatrix(0, 3), DoubleArray(0), 0.0, y, transpose = true)
        assertTrue(y.all { it == 0.0 }, "gemv beta=0 with empty sum left ${y.toList()}")
        // gemm with k == 0 and beta == 0 zeroes C without reading it.
        val c = DenseMatrix.wrap(2, 2, DoubleArray(4) { Double.NaN })
        cblas.gemm(1.0, DenseMatrix(2, 0), false, DenseMatrix(0, 2), false, 0.0, c)
        assertTrue(c.data.all { it == 0.0 }, "gemm k=0 beta=0 left ${c.data.toList()}")
        // syrk with alpha == 0 reduces to the beta scale.
        val s = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        cblas.syrk(0.0, DenseMatrix(2, 5), transpose = false, beta = 2.0, c = s)
        assertClose(doubleArrayOf(2.0, 4.0, 6.0, 8.0), s.data, context = "syrk alpha=0")
    }
}
