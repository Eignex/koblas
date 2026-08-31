package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseDecompositions
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BlasConformanceTest {

    private val eps = 2.220446049250313e-16

    private fun infNorm(v: DoubleArray): Double {
        var m = 0.0
        for (x in v) m = maxOf(m, abs(x))
        return m
    }

    private fun infNorm(a: F64DenseMatrix): Double {
        var m = 0.0
        for (i in 0 until a.rows) {
            var r = 0.0
            for (j in 0 until a.cols) r += abs(a[i, j])
            m = maxOf(m, r)
        }
        return m
    }

    private fun hilbert(n: Int): F64DenseMatrix =
        F64DenseMatrix(n, n).also { for (i in 0 until n) for (j in 0 until n) it[i, j] = 1.0 / (i + j + 1.0) }

    private fun diagonal(n: Int, rng: Random): F64DenseMatrix =
        F64DenseMatrix(n, n).also { for (i in 0 until n) it[i, i] = rng.nextDouble(1.0, 5.0) }

    private fun spd(n: Int, rng: Random): F64DenseMatrix {
        val a = F64DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        val m = F64DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                var s = 0.0
                for (k in 0 until n) s += a[k, i] * a[k, j]
                m[i, j] = s + if (i == j) n.toDouble() else 0.0
            }
        }
        return m
    }

    // The bound scales the residual by the norms of A, x and b and by n times the unit roundoff. Only the
    // residual is checked, not the forward error, which the condition number of a Hilbert matrix inflates.
    private fun assertSolveResidual(a: F64DenseMatrix, x: DoubleArray, b: DoubleArray, name: String) {
        val ax = koblas.gemv(a, x)
        val residual = DoubleArray(a.rows) { ax[it] - b[it] }
        val bound = 100.0 * a.rows * eps * (infNorm(a) * infNorm(x) + infNorm(b))
        assertTrue(infNorm(residual) <= bound + 1e-12, "$name: residual ${infNorm(residual)} > bound $bound")
    }

    @Test
    fun `dense lower syr uses contiguous axpy runs`() {
        data class AxpyCall(val yOff: Int, val xOff: Int, val len: Int)
        val calls = ArrayList<AxpyCall>()
        val recording = object : F64Kernels by F64ScalarKernels {
            override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
                calls.add(AxpyCall(yOff, xOff, len))
                F64ScalarKernels.axpy(y, yOff, alpha, x, xOff, len)
            }
        }
        val blas = F64ReferenceBlas(recording)
        val x = F64DenseVector.wrap(doubleArrayOf(2.0, -3.0, 5.0, 7.0))

        val lower = F64DenseMatrix(4, 4)
        blas.syr(1.5, x, lower, lower = true)

        assertEquals(
            listOf(AxpyCall(0, 0, 4), AxpyCall(5, 1, 3), AxpyCall(10, 2, 2), AxpyCall(15, 3, 1)),
            calls,
        )
        for (j in 0 until 4) for (i in j until 4) assertEquals(1.5 * x[i] * x[j], lower[i, j])
    }

    @Test
    fun `dense upper syr uses contiguous axpy runs`() {
        data class AxpyCall(val yOff: Int, val xOff: Int, val len: Int)
        val calls = ArrayList<AxpyCall>()
        val recording = object : F64Kernels by F64ScalarKernels {
            override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
                calls.add(AxpyCall(yOff, xOff, len))
                F64ScalarKernels.axpy(y, yOff, alpha, x, xOff, len)
            }
        }
        val upper = F64DenseMatrix(2, 2)
        F64ReferenceBlas(recording).syr(
            1.0,
            F64DenseVector.wrap(doubleArrayOf(2.0, 3.0)),
            upper,
            lower = false,
        )

        assertEquals(listOf(AxpyCall(0, 0, 1), AxpyCall(2, 0, 2)), calls)
    }

    @Test
    fun `reference symv composes dot and axpy kernels`() {
        var dots = 0
        var axpys = 0
        val recording = object : F64Kernels by F64ScalarKernels {
            override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
                dots++
                return F64ScalarKernels.dot(a, aOff, b, bOff, len)
            }

            override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
                axpys++
                F64ScalarKernels.axpy(y, yOff, alpha, x, xOff, len)
            }
        }
        val a = F64DenseMatrix(3, 3, doubleArrayOf(2.0, 3.0, 5.0, 0.0, 7.0, 11.0, 0.0, 0.0, 13.0))
        val y = DoubleArray(3)

        F64ReferenceBlas(recording).symv(1.0, a, doubleArrayOf(17.0, 19.0, 23.0), 0.0, y, lower = true)

        assertEquals(3, dots)
        assertEquals(3, axpys)
        assertEquals(doubleArrayOf(206.0, 437.0, 593.0).toList(), y.toList())
    }

    @Test
    fun `reference syr2 uses two axpy kernels per column`() {
        var axpys = 0
        val recording = object : F64Kernels by F64ScalarKernels {
            override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
                axpys++
                F64ScalarKernels.axpy(y, yOff, alpha, x, xOff, len)
            }
        }

        F64ReferenceBlas(recording).syr2(
            1.0,
            F64DenseVector.wrap(doubleArrayOf(2.0, 3.0, 5.0)),
            F64DenseVector.wrap(doubleArrayOf(7.0, 11.0, 13.0)),
            F64DenseMatrix(3, 3),
        )

        assertEquals(6, axpys)
    }

    @Test
    fun `dense LU solve has a small residual on the standard test matrices`() {
        val rng = Random(20260716)
        for (n in intArrayOf(1, 2, 5, 12, 40)) {
            val matrices = listOf(
                "identity" to F64DenseMatrix.diagonal(n),
                "diagonal" to diagonal(n, rng),
                "hilbert" to hilbert(n),
                "random" to wellConditioned(n, rng),
                "spd" to spd(n, rng),
            )
            for ((label, a) in matrices) {
                val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
                val b = koblas.gemv(a, xTrue)
                val lu = a.lu()
                if (lu.singular) continue
                val x = lu.solve(b)
                assertSolveResidual(a, x, b, "LU/$label/n=$n")
            }
        }
    }

    @Test
    fun `dense LU transpose-solve has a small residual`() {
        val rng = Random(99)
        for (n in intArrayOf(2, 7, 25)) {
            val a = wellConditioned(n, rng)
            val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
            val b = koblas.gemv(a, xTrue, transpose = true)
            val x = a.lu().solve(b, transpose = true)
            val atx = koblas.gemv(a, x, transpose = true)
            val residual = DoubleArray(n) { atx[it] - b[it] }
            val bound = 100.0 * n * eps * (infNorm(a) * infNorm(x) + infNorm(b))
            assertTrue(infNorm(residual) <= bound + 1e-12, "transpose-solve n=$n: ${infNorm(residual)} > $bound")
        }
    }

    @Test
    fun `cholesky solve has a small residual on SPD matrices`() {
        val rng = Random(7)
        for (n in intArrayOf(1, 3, 10, 30)) {
            val a = spd(n, rng)
            val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
            val b = koblas.gemv(a, xTrue)
            val l = a.cholesky()
            val x = l.solve(b)
            assertSolveResidual(a, x, b, "cholesky/n=$n")
        }
    }

    @Test
    fun `full gemv matches the naive reference across alpha beta and transpose`() {
        val rng = Random(20260727)
        for (n in intArrayOf(1, 4, 17)) {
            val m = n + 3 // non-square to catch row/col mixups
            val a = F64DenseMatrix(m, n, DoubleArray(m * n) { rng.nextDouble(-1.0, 1.0) })
            for (transpose in booleanArrayOf(false, true)) {
                val xLen = if (transpose) m else n
                val yLen = if (transpose) n else m
                for (alpha in doubleArrayOf(0.0, 1.0, -1.5)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        val x = DoubleArray(xLen) { rng.nextDouble(-2.0, 2.0) }
                        // beta == 0 must overwrite without reading, so y starts poisoned with NaN.
                        val y0 = DoubleArray(yLen) { if (beta == 0.0) Double.NaN else rng.nextDouble(-2.0, 2.0) }
                        val expected = DoubleArray(yLen) { i ->
                            var s = 0.0
                            for (k in 0 until xLen) s += (if (transpose) a[k, i] else a[i, k]) * x[k]
                            alpha * s + (if (beta == 0.0) 0.0 else beta * y0[i])
                        }
                        val y = y0.copyOf()
                        koblas.gemv(alpha, a, x, beta, y, transpose)
                        val bound = 100.0 * maxOf(m, n) * eps * (infNorm(a) * infNorm(x) + infNorm(expected))
                        for (i in 0 until yLen) {
                            assertTrue(
                                abs(y[i] - expected[i]) <= bound + 1e-12,
                                "gemv n=$n t=$transpose a=$alpha b=$beta at $i: ${y[i]} vs ${expected[i]}",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `gemv forms zero products from nonzero alpha`() {
        val a = F64DenseMatrix(2, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0, 2.0, 3.0))
        val y = DoubleArray(2)

        F64ReferenceLinearAlgebra.gemv(1.0, a, doubleArrayOf(0.0, 1.0), 0.0, y)

        assertTrue(y[0].isNaN(), "zero times infinity was ${y[0]}")
        assertEquals(3.0, y[1])
    }

    @Test
    fun `invertSpd produces an inverse with a small identity residual`() {
        val rng = Random(20260805)
        for (n in intArrayOf(1, 3, 10, 30)) {
            val a = spd(n, rng)
            val inv = a.cholesky().invert()
            val prod = a * inv
            var maxOffIdentity = 0.0
            for (i in 0 until n) {
                for (j in 0 until n) {
                    maxOffIdentity = maxOf(maxOffIdentity, abs(prod[i, j] - if (i == j) 1.0 else 0.0))
                }
            }
            val bound = 100.0 * n * eps * infNorm(a) * infNorm(inv)
            assertTrue(maxOffIdentity <= bound + 1e-12, "invertSpd n=$n: $maxOffIdentity > $bound")
        }
    }

    @Test
    fun `gemm reproduces the identity and is associative within tolerance`() {
        val rng = Random(3)
        for (n in intArrayOf(1, 4, 16)) {
            val a = wellConditioned(n, rng)
            val id = F64DenseMatrix.diagonal(n)
            assertEquals(a * id, a, "A·I != A at n=$n")
            val b = F64DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val c = F64DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val left = a * b * c
            val right = a * (b * c)
            val bound = 100.0 * n * eps * infNorm(a) * infNorm(b) * infNorm(c)
            var maxDiff = 0.0
            for (k in left.data.indices) maxDiff = maxOf(maxDiff, abs(left.data[k] - right.data[k]))
            assertTrue(maxDiff <= bound + 1e-9, "gemm associativity n=$n: $maxDiff > $bound")
        }
    }

    @Test
    fun `gemm forms zero products in every transpose mode`() {
        val a = F64DenseMatrix.diagonal(2).also { it[0, 0] = Double.POSITIVE_INFINITY }
        val b = F64DenseMatrix(2, 2)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val c = F64DenseMatrix(2, 2)

                F64ReferenceLinearAlgebra.gemm(1.0, a, transposeA, b, transposeB, 0.0, c)

                assertTrue(c[0, 0].isNaN(), "transposeA=$transposeA transposeB=$transposeB produced ${c[0, 0]}")
            }
        }
    }

    @Test
    fun `full gemm matches the naive reference across transpose flags alpha and beta`() {
        val rng = Random(20260728)
        // Two shapes, so the doubly-transposed case transposes A in one and B in the other: it copies
        // whichever operand is smaller, and one shape alone would only ever reach one of those.
        for ((m, k, n) in listOf(Triple(5, 7, 4), Triple(3, 7, 6))) {
            checkGemmShape(rng, m, k, n)
        }
    }

    @Test
    fun `gemm agrees with a naive product across every cache tile boundary`() {
        val rng = Random(20261031)
        val m = REFERENCE_MC + 7
        val k = REFERENCE_KC + 3
        val n = REFERENCE_NC + 3
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val a = if (transposeA) randomMatrix(k, m, rng) else randomMatrix(m, k, rng)
                val b = if (transposeB) randomMatrix(n, k, rng) else randomMatrix(k, n, rng)
                val expected = F64DenseMatrix(m, n)
                for (j in 0 until n) {
                    for (i in 0 until m) {
                        var sum = 0.0
                        for (p in 0 until k) {
                            val av = if (transposeA) a[p, i] else a[i, p]
                            val bv = if (transposeB) b[j, p] else b[p, j]
                            sum += av * bv
                        }
                        expected[i, j] = sum
                    }
                }
                val actual = F64DenseMatrix(m, n)
                F64ReferenceLinearAlgebra.gemm(1.0, a, transposeA, b, transposeB, 0.0, actual)
                assertClose(expected, actual, "gemm tA=$transposeA tB=$transposeB", tolerance = 1e-10)
            }
        }
    }

    @Test
    fun `gemm with a transposed left operand uses dot kernels`() {
        var dots = 0
        var dotFours = 0
        val recording = object : F64Kernels by F64ScalarKernels {
            override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
                dots++
                return F64ScalarKernels.dot(a, aOff, b, bOff, len)
            }

            override fun dot4(
                a: DoubleArray,
                aOff: Int,
                stride: Int,
                b: DoubleArray,
                bOff: Int,
                len: Int,
                out: DoubleArray,
                outOff: Int,
            ) {
                dotFours++
                F64ScalarKernels.dot4(a, aOff, stride, b, bOff, len, out, outOff)
            }
        }
        val a = F64DenseMatrix(7, 9, DoubleArray(63) { it.toDouble() / 31.0 })
        val b = F64DenseMatrix(7, 3, DoubleArray(21) { it.toDouble() / 17.0 })

        F64ReferenceBlas(recording).gemm(1.0, a, true, b, false, 0.0, F64DenseMatrix(9, 3))

        assertTrue(dots > 0 || dotFours > 0, "transposed A did not use dot kernels")
    }

    @Test
    fun `double transpose gemm packs the smaller operand`() {
        fun kernelCalls(m: Int, n: Int): Pair<Int, Int> {
            var dots = 0
            var axpys = 0
            val recording = object : F64Kernels by F64ScalarKernels {
                override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
                    dots++
                    return F64ScalarKernels.dot(a, aOff, b, bOff, len)
                }

                override fun dot4(
                    a: DoubleArray,
                    aOff: Int,
                    stride: Int,
                    b: DoubleArray,
                    bOff: Int,
                    len: Int,
                    out: DoubleArray,
                    outOff: Int,
                ) {
                    dots++
                    F64ScalarKernels.dot4(a, aOff, stride, b, bOff, len, out, outOff)
                }

                override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
                    axpys++
                    F64ScalarKernels.axpy(y, yOff, alpha, x, xOff, len)
                }
            }
            val k = 5
            val a = F64DenseMatrix(k, m, DoubleArray(k * m) { (it + 1).toDouble() })
            val b = F64DenseMatrix(n, k, DoubleArray(n * k) { (it + 1).toDouble() })
            F64ReferenceBlas(recording).gemm(1.0, a, true, b, true, 0.0, F64DenseMatrix(m, n))
            return dots to axpys
        }

        val packedB = kernelCalls(m = 9, n = 3)
        val packedA = kernelCalls(m = 3, n = 9)

        assertTrue(packedB.first > 0 && packedB.second == 0, "smaller B was not packed")
        assertTrue(packedA.first == 0 && packedA.second > 0, "smaller A was not packed")
    }

    private fun checkGemmShape(rng: Random, m: Int, k: Int, n: Int) {
        for (tA in booleanArrayOf(false, true)) {
            for (tB in booleanArrayOf(false, true)) {
                val a = if (tA) F64DenseMatrix(k, m) else F64DenseMatrix(m, k)
                val b = if (tB) F64DenseMatrix(n, k) else F64DenseMatrix(k, n)
                for (idx in a.data.indices) a.data[idx] = rng.nextDouble(-1.0, 1.0)
                for (idx in b.data.indices) b.data[idx] = rng.nextDouble(-1.0, 1.0)
                for (alpha in doubleArrayOf(0.0, 1.0, -2.0)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        // beta == 0 must overwrite without reading, so C starts poisoned with NaN.
                        val c0 = F64DenseMatrix(m, n)
                        for (idx in c0.data.indices) {
                            c0.data[idx] = if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                        }
                        val expected = F64DenseMatrix(m, n)
                        for (i in 0 until m) {
                            for (j in 0 until n) {
                                var s = 0.0
                                for (p in 0 until k) {
                                    s += (if (tA) a[p, i] else a[i, p]) * (if (tB) b[j, p] else b[p, j])
                                }
                                expected[i, j] = alpha * s + (if (beta == 0.0) 0.0 else beta * c0[i, j])
                            }
                        }
                        val c = F64DenseMatrix(m, n, c0.data.copyOf())
                        koblas.gemm(alpha, a, tA, b, tB, beta, c)
                        val bound = 100.0 * k * eps * (infNorm(a) * infNorm(b) + infNorm(expected))
                        for (idx in c.data.indices) {
                            assertTrue(
                                abs(c.data[idx] - expected.data[idx]) <= bound + 1e-12,
                                "gemm tA=$tA tB=$tB a=$alpha b=$beta at $idx: ${c.data[idx]} vs ${expected.data[idx]}",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `syrk matches gemm over the selected triangle`() {
        val rng = Random(20260731)
        for ((n, k) in listOf(1 to 1, 4 to 7, 9 to 3)) {
            for (transpose in booleanArrayOf(false, true)) {
                val a = if (transpose) F64DenseMatrix(k, n) else F64DenseMatrix(n, k)
                for (idx in a.data.indices) a.data[idx] = rng.nextDouble(-1.0, 1.0)
                for (alpha in doubleArrayOf(0.0, 1.0, -1.5)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        // C is asymmetric on purpose, since only syrk's alpha term is symmetric.
                        val c0 = F64DenseMatrix(n, n)
                        for (idx in c0.data.indices) {
                            c0.data[idx] = if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                        }
                        val expected = F64DenseMatrix(n, n, if (beta == 0.0) DoubleArray(n * n) else c0.data.copyOf())
                        koblas.gemm(alpha, a, transpose, a, !transpose, if (beta == 0.0) 0.0 else beta, expected)
                        val c = F64DenseMatrix(n, n, c0.data.copyOf())
                        koblas.syrk(alpha, a, transpose, beta, c)
                        val bound = 100.0 * k * eps * (infNorm(a) * infNorm(a) + infNorm(expected)) + 1e-12
                        for (j in 0 until n) {
                            for (i in j until n) {
                                assertTrue(
                                    abs(c[i, j] - expected[i, j]) <= bound,
                                    "syrk n=$n k=$k t=$transpose a=$alpha b=$beta at ($i,$j)",
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `sparse LU solve has a small residual on standard matrices`() {
        val rng = Random(20260101)
        for (n in intArrayOf(2, 8, 30)) {
            val dense = wellConditioned(n, rng)
            val cols = List(n) { j ->
                (0 until n).mapNotNull { i -> if (dense[i, j] != 0.0) i to dense[i, j] else null }
            }
            val sparse = F64SparseMatrix.ofColumns(n, n, cols)
            val lu = F64ReferenceSparseDecompositions(equilibrate = true).factor(sparse)
            if (lu.singular) continue
            val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
            val b = koblas.gemv(sparse, xTrue)
            val x = lu.solve(b)
            assertSolveResidual(dense, x, b, "sparseLU/n=$n")
        }
    }
}
