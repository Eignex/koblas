package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.random.Random
import kotlin.test.*

class LinearAlgebraSymmetricOpsTest {

    @Test
    fun `symv matches gemv on the full matrix and reads only the selected triangle`() {
        val rng = Random(20260910)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 2, 7, 16)) {
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val x = randomVector(n, rng)
                        val y0 = DoubleArray(n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = y0.copyOf()
                        koblas.gemv(alpha, full, x, beta, expected)
                        val actual = y0.copyOf()
                        koblas.symv(alpha, poisoned, x, beta, actual, lower)
                        assertClose(expected, actual, "symv n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    @Test
    fun `symv matches gemv when x holds zeros`() {
        val rng = Random(20260911)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(4, 7, 9, 16)) {
                val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                // A zero inside a block sends it down the column by column path.
                val x = DoubleArray(n) { if (it % 3 == 0) 0.0 else rng.nextDouble(-1.0, 1.0) }
                val expected = DoubleArray(n)
                koblas.gemv(1.0, full, x, 0.0, expected)
                val actual = DoubleArray(n)
                koblas.symv(1.0, poisoned, x, 0.0, actual, lower)
                assertClose(expected, actual, "symv zeros n=$n lower=$lower")
            }
        }
    }

    /**
     * `dsymv` derives its extent from one dimension, so a non-square matrix would have the host backend read
     * `n²` entries from a shorter array. The sizes straddle every backend's level-2 gate.
     */
    @Test
    fun `symv refuses a non-square matrix at every size`() {
        for (n in intArrayOf(2, 17, 64, 129)) {
            val wide = F64DenseMatrix.zero(n, n + 1)
            val tall = F64DenseMatrix.zero(n + 1, n)
            assertFailsWith<DimensionMismatch>("symv ${n}x${n + 1}") {
                koblas.symv(1.0, wide, DoubleArray(n), 0.0, DoubleArray(n))
            }
            assertFailsWith<DimensionMismatch>("symv ${n + 1}x$n") {
                koblas.symv(1.0, tall, DoubleArray(n + 1), 0.0, DoubleArray(n + 1))
            }
        }
    }

    @Test
    fun `symm matches gemm on the full matrix and reads only the selected triangle`() {
        val rng = Random(20260911)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 5, 12)) {
                val p = 4
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val b = randomMatrix(n, p, rng)
                        val c0 = DoubleArray(n * p) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = F64DenseMatrix(n, p, c0.copyOf())
                        koblas.gemm(alpha, full, transposeA = false, b, transposeB = false, beta, expected)
                        val actual = F64DenseMatrix(n, p, c0.copyOf())
                        koblas.symm(alpha, poisoned, b, beta, actual, lower)
                        assertClose(expected, actual, "symm n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    @Test
    fun `symm right multiplies from the right and reads only the selected triangle`() {
        val rng = Random(20260942)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (alpha in doubleArrayOf(0.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                    val b = randomMatrix(rows, n, rng)
                    val c0 = DoubleArray(rows * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = F64DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.gemm(alpha, b, false, full, false, beta, expected)
                    val actual = F64DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.symm(alpha, poisoned, b, beta, actual, lower, right = true)
                    assertClose(expected, actual, "symm right l=$lower a=$alpha b=$beta", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `syrk triangle modes write only the selected triangle with strict beta semantics`() {
        val rng = Random(20260933)
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (alpha in doubleArrayOf(0.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        checkSyrk(rng, lower, transpose, alpha, beta)
                    }
                }
            }
        }
    }

    @Suppress("CyclomaticComplexMethod")
    private fun checkSyrk(rng: Random, lower: Boolean, transpose: Boolean, alpha: Double, beta: Double) {
        val a = randomMatrix(6, 4, rng)
        val n = if (transpose) a.cols else a.rows
        val term = F64DenseMatrix(n, n)
        koblas.syrk(1.0, a, transpose, 0.0, term, lower = lower)
        // The output is NaN where beta == 0 must overwrite without reading, and the unselected triangle stays NaN.
        val c = F64DenseMatrix(n, n)
        val c0 = F64DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (lower) j <= i else j >= i
                val v = if (!selected || beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                c[i, j] = v
                c0[i, j] = v
            }
        }
        koblas.syrk(alpha, a, transpose, beta, c, lower)
        val context = "syrk lower=$lower t=$transpose a=$alpha b=$beta"
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (lower) j <= i else j >= i
                if (selected) {
                    val expected = (if (beta == 0.0) 0.0 else beta * c0[i, j]) + alpha * term[i, j]
                    assertClose(expected, c[i, j], "$context ($i;$j)")
                } else {
                    assertTrue(c[i, j].isNaN(), "$context ($i;$j): untouched triangle was written")
                }
            }
        }
    }

    @Test
    fun `syrk through a reused workspace does not accumulate the previous call`() {
        val rng = Random(20260946)
        val n = 8
        val k = 5
        val ws = Workspace()
        for (lower in booleanArrayOf(true, false)) {
            val a = randomMatrix(n, k, rng)
            val first = F64DenseMatrix(n, n)
            koblas.syrk(0.75, a, transpose = false, beta = 0.0, c = first, lower = lower, workspace = ws)
            val second = F64DenseMatrix(n, n)
            koblas.syrk(0.75, a, transpose = false, beta = 0.0, c = second, lower = lower, workspace = ws)
            val fresh = F64DenseMatrix(n, n)
            koblas.syrk(0.75, a, transpose = false, beta = 0.0, c = fresh, lower = lower)
            for (i in 0 until n) {
                for (j in 0 until n) {
                    val f = first[i, j]
                    if (f.isNaN()) continue // an untouched triangle stays untouched; checked elsewhere
                    assertClose(f, second[i, j], "syrk lower=$lower reused workspace ($i;$j)")
                    assertClose(f, fresh[i, j], "syrk lower=$lower against no workspace ($i;$j)")
                }
            }
        }
    }

    /** The `dsyr2k` sum for one entry, over whichever orientation [transpose] selects. */
    private fun syr2kEntry(a: F64DenseMatrix, b: F64DenseMatrix, transpose: Boolean, k: Int, i: Int, j: Int): Double {
        var s = 0.0
        for (p in 0 until k) {
            val ai = if (transpose) a[p, i] else a[i, p]
            val aj = if (transpose) a[p, j] else a[j, p]
            val bi = if (transpose) b[p, i] else b[i, p]
            val bj = if (transpose) b[p, j] else b[j, p]
            s += ai * bj + bi * aj
        }
        return s
    }

    /**
     * A triangle mode has to beta-scale and write the same half, and leave the other half of an asymmetric
     * destination exactly as it found it. Beta is non-zero and C starts asymmetric so that scaling the wrong
     * region, or mirroring where the routine should accumulate, both show up.
     */
    @Test
    fun `syr2k writes only the triangle it is given and scales only that`() {
        val rng = Random(20260825)
        val n = 5
        val k = 3
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(true, false)) {
                val a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
                val b = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
                val before = randomMatrix(n, n, rng)
                val c = F64DenseMatrix.wrap(n, n, before.data.copyOf())
                koblas.syr2k(0.5, a, b, transpose, beta = 2.0, c = c, lower = lower)
                for (i in 0 until n) {
                    for (j in 0 until n) {
                        val written = if (lower) i >= j else i <= j
                        val expected = if (written) {
                            2.0 * before[i, j] + 0.5 * syr2kEntry(a, b, transpose, k, i, j)
                        } else {
                            before[i, j]
                        }
                        assertClose(expected, c[i, j], "transpose=$transpose lower=$lower at [$i,$j]")
                    }
                }
            }
        }
    }

    @Test
    fun `syr2k matches the gemm expansion in both orientations`() {
        val rng = Random(20260809)
        for (transpose in booleanArrayOf(false, true)) {
            val n = 4
            val k = 3
            val a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
            val b = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
            val c = F64DenseMatrix(n, n)
            koblas.syr2k(0.5, a, b, transpose, 0.0, c)
            for (j in 0 until n) {
                for (i in j until n) {
                    assertClose(
                        0.5 * syr2kEntry(a, b, transpose, k, i, j),
                        c[i, j],
                        "transpose=$transpose at [$i,$j]",
                    )
                }
            }
        }
    }

    @Test
    fun `symmetric ops handle empty shapes`() {
        koblas.symm(1.0, F64DenseMatrix(0, 0), F64DenseMatrix(0, 3), 0.0, F64DenseMatrix(0, 3))
        val c = F64DenseMatrix(2, 0)
        koblas.symm(1.0, F64DenseMatrix(2, 2), F64DenseMatrix(2, 0), 0.0, c)
        val y = DoubleArray(2) { Double.NaN }
        koblas.symv(0.0, F64DenseMatrix(2, 2), DoubleArray(2), 0.0, y)
        assertTrue(y.all { it == 0.0 }, "symv alpha=0 beta=0 left ${y.toList()}")
    }

    @Test
    fun `symm leaves C alone when the operands do not line up`() {
        // Scaling C is part of the operation, so a call that cannot go through must not have done it.
        for (right in booleanArrayOf(true, false)) {
            for (beta in doubleArrayOf(0.0, 2.0)) {
                // C matches B, so the shape that fails is A against B: too few columns from the right,
                // too few rows from the left.
                val a = F64DenseMatrix.diagonal(if (right) 2 else 3)
                val b = F64DenseMatrix(2, 3)
                val c = F64DenseMatrix(2, 3)
                c.data.fill(7.0)
                assertFailsWith<DimensionMismatch>("symm right=$right beta=$beta should reject these shapes") {
                    koblas.symm(1.0, a, b, beta, c, lower = true, right = right)
                }
                assertTrue(
                    c.data.all { it == 7.0 },
                    "symm right=$right beta=$beta scaled C before rejecting the call: ${c.data.toList()}",
                )
            }
        }
    }

    /** Kernels whose `dot` fails, standing in for a host backend that cannot complete a call. */
    private class FailingDot : F64Kernels {
        override val name: String get() = "failing-dot"
        override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
            error("kernel failed")
        override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
            for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
        }
        override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
            for (i in 0 until len) v[vOff + i] *= alpha
        }
        override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)
        override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)
    }

    @Test
    fun `syrk gives its workspace buffer back when the inner loop throws`() {
        val n = 4
        val k = 3
        val ws = Workspace()
        // Park one buffer of the staging width in the pool, so syrk borrows this exact instance.
        val parked = ws.take(n * k)
        ws.release(parked)
        val blas = F64ReferenceBlas(FailingDot())
        assertFailsWith<IllegalStateException> {
            blas.syrk(
                1.0,
                F64DenseMatrix(n, k),
                transpose = false,
                beta = 0.0,
                c = F64DenseMatrix(n, n),
                workspace = ws,
            )
        }
        assertSame(parked, ws.take(n * k), "syrk kept its workspace buffer after the inner loop threw")
    }

    @Test
    fun `syr2k through a reused workspace does not accumulate the previous call`() {
        val rng = Random(20260827)
        val n = 7
        val k = 4
        val ws = Workspace()
        for (transpose in booleanArrayOf(false, true)) {
            val a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
            val b = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
            // Two staging buffers of one width, so a routine that borrowed the same one twice would hold
            // one operand where the other belongs and both dots would read it.
            val pooled = F64DenseMatrix(n, n)
            koblas.syr2k(0.75, a, b, transpose, beta = 0.0, c = pooled, lower = true, workspace = ws)
            val fresh = F64DenseMatrix(n, n)
            koblas.syr2k(0.75, a, b, transpose, beta = 0.0, c = fresh, lower = true)
            assertClose(fresh.data, pooled.data, "transpose=$transpose against the unpooled result")
            val again = F64DenseMatrix(n, n)
            koblas.syr2k(0.75, a, b, transpose, beta = 0.0, c = again, lower = true, workspace = ws)
            assertClose(fresh.data, again.data, "transpose=$transpose on the second pooled call")
        }
    }

    @Test
    fun `syr2k gives its workspace buffers back when the inner loop throws`() {
        val n = 4
        val k = 3
        val ws = Workspace()
        val parked = listOf(ws.take(n * k), ws.take(n * k))
        parked.forEach { ws.release(it) }
        val blas = F64ReferenceBlas(FailingDot())
        assertFailsWith<IllegalStateException> {
            blas.syr2k(
                1.0,
                F64DenseMatrix(n, k),
                F64DenseMatrix(n, k),
                transpose = false,
                beta = 0.0,
                c = F64DenseMatrix(n, n),
                workspace = ws,
            )
        }
        for (buffer in listOf(ws.take(n * k), ws.take(n * k))) {
            assertTrue(
                buffer === parked[0] || buffer === parked[1],
                "syr2k kept a workspace buffer after the inner loop threw",
            )
        }
    }
}
