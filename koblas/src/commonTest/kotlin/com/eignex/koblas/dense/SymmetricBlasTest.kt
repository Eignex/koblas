package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseVector
import kotlin.random.Random
import kotlin.test.*

class SymmetricBlasTest {

    /**
     * The same check as the sibling above, at sizes that reach the blocked path.
     *
     * That one runs to order 12, which is inside a single tile: `REFERENCE_KC` is 128 and `REFERENCE_MC` is
     * 256, so it never packs a panel whose row block starts away from zero, and never crosses from one tile
     * to the next. The packing that feeds a blocked `symm` reads one side of the diagonal down a stored
     * column and the other along a stored row, and which is which flips with the triangle and with the side
     * `B` sits on, so the interesting cases are exactly the ones a small matrix cannot reach.
     *
     * Checked against `gemm` on the full matrix, which does not share the packing, rather than against
     * another `symm`.
     */
    @Test
    fun `symm matches gemm on the full matrix across block boundaries`() {
        val rng = Random(20260928)
        val reference = ReferenceBlas
        for (lower in booleanArrayOf(true, false)) {
            for (right in booleanArrayOf(false, true)) {
                for (n in intArrayOf(1, 2, 7, 129, 257, 300)) {
                    val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                    val b = if (right) randomMatrix(3, n, rng) else randomMatrix(n, 3, rng)
                    val expected = if (right) DenseMatrix(3, n) else DenseMatrix(n, 3)
                    val actual = if (right) DenseMatrix(3, n) else DenseMatrix(n, 3)

                    if (right) {
                        reference.gemm(0.75, b, false, full, false, 0.0, expected)
                    } else {
                        reference.gemm(0.75, full, false, b, false, 0.0, expected)
                    }
                    reference.symm(0.75, poisoned, b, 0.0, actual, lower, right)

                    val where = "lower=$lower right=$right n=$n"
                    assertClose(expected.data, actual.data, where, tolerance = 1e-9)
                }
            }
        }
    }

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

    @Test
    fun `symv four column traversal agrees with the scalar reference`() {
        val rng = Random(20260909)
        val n = DenseTuning.symvFourColumnCrossover + 1
        for (lower in booleanArrayOf(true, false)) {
            val (_, selected) = poisonedSymmetric(rng, n, lower)
            val x = DoubleArray(n) { if (it % 11 == 0) 0.0 else rng.nextDouble(-1.0, 1.0) }
            val initial = randomVector(n, rng)
            val expected = initial.copyOf()
            val actual = initial.copyOf()

            ReferenceBlas.symv(0.75, selected, x, -0.5, expected, lower)
            koblas.symv(0.75, selected, x, -0.5, actual, lower)

            assertClose(expected, actual, "four column symv lower=$lower", tolerance = 1e-10)
        }
    }

    @Test
    fun `symv four column traversal preserves selected exceptional arithmetic`() {
        val n = DenseTuning.symvFourColumnCrossover
        for (lower in booleanArrayOf(true, false)) {
            val a = DenseMatrix(n, n, DoubleArray(n * n) { Double.NaN })
            for (column in 0 until n) {
                val rows = if (lower) column until n else 0..column
                for (row in rows) a[row, column] = 0.0
            }
            val offDiagonalRow = if (lower) 300 else 7
            val offDiagonalColumn = if (lower) 7 else 300
            a[offDiagonalRow, offDiagonalColumn] = Double.POSITIVE_INFINITY
            a[31, 31] = Double.NEGATIVE_INFINITY
            val x = DoubleArray(n) { 1.0 }
            x[offDiagonalColumn] = 0.0
            x[31] = 0.0
            val expected = DoubleArray(n)
            val actual = DoubleArray(n)

            ReferenceBlas.symv(1.0, a, x, 0.0, expected, lower)
            koblas.symv(1.0, a, x, 0.0, actual, lower)

            for (i in 0 until n) {
                if (expected[i].isNaN()) {
                    assertTrue(actual[i].isNaN(), "lower=$lower index=$i was ${actual[i]}")
                } else {
                    assertEquals(expected[i], actual[i], "lower=$lower index=$i")
                }
            }
        }
    }

    @Test
    fun `reference symv evaluates its diagonal product for a zero multiplier`() {
        for (lower in booleanArrayOf(true, false)) {
            val a = DenseMatrix.diagonal(3)
            a[1, 1] = Double.POSITIVE_INFINITY
            val x = doubleArrayOf(2.0, 0.0, -3.0)
            val actual = DoubleArray(3)

            ReferenceBlas.symv(1.0, a, x, 0.0, actual, lower)

            assertEquals(2.0, actual[0], "lower=$lower first diagonal")
            assertTrue(actual[1].isNaN(), "lower=$lower middle diagonal was ${actual[1]}")
            assertEquals(-3.0, actual[2], "lower=$lower last diagonal")
        }
    }

    @Test
    fun `sparse syr follows dense BLAS arithmetic for implicit zeros`() {
        val values = doubleArrayOf(Double.POSITIVE_INFINITY, 0.0, 2.0)
        val sparse = SparseVector.of(3, intArrayOf(0, 2), doubleArrayOf(values[0], values[2]))
        for (lower in booleanArrayOf(true, false)) {
            val expected = DenseMatrix(3, 3)
            val actual = DenseMatrix(3, 3)

            ReferenceBlas.syr(1.0, DenseVector.wrap(values), expected, lower)
            ReferenceBlas.syr(1.0, sparse, actual, lower)

            assertContentEquals(expected.data, actual.data, "lower=$lower")
        }
    }

    @Test
    fun `sparse syr2 follows dense BLAS arithmetic for implicit zeros`() {
        val xValues = doubleArrayOf(Double.POSITIVE_INFINITY, 0.0, 0.0)
        val yValues = doubleArrayOf(0.0, 0.0, 2.0)
        val sparseX = SparseVector.of(3, intArrayOf(0), doubleArrayOf(xValues[0]))
        val sparseY = SparseVector.of(3, intArrayOf(2), doubleArrayOf(yValues[2]))
        for (lower in booleanArrayOf(true, false)) {
            val expected = DenseMatrix(3, 3)
            val actual = DenseMatrix(3, 3)

            ReferenceBlas.syr2(
                1.0,
                DenseVector.wrap(xValues),
                DenseVector.wrap(yValues),
                expected,
                lower,
            )
            ReferenceBlas.syr2(1.0, sparseX, sparseY, actual, lower)

            assertContentEquals(expected.data, actual.data, "lower=$lower")
        }
    }

    /**
     * `dsymv` derives its extent from one dimension, so a non-square matrix would have the host backend read
     * `n²` entries from a shorter array. The sizes straddle every backend's level-2 gate.
     */
    @Test
    fun `symv refuses a non-square matrix at every size`() {
        for (n in intArrayOf(2, 17, 64, 129)) {
            val wide = DenseMatrix.zero(n, n + 1)
            val tall = DenseMatrix.zero(n + 1, n)
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
                        val expected = DenseMatrix(n, p, c0.copyOf())
                        koblas.gemm(alpha, full, transposeA = false, b, transposeB = false, beta, expected)
                        val actual = DenseMatrix(n, p, c0.copyOf())
                        koblas.symm(alpha, poisoned, b, beta, actual, lower, workspace = Workspace())
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
                    val expected = DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.gemm(alpha, b, false, full, false, beta, expected)
                    val actual = DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.symm(alpha, poisoned, b, beta, actual, lower, right = true, workspace = Workspace())
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
        val term = DenseMatrix(n, n)
        koblas.syrk(1.0, a, transpose, 0.0, term, lower = lower)
        // The output is NaN where beta == 0 must overwrite without reading, and the unselected triangle stays NaN.
        val c = DenseMatrix(n, n)
        val c0 = DenseMatrix(n, n)
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
            val first = DenseMatrix(n, n)
            koblas.syrk(0.75, a, transpose = false, beta = 0.0, c = first, lower = lower, workspace = ws)
            val second = DenseMatrix(n, n)
            koblas.syrk(0.75, a, transpose = false, beta = 0.0, c = second, lower = lower, workspace = ws)
            val fresh = DenseMatrix(n, n)
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

    @Test
    fun `syrk preserves the netlib zero multiplier rule with infinities`() {
        val blas = BuiltinBlas(scalarDenseKernelFamilies)
        for (lower in booleanArrayOf(true, false)) {
            val values = if (lower) {
                doubleArrayOf(0.0, Double.POSITIVE_INFINITY)
            } else {
                doubleArrayOf(Double.POSITIVE_INFINITY, 0.0)
            }
            val row = if (lower) 1 else 0
            val column = if (lower) 0 else 1

            val outer = DenseMatrix(2, 2)
            blas.syrk(1.0, DenseMatrix(2, 1, values.copyOf()), false, 0.0, outer, lower)
            assertEquals(
                0.0,
                outer[row, column],
                "lower=$lower non-transposed syrk multiplied a skipped zero by infinity",
            )

            val dot = DenseMatrix(2, 2)
            blas.syrk(1.0, DenseMatrix(1, 2, values.copyOf()), true, 0.0, dot, lower)
            assertTrue(dot[row, column].isNaN(), "lower=$lower transposed syrk skipped zero times infinity")
        }
    }

    @Test
    fun `syrk preserves the netlib zero multiplier rule when finite scaling overflows`() {
        val blas = BuiltinBlas(scalarDenseKernelFamilies)
        for (lower in booleanArrayOf(true, false)) {
            val values = if (lower) {
                doubleArrayOf(0.0, Double.MAX_VALUE)
            } else {
                doubleArrayOf(Double.MAX_VALUE, 0.0)
            }
            val row = if (lower) 1 else 0
            val column = if (lower) 0 else 1
            val result = DenseMatrix(2, 2)

            blas.syrk(2.0, DenseMatrix(2, 1, values), false, 0.0, result, lower)

            assertEquals(
                0.0,
                result[row, column],
                "lower=$lower non-transposed syrk multiplied a skipped zero by a scaling overflow",
            )

            val transposed = DenseMatrix(2, 2)
            blas.syrk(2.0, DenseMatrix(1, 2, values.copyOf()), true, 0.0, transposed, lower)
            assertEquals(
                0.0,
                transposed[row, column],
                "lower=$lower transposed syrk introduced infinity before multiplying by zero",
            )
        }
    }

    @Test
    fun `syrk snapshots an aliased destination`() {
        val rng = Random(20260908)
        val blas = BuiltinBlas(scalarDenseKernelFamilies)
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                val original = DoubleArray(25) { rng.nextDouble(-1.0, 1.0) }
                val source = DenseMatrix(5, 5, original.copyOf())
                val expected = DenseMatrix(5, 5, original.copyOf())
                blas.syrk(0.75, source, transpose, -0.5, expected, lower)

                val shared = original.copyOf()
                blas.syrk(
                    0.75,
                    DenseMatrix.wrap(5, 5, shared),
                    transpose,
                    -0.5,
                    DenseMatrix.wrap(5, 5, shared),
                    lower,
                    Workspace(),
                )

                assertClose(expected.data, shared, "transpose=$transpose lower=$lower aliased destination")
            }
        }
    }

    /** The `dsyr2k` sum for one entry, over whichever orientation [transpose] selects. */
    private fun syr2kEntry(a: DenseMatrix, b: DenseMatrix, transpose: Boolean, k: Int, i: Int, j: Int): Double {
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
                val c = DenseMatrix.wrap(n, n, before.data.copyOf())
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
            val c = DenseMatrix(n, n)
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
    fun `syr2k preserves exceptional rank two evaluation`() {
        val blas = BuiltinBlas(scalarDenseKernelFamilies)
        for (lower in booleanArrayOf(true, false)) {
            val infinityA = if (lower) {
                doubleArrayOf(1.0, Double.POSITIVE_INFINITY)
            } else {
                doubleArrayOf(Double.POSITIVE_INFINITY, 1.0)
            }
            val overflowA = if (lower) {
                doubleArrayOf(0.0, Double.MAX_VALUE)
            } else {
                doubleArrayOf(Double.MAX_VALUE, 0.0)
            }
            val row = if (lower) 1 else 0
            val column = if (lower) 0 else 1
            for (transpose in booleanArrayOf(false, true)) {
                fun operand(values: DoubleArray): DenseMatrix = if (transpose) {
                    DenseMatrix(1, 2, values.copyOf())
                } else {
                    DenseMatrix(2, 1, values.copyOf())
                }

                val infinity = DenseMatrix(2, 2)
                blas.syr2k(
                    1.0,
                    operand(infinityA),
                    operand(doubleArrayOf(0.0, 0.0)),
                    transpose,
                    0.0,
                    infinity,
                    lower,
                )
                assertTrue(
                    infinity[row, column].isNaN(),
                    "lower=$lower transpose=$transpose independently skipped a zero coefficient",
                )

                val overflow = DenseMatrix(2, 2)
                blas.syr2k(
                    2.0,
                    operand(overflowA),
                    operand(doubleArrayOf(0.0, 0.0)),
                    transpose,
                    0.0,
                    overflow,
                    lower,
                )
                assertEquals(
                    0.0,
                    overflow[row, column],
                    "lower=$lower transpose=$transpose moved alpha before a zero product",
                )
            }
        }
    }

    @Test
    fun `syr2k interleaves cross products before accumulation overflow`() {
        val blas = BuiltinBlas(scalarDenseKernelFamilies)
        val magnitude = 1e154
        val normalA = doubleArrayOf(magnitude, magnitude, magnitude, magnitude)
        val normalB = doubleArrayOf(magnitude, -magnitude, magnitude, -magnitude)
        val transposedB = doubleArrayOf(magnitude, magnitude, -magnitude, -magnitude)
        for (transpose in booleanArrayOf(false, true)) {
            val a = DenseMatrix(2, 2, normalA.copyOf())
            val b = DenseMatrix(2, 2, if (transpose) transposedB.copyOf() else normalB.copyOf())
            for (lower in booleanArrayOf(true, false)) {
                val c = DenseMatrix(2, 2)
                blas.syr2k(1.0, a, b, transpose, 0.0, c, lower)
                val row = if (lower) 1 else 0
                val column = if (lower) 0 else 1
                assertEquals(
                    0.0,
                    c[row, column],
                    "lower=$lower transpose=$transpose separated cancelling cross-products",
                )
            }
        }
    }

    @Test
    fun `syr2k snapshots either aliased input`() {
        val rng = Random(20260909)
        val blas = BuiltinBlas(scalarDenseKernelFamilies)
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                for (aliased in 0..2) {
                    val original = DoubleArray(25) { rng.nextDouble(-1.0, 1.0) }
                    val other = DoubleArray(25) { rng.nextDouble(-1.0, 1.0) }
                    val expected = DenseMatrix(5, 5, original.copyOf())
                    val expectedA = DenseMatrix(5, 5, if (aliased == 1) other.copyOf() else original.copyOf())
                    val expectedB = DenseMatrix(5, 5, if (aliased == 0) other.copyOf() else original.copyOf())
                    blas.syr2k(0.75, expectedA, expectedB, transpose, -0.5, expected, lower)

                    val shared = original.copyOf()
                    val sharedMatrix = DenseMatrix.wrap(5, 5, shared)
                    val actualA = if (aliased == 1) DenseMatrix(5, 5, other.copyOf()) else sharedMatrix
                    val actualB = if (aliased == 0) DenseMatrix(5, 5, other.copyOf()) else sharedMatrix
                    blas.syr2k(0.75, actualA, actualB, transpose, -0.5, sharedMatrix, lower, Workspace())

                    assertClose(
                        expected.data,
                        shared,
                        "transpose=$transpose lower=$lower aliased=$aliased",
                    )
                }
            }
        }
    }

    @Test
    fun `syrk crosses its cache tile boundaries`() {
        val rng = Random(20261031)
        for ((n, k) in listOf((REFERENCE_MC + 1) to 3, (REFERENCE_NC + 3) to (REFERENCE_KC + 2))) {
            for (transpose in booleanArrayOf(false, true)) {
                val a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
                for (lower in booleanArrayOf(false, true)) {
                    val rankOne = DenseMatrix(n, n)
                    ReferenceBlas.syrk(0.75, a, transpose, 0.0, rankOne, lower)
                    for (j in 0 until n) {
                        val range = if (lower) j until n else 0..j
                        for (i in range) {
                            var aa = 0.0
                            for (p in 0 until k) {
                                val ai = if (transpose) a[p, i] else a[i, p]
                                val aj = if (transpose) a[p, j] else a[j, p]
                                aa += ai * aj
                            }
                            assertClose(0.75 * aa, rankOne[i, j], "syrk n=$n t=$transpose l=$lower")
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `syr2k crosses packed tile boundaries`() = assertSyr2kShapes(listOf(3 to 1, 4 to 3, 5 to 7))

    @Test
    fun `syr2k crosses packed row block boundaries`() = assertSyr2kShapes(
        listOf(
            (DenseTuning.packedBlockRows - 1) to 3,
            DenseTuning.packedBlockRows to 5,
            (DenseTuning.packedBlockRows + 1) to 7,
        ),
    )

    @Test
    fun `syr2k crosses packed column block boundaries`() = assertSyr2kShapes(
        listOf(
            (DenseTuning.packedBlockColumns - 1) to 1,
            DenseTuning.packedBlockColumns to 1,
            (DenseTuning.packedBlockColumns + 1) to 1,
        ),
    )

    @Test
    fun `syr2k crosses packed depth block boundaries`() = assertSyr2kShapes(
        listOf(
            17 to (DenseTuning.packedBlockDepth - 1),
            17 to DenseTuning.packedBlockDepth,
            17 to (DenseTuning.packedBlockDepth + 1),
        ),
    )

    private fun assertSyr2kShapes(shapes: List<Pair<Int, Int>>) {
        val rng = Random(20261031)
        for ((n, k) in shapes) {
            for (transpose in booleanArrayOf(false, true)) {
                val a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
                val b = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
                val normalizedA = if (transpose) {
                    DoubleArray(n * k) { index -> a[index / n, index % n] }
                } else {
                    a.data
                }
                val normalizedB = if (transpose) {
                    DoubleArray(n * k) { index -> b[index / n, index % n] }
                } else {
                    b.data
                }
                for (lower in booleanArrayOf(false, true)) {
                    val expected = DenseMatrix(n, n)
                    blockedSyr2kUpdate(
                        platformDenseKernelFamilies.panel,
                        0.75,
                        normalizedA,
                        normalizedB,
                        expected.data,
                        n,
                        k,
                        lower,
                        guardZeroColumns = !transpose,
                    )
                    val actual = DenseMatrix(n, n)
                    ReferenceBlas.syr2k(0.75, a, b, transpose, 0.0, actual, lower)
                    assertClose(expected.data, actual.data, "syr2k n=$n t=$transpose l=$lower", tolerance = 1e-9)
                }
            }
        }
    }

    @Test
    fun `left symm crosses its cache tile boundaries`() {
        val rng = Random(20261031)
        val n = REFERENCE_MC + 1
        val columns = REFERENCE_NC + 1
        for (lower in booleanArrayOf(false, true)) {
            val (full, selected) = poisonedSymmetric(rng, n, lower)
            val b = randomMatrix(n, columns, rng)
            val expected = DenseMatrix(n, columns)
            ReferenceBlas.gemm(0.75, full, false, b, false, 0.0, expected)
            val actual = DenseMatrix(n, columns)
            ReferenceBlas.symm(0.75, selected, b, 0.0, actual, lower)
            assertClose(expected, actual, "symm n=$n lower=$lower", tolerance = 1e-10)
        }
    }

    @Test
    fun `right symm crosses its cache tile boundaries`() {
        val rng = Random(20261031)
        val rows = REFERENCE_MC + 1
        val n = REFERENCE_NC + 1
        for (lower in booleanArrayOf(false, true)) {
            val (full, selected) = poisonedSymmetric(rng, n, lower)
            val b = randomMatrix(rows, n, rng)
            val expected = DenseMatrix(rows, n)
            ReferenceBlas.gemm(0.75, b, false, full, false, 0.0, expected)
            val actual = DenseMatrix(rows, n)
            ReferenceBlas.symm(0.75, selected, b, 0.0, actual, lower, right = true)
            assertClose(expected, actual, "right symm n=$n lower=$lower", tolerance = 1e-10)
        }
    }

    @Test
    fun `symm handles empty shapes`() {
        koblas.symm(1.0, DenseMatrix(0, 0), DenseMatrix(0, 3), 0.0, DenseMatrix(0, 3))
        val c = DenseMatrix(2, 0)
        koblas.symm(1.0, DenseMatrix(2, 2), DenseMatrix(2, 0), 0.0, c)
    }

    @Test
    fun `symv with zero alpha overwrites the destination`() {
        val y = DoubleArray(2) { Double.NaN }
        koblas.symv(0.0, DenseMatrix(2, 2), DoubleArray(2), 0.0, y)
        assertTrue(y.all { it == 0.0 }, "symv alpha=0 beta=0 left ${y.toList()}")
    }

    @Test
    fun `symm leaves C alone when the operands do not line up`() {
        // Scaling C is part of the operation, so a call that cannot go through must not have done it.
        for (right in booleanArrayOf(true, false)) {
            for (beta in doubleArrayOf(0.0, 2.0)) {
                // C matches B, so the shape that fails is A against B: too few columns from the right,
                // too few rows from the left.
                val a = DenseMatrix.diagonal(if (right) 2 else 3)
                val b = DenseMatrix(2, 3)
                val c = DenseMatrix(2, 3)
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

    /** Panel kernels whose arithmetic AXPY fails, standing in for an incomplete blocked update. */
    private class FailingAxpy : DensePanelKernels by ScalarPanelKernels {
        override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
            error("kernel failed")
    }

    /** Packed kernels whose tile fails after all of its scratch buffers have been borrowed. */
    private class FailingTile : PackedKernels by PortablePackedKernels {
        override fun gemmTile(
            depth: Int,
            packedA: DoubleArray,
            aOff: Int,
            packedB: DoubleArray,
            bOff: Int,
            c: DoubleArray,
            cOff: Int,
            ldc: Int,
        ) = error("kernel failed")
    }

    @Test
    fun `syrk gives its workspace buffer back when the inner loop throws`() {
        val n = 4
        val k = 3
        val ws = Workspace()
        ws.reserve(n * k, 2)
        ws.reserve(PORTABLE_TILE * PORTABLE_TILE, 1)
        val blas = BuiltinBlas(testDenseKernelFamilies(packed = FailingTile()))
        assertFailsWith<IllegalStateException> {
            blas.syrk(
                1.0,
                DenseMatrix(k, n, DoubleArray(k * n).also { it.fill(1.0) }),
                transpose = true,
                beta = 0.0,
                c = DenseMatrix(n, n),
                workspace = ws,
            )
        }
        assertEquals(2, ws.available(n * k))
        assertEquals(1, ws.available(PORTABLE_TILE * PORTABLE_TILE))
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
            val pooled = DenseMatrix(n, n)
            koblas.syr2k(0.75, a, b, transpose, beta = 0.0, c = pooled, lower = true, workspace = ws)
            val fresh = DenseMatrix(n, n)
            koblas.syr2k(0.75, a, b, transpose, beta = 0.0, c = fresh, lower = true)
            assertClose(fresh.data, pooled.data, "transpose=$transpose against the unpooled result")
            val again = DenseMatrix(n, n)
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
        val blas = BuiltinBlas(testDenseKernelFamilies(packed = FailingTile()))
        assertFailsWith<IllegalStateException> {
            blas.syr2k(
                1.0,
                DenseMatrix(k, n, DoubleArray(k * n).also { it.fill(1.0) }),
                DenseMatrix(k, n, DoubleArray(k * n).also { it.fill(1.0) }),
                transpose = true,
                beta = 0.0,
                c = DenseMatrix(n, n),
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

    @Test
    fun `syr2k exceptional fallback returns its workspace after failure`() {
        val n = 4
        val k = 3
        val ws = Workspace()
        ws.reserve(n * k, 2)
        val a = DoubleArray(k * n).also {
            it.fill(1.0)
            it[0] = Double.POSITIVE_INFINITY
        }
        val blas = BuiltinBlas(testDenseKernelFamilies(panel = FailingAxpy()))

        assertFailsWith<IllegalStateException> {
            blas.syr2k(
                1.0,
                DenseMatrix(k, n, a),
                DenseMatrix(k, n, DoubleArray(k * n).also { it.fill(1.0) }),
                transpose = true,
                beta = 0.0,
                c = DenseMatrix(n, n),
                workspace = ws,
            )
        }

        assertEquals(2, ws.available(n * k))
    }
}
