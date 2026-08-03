@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas

import kotlin.concurrent.Volatile
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Both halves of the compute seam at once: the [Blas] routines and the [Lapack] factorizations built on
 * them. [koblas] is one of these, composed from whichever backend won each half.
 *
 * Implement this when a backend provides both, which is the usual case for a host library. Implement
 * [Blas] or [Lapack] alone when it does not — the two are ranked and installed independently, so a host
 * with CBLAS but no LAPACKE still accelerates its level-2 and level-3 work.
 */
interface LinearAlgebra :
    Blas,
    Lapack

/** Bunch–Kaufman pivot threshold `(1 + √17) / 8`, the value minimizing element growth (netlib dsytf2). */
private const val BUNCH_KAUFMAN_ALPHA = 0.6403882032022076

/**
 * A general LU factorization with partial pivoting: `P·A = L·U`, the unit-lower `L` and upper `U` packed
 * into one flat column-major [lu] buffer (`L` below the diagonal, `U` on and above) and the row permutation
 * in [piv] (`piv[k]` is the original row now at position `k`). [singular] is set when a zero pivot was
 * hit; [LinearAlgebra.solve] on a singular factorization is not meaningful. Produced by
 * [LinearAlgebra.factor].
 *
 * The constructor and the factor buffers are public so that out-of-repo [LinearAlgebra] backends can
 * produce and consume factorizations in the shared format; every backend packs identically, so a
 * decomposition from one backend solves correctly on another. [lu] and [piv] are live buffers, not
 * copies: treat them as read-only, and note that [LinearAlgebra.factorInto] rewrites them in place so a
 * periodic refactorization need not allocate new ones.
 *
 * @property n the matrix dimension.
 * @property lu the packed `L`\`U` factors, column-major, length `n * n`.
 * @property piv the row permutation.
 * @param singular whether a zero pivot was encountered; readable afterwards through [singular].
 */
class LuDecomposition(val n: Int, val lu: DoubleArray, val piv: IntArray, singular: Boolean) {
    /** Whether a zero pivot was encountered. Refactorizing in place via [LinearAlgebra.factorInto]
     *  updates it along with the buffers. */
    var singular: Boolean = singular
        internal set

    init {
        require(lu.size == n * n) { "lu length ${lu.size} != ${n * n}" }
        require(piv.size == n) { "piv length ${piv.size} != $n" }
    }
}

/** `det(A)` from the factorization: `sign(P) · ∏ U[k][k]`, or exactly `0.0` when [LuDecomposition.singular].
 *  The floating-point counterpart of [SparseLu.determinant]. */
fun LuDecomposition.determinant(): Double {
    if (singular) return 0.0
    var d = permutationSign(piv)
    for (k in 0 until n) d *= lu[k * n + k]
    return d
}

/** Sign of the permutation [p] (`p[k]` = original index at position `k`): `(-1)^(size − cycles)`. */
internal fun permutationSign(p: IntArray): Double {
    val seen = BooleanArray(p.size)
    var cycles = 0
    for (s in p.indices) {
        if (seen[s]) continue
        cycles++
        var i = s
        while (!seen[i]) {
            seen[i] = true
            i = p[i]
        }
    }
    return if ((p.size - cycles) % 2 == 0) 1.0 else -1.0
}

/**
 * Portable pure-Kotlin backend — correct on every target, no native dependency, and the semantic
 * reference a native backend is validated against. Textbook Doolittle LU with partial pivoting and naive
 * (SIMD-assisted where the level-1 primitives kick in) level-2/3 loops.
 */
object ReferenceLinearAlgebra : LinearAlgebra {
    override val name: String get() = "reference"

    override fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        require(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        require(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        if (beta == 0.0) {
            y.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(y, 0, beta, y.size)
        }
        if (alpha == 0.0) return
        val ad = a.data
        val rows = a.rows
        if (!transpose) {
            // y += alpha·Σ_j x_j·A[:,j]: one axpy per column, every run contiguous and no reduction.
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                if (xj != 0.0) denseAxpy(y, 0, xj, ad, j * rows, rows)
            }
        } else {
            // y_j = alpha·A[:,j]ᵀ·x, the dot of a contiguous column with x. Four columns at a time: one
            // shared load of x per segment, four independent accumulators.
            val quads = DoubleArray(4)
            var j = 0
            val bound = a.cols - 3
            while (j < bound) {
                denseDot4(ad, j * rows, rows, x, 0, rows, quads, 0)
                y[j] += alpha * quads[0]
                y[j + 1] += alpha * quads[1]
                y[j + 2] += alpha * quads[2]
                y[j + 3] += alpha * quads[3]
                j += 4
            }
            while (j < a.cols) {
                y[j] += alpha * denseDot(ad, j * rows, x, 0, rows)
                j++
            }
        }
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod")
    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val kB = if (transposeB) b.cols else b.rows
        val n = if (transposeB) b.rows else b.cols
        require(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        require(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        val cd = c.data
        if (beta == 0.0) {
            cd.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(cd, 0, beta, cd.size)
        }
        if (alpha == 0.0 || m == 0 || n == 0 || k == 0) return
        if (transposeA && transposeB) {
            // C = alpha·Aᵀ·Bᵀ + C: the only combination that has a contiguous run on neither side of the
            // inner product — A's columns line up, but Bᵀ needs B's rows. Materialize Bᵀ once, then reuse
            // the (T, noT) column-dot kernel.
            val bt = DenseMatrix(k, n)
            for (j in 0 until n) for (p in 0 until k) bt[p, j] = b[j, p]
            gemm(alpha, a, transposeA = true, bt, transposeB = false, beta = 1.0, c = c)
            return
        }
        val ad = a.data
        val bd = b.data
        when {
            // C += alpha·Aᵀ·B: entry (i, j) is the dot of column i of A with column j of B, both
            // contiguous. Four A columns share each load of the B column, the shape gemv exploits.
            transposeA -> {
                val quads = DoubleArray(4)
                for (j in 0 until n) {
                    var i = 0
                    val bound = m - 3
                    while (i < bound) {
                        denseDot4(ad, i * k, k, bd, j * k, k, quads, 0)
                        cd[i + j * m] += alpha * quads[0]
                        cd[i + 1 + j * m] += alpha * quads[1]
                        cd[i + 2 + j * m] += alpha * quads[2]
                        cd[i + 3 + j * m] += alpha * quads[3]
                        i += 4
                    }
                    while (i < m) {
                        cd[i + j * m] += alpha * denseDot(ad, i * k, bd, j * k, k)
                        i++
                    }
                }
            }

            // C += alpha·A·B: C[:,j] = Σ_p B[p,j]·A[:,p], rank-1 axpy sweeps down contiguous columns.
            !transposeB -> for (j in 0 until n) {
                for (p in 0 until k) {
                    val bpj = alpha * bd[p + j * k]
                    if (bpj != 0.0) denseAxpy(cd, j * m, bpj, ad, p * m, m)
                }
            }

            // C += alpha·A·Bᵀ: same column sweeps, but the B scalar comes from row j of the n×k B.
            else -> for (j in 0 until n) {
                for (p in 0 until k) {
                    val bjp = alpha * bd[j + p * n]
                    if (bjp != 0.0) denseAxpy(cd, j * m, bjp, ad, p * m, m)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        require(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        val cd = c.data
        scaleUplo(cd, n, beta, uplo)
        if (alpha == 0.0 || n == 0 || k == 0) return
        val ad = a.data
        if (transpose) {
            // C += alpha·Aᵀ·A: entry (i, j) is the dot of contiguous columns i and j of the k×n A;
            // computed once per pair, then written to the selected triangle(s).
            for (j in 0 until n) {
                for (i in j until n) {
                    val v = alpha * denseDot(ad, i * k, ad, j * k, k)
                    addUplo(cd, n, i, j, v, uplo)
                }
            }
        } else {
            // C += alpha·A·Aᵀ = alpha·Σₚ A[:,p]·A[:,p]ᵀ: rank-1 sweeps over contiguous columns of the
            // n×k A into a scratch buffer, then written per triangle — the sweeps alone are not
            // bit-symmetric ((alpha·x)·y and (alpha·y)·x round differently), and the contract promises an
            // exactly symmetric alpha term.
            // The n² mirror is the largest scratch in the library, so the one most worth lending; pools
            // key by width, so this just borrows an n²-wide vector.
            // Cleared first: take() promises nothing about the contents, and the sweeps below accumulate.
            val w = workspace?.take(n * n) ?: DoubleArray(n * n)
            w.fill(0.0, 0, n * n)
            for (p in 0 until k) {
                val base = p * n
                for (j in 0 until n) {
                    val f = alpha * ad[base + j]
                    if (f != 0.0) denseAxpy(w, j * n, f, ad, base, n)
                }
            }
            for (j in 0 until n) {
                for (i in j until n) {
                    addUplo(cd, n, i, j, w[i + j * n], uplo)
                }
            }
            workspace?.release(w)
        }
    }

    /** Add the symmetric term `v` for the pair `(i, j)`, `j <= i`, into the triangle(s) [uplo] selects. */
    @Suppress("LongParameterList")
    private fun addUplo(cd: DoubleArray, n: Int, i: Int, j: Int, v: Double, uplo: Uplo) {
        if (uplo != Uplo.UPPER) {
            cd[i + j * n] += v
        } else if (i == j) {
            cd[i + i * n] += v // the diagonal belongs to both triangles
        }
        if (uplo != Uplo.LOWER && i != j) cd[j + i * n] += v
    }

    /** `beta` scale of the region [uplo] selects, honoring the `beta == 0` overwrite convention. */
    private fun scaleUplo(cd: DoubleArray, n: Int, beta: Double, uplo: Uplo) {
        if (uplo == Uplo.FULL) {
            if (beta == 0.0) {
                cd.fill(0.0)
            } else if (beta != 1.0) {
                denseScale(cd, 0, beta, cd.size)
            }
            return
        }
        if (beta == 1.0) return
        // Column j holds its lower triangle at rows j..n-1 and its upper triangle at rows 0..j.
        for (j in 0 until n) {
            val from = if (uplo == Uplo.LOWER) j + j * n else j * n
            val len = if (uplo == Uplo.LOWER) n - j else j + 1
            if (beta == 0.0) cd.fill(0.0, from, from + len) else denseScale(cd, from, beta, len)
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymv signature
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        require(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        require(x.size == n) { "symv: x length ${x.size} != $n" }
        require(y.size == n) { "symv: y length ${y.size} != $n" }
        if (beta == 0.0) {
            y.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(y, 0, beta, n)
        }
        if (alpha == 0.0) return
        symvAccumulate(alpha, a.data, n, x, 0, y, 0, lower)
    }

    /**
     * `y[yOff..] += alpha · A · x[xOff..]` for the symmetric `n×n` [ad], reading only the [lower] (or
     * upper) triangle.
     *
     * Column `j` of the stored triangle serves both `A[i, j]` (a contiguous dot) and its mirror
     * `A[j, i]` (a contiguous axpy), so symmetry never forces a strided walk. The offsets let [symm]
     * apply this straight to a column of its operands, which are contiguous runs in this layout.
     */
    @Suppress("LongParameterList") // two buffers with offsets plus the triangle flag
    private fun symvAccumulate(
        alpha: Double,
        ad: DoubleArray,
        n: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        lower: Boolean,
    ) {
        for (j in 0 until n) {
            val base = j + j * n
            val xj = alpha * x[xOff + j]
            if (lower) {
                val len = n - j - 1
                y[yOff + j] += alpha * denseDot(ad, base + 1, x, xOff + j + 1, len) + xj * ad[base]
                if (xj != 0.0) denseAxpy(y, yOff + j + 1, xj, ad, base + 1, len)
            } else {
                y[yOff + j] += alpha * denseDot(ad, j * n, x, xOff, j) + xj * ad[base]
                if (xj != 0.0) denseAxpy(y, yOff, xj, ad, j * n, j)
            }
        }
    }

    /** `A[i, j]` of a symmetric `n×n` matrix stored in its [lower] (or upper) triangle only. */
    private fun symEntry(ad: DoubleArray, n: Int, i: Int, j: Int, lower: Boolean): Double {
        val hi = if (i > j) i else j
        val lo = if (i > j) j else i
        return if (lower) ad[hi + lo * n] else ad[lo + hi * n]
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod") // the BLAS dsymm signature
    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        require(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
        val m = a.rows
        require(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        val cd = c.data
        if (beta == 0.0) {
            cd.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(cd, 0, beta, cd.size)
        }
        if (right) {
            require(b.cols == m) { "symm right: B has ${b.cols} cols, expected $m" }
            if (alpha == 0.0 || m == 0 || b.rows == 0) return
            // C = alpha·B·A: column j of the product is alpha·Σₚ B[:,p]·A[p,j], so every run is a
            // contiguous column of B and C and the symmetric entry is a scalar read.
            val rows = b.rows
            val ad = a.data
            val bd = b.data
            for (j in 0 until m) {
                for (p in 0 until m) {
                    val apj = alpha * symEntry(ad, m, p, j, lower)
                    if (apj != 0.0) denseAxpy(cd, j * rows, apj, bd, p * rows, rows)
                }
            }
            return
        }
        require(b.rows == m) { "symm: B has ${b.rows} rows, expected $m" }
        if (alpha == 0.0 || m == 0 || b.cols == 0) return
        // C = alpha·A·B: column j of the product is alpha·A·B[:,j], a symv applied straight to the
        // contiguous columns of B and C.
        for (j in 0 until b.cols) {
            symvAccumulate(alpha, a.data, m, b.data, j * m, cd, j * m, lower)
        }
    }

    @Suppress("CyclomaticComplexMethod", "LongMethod") // netlib dsytf2's control flow, kept recognizable
    override fun ldl(a: DenseMatrix, workspace: Workspace?): LdlDecomposition {
        require(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val w = a.data.copyOf()
        val ipiv = IntArray(n)
        var singular = false
        val colK = workspace?.take(n) ?: DoubleArray(n)
        val colK1 = workspace?.take(n) ?: DoubleArray(n)
        var k = 0
        while (k < n) {
            var kstep = 1
            val absakk = abs(w[k + k * n])
            var imax = k
            var colmax = 0.0
            // The pivot column is a contiguous run in this layout, so the search sweeps it linearly.
            for (i in k + 1 until n) {
                val v = abs(w[i + k * n])
                if (v > colmax) {
                    colmax = v
                    imax = i
                }
            }
            if (maxOf(absakk, colmax) == 0.0) {
                // Zero pivot column: record and move on without eliminating, as dsytf2 does.
                singular = true
                ipiv[k] = k + 1
                k += 1
                continue
            }
            val kp: Int
            if (absakk >= BUNCH_KAUFMAN_ALPHA * colmax) {
                kp = k
            } else {
                var rowmax = 0.0
                for (j in k until imax) rowmax = maxOf(rowmax, abs(w[imax + j * n]))
                for (j in imax + 1 until n) rowmax = maxOf(rowmax, abs(w[j + imax * n]))
                if (absakk * rowmax >= BUNCH_KAUFMAN_ALPHA * colmax * colmax) {
                    kp = k
                } else if (abs(w[imax + imax * n]) >= BUNCH_KAUFMAN_ALPHA * rowmax) {
                    kp = imax
                } else {
                    kp = imax
                    kstep = 2
                }
            }
            val kk = k + kstep - 1
            if (kp != kk) {
                // Interchange rows/columns kk and kp of the trailing submatrix, dsytf2-style: the
                // already-computed L columns to the left stay put; solve replays ipiv instead.
                for (i in kp + 1 until n) {
                    val t = w[i + kk * n]
                    w[i + kk * n] = w[i + kp * n]
                    w[i + kp * n] = t
                }
                for (j in kk + 1 until kp) {
                    val t = w[j + kk * n]
                    w[j + kk * n] = w[kp + j * n]
                    w[kp + j * n] = t
                }
                val t = w[kk + kk * n]
                w[kk + kk * n] = w[kp + kp * n]
                w[kp + kp * n] = t
                if (kstep == 2) {
                    val s = w[(k + 1) + k * n]
                    w[(k + 1) + k * n] = w[kp + k * n]
                    w[kp + k * n] = s
                }
            }
            if (kstep == 1) {
                // A[k+1.., k+1..] -= d11·col·colᵀ with col the raw pivot column, then scale that column
                // into multipliers. Column j of the update reads the pivot column and writes its own,
                // both contiguous, so no staging copy is needed and the scaling is one denseScale.
                val d11 = 1.0 / w[k + k * n]
                for (j in k + 1 until n) {
                    val f = -d11 * w[j + k * n]
                    if (f != 0.0) denseAxpy(w, j + j * n, f, w, j + k * n, n - j)
                }
                denseScale(w, k + 1 + k * n, d11, n - k - 1)
                ipiv[k] = kp + 1
            } else {
                if (k < n - 2) {
                    // 2×2 pivot block: netlib's d21-scaled inverse, applied column-wise via two axpys.
                    // The transformed multipliers have to be staged, because the update still reads the
                    // raw columns k and k+1 that they will replace.
                    val d21 = w[(k + 1) + k * n]
                    val d11 = w[(k + 1) + (k + 1) * n] / d21
                    val d22 = w[k + k * n] / d21
                    val t = 1.0 / (d11 * d22 - 1.0)
                    val d21inv = t / d21
                    for (j in k + 2 until n) {
                        val cj = w[j + k * n]
                        val cj1 = w[j + (k + 1) * n]
                        colK[j] = d21inv * (d11 * cj - cj1)
                        colK1[j] = d21inv * (d22 * cj1 - cj)
                    }
                    for (j in k + 2 until n) {
                        val fj = colK[j]
                        val fj1 = colK1[j]
                        if (fj != 0.0) denseAxpy(w, j + j * n, -fj, w, j + k * n, n - j)
                        if (fj1 != 0.0) denseAxpy(w, j + j * n, -fj1, w, j + (k + 1) * n, n - j)
                    }
                    for (i in k + 2 until n) {
                        w[i + k * n] = colK[i]
                        w[i + (k + 1) * n] = colK1[i]
                    }
                }
                ipiv[k] = -(kp + 1)
                ipiv[k + 1] = -(kp + 1)
            }
            k += kstep
        }
        if (workspace != null) {
            workspace.release(colK1)
            workspace.release(colK)
        }
        return LdlDecomposition(n, w, ipiv, singular)
    }

    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray {
        val n = ldl.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        require(out.size == n) { "solve: out length ${out.size} != $n" }
        val w = ldl.ldl
        val ipiv = ldl.ipiv
        val x = out
        if (out !== b) b.copyInto(out)
        // Forward: replay interchanges, apply L block columns, divide by the D blocks (dsytrs lower).
        var k = 0
        while (k < n) {
            if (ipiv[k] > 0) {
                val kp = ipiv[k] - 1
                if (kp != k) {
                    val t = x[k]
                    x[k] = x[kp]
                    x[kp] = t
                }
                val xk = x[k]
                if (xk != 0.0) denseAxpy(x, k + 1, -xk, w, k + 1 + k * n, n - k - 1)
                x[k] = xk / w[k + k * n]
                k += 1
            } else {
                val kp = -ipiv[k] - 1
                if (kp != k + 1) {
                    val t = x[k + 1]
                    x[k + 1] = x[kp]
                    x[kp] = t
                }
                val xk = x[k]
                val xk1 = x[k + 1]
                val len = n - k - 2
                if (xk != 0.0) denseAxpy(x, k + 2, -xk, w, k + 2 + k * n, len)
                if (xk1 != 0.0) denseAxpy(x, k + 2, -xk1, w, k + 2 + (k + 1) * n, len)
                val akm1k = w[(k + 1) + k * n]
                val akm1 = w[k + k * n] / akm1k
                val ak = w[(k + 1) + (k + 1) * n] / akm1k
                val denom = akm1 * ak - 1.0
                val bkm1 = xk / akm1k
                val bk = xk1 / akm1k
                x[k] = (ak * bkm1 - bk) / denom
                x[k + 1] = (akm1 * bk - bkm1) / denom
                k += 2
            }
        }
        // Backward: apply Lᵀ block rows and undo the interchanges in reverse.
        k = n - 1
        while (k >= 0) {
            if (ipiv[k] > 0) {
                x[k] -= denseDot(w, k + 1 + k * n, x, k + 1, n - k - 1)
                val kp = ipiv[k] - 1
                if (kp != k) {
                    val t = x[k]
                    x[k] = x[kp]
                    x[kp] = t
                }
                k -= 1
            } else {
                val k0 = k - 1
                val len = n - k - 1
                val s0 = x[k0] - denseDot(w, k + 1 + k0 * n, x, k + 1, len)
                val s1 = x[k] - denseDot(w, k + 1 + k * n, x, k + 1, len)
                x[k0] = s0
                x[k] = s1
                val kp = -ipiv[k] - 1
                if (kp != k) {
                    val t = x[k]
                    x[k] = x[kp]
                    x[kp] = t
                }
                k -= 2
            }
        }
        return x
    }

    override fun qr(a: DenseMatrix, workspace: Workspace?): QrDecomposition {
        val m = a.rows
        val n = a.cols
        val k = minOf(m, n)
        val buf = a.data.copyOf()
        val tau = DoubleArray(k)
        for (col in 0 until k) {
            tau[col] = householderColumn(buf, m, col)
            val t = tau[col]
            if (t == 0.0) continue
            // Apply H = I − tau·v·vᵀ to each trailing column independently: the reflector and the column
            // it updates are both contiguous runs, so this is one dot for vᵀ·A[:,c] and one axpy back.
            // The row-major form needed an n-wide accumulator here; this needs none, so the workspace
            // argument is accepted for signature compatibility and left unused.
            val len = m - col - 1
            val vBase = col + 1 + col * m
            for (c in col + 1 until n) {
                val head = col + c * m
                val s = buf[head] + denseDot(buf, vBase, buf, head + 1, len)
                val f = -t * s
                if (f != 0.0) {
                    buf[head] += f
                    denseAxpy(buf, head + 1, f, buf, vBase, len)
                }
            }
        }
        return QrDecomposition(m, n, buf, tau)
    }

    /** Build the Householder reflector for [col] in place (LAPACK `dlarfg`): returns `tau`, stores the
     *  scaled vector below the diagonal and `beta` (the new `R` diagonal) on it. `tau == 0` means the
     *  column was already zero below the diagonal position and `H = I`. The column is contiguous, so the
     *  norm is a self-dot over it. */
    private fun householderColumn(buf: DoubleArray, m: Int, col: Int): Double {
        val base = col + col * m
        val len = m - col
        val normSq = denseDot(buf, base, buf, base, len)
        if (normSq == 0.0) return 0.0
        val alpha = buf[base]
        val norm = sqrt(normSq)
        val beta = if (alpha >= 0.0) -norm else norm
        // Divided rather than scaled by a reciprocal: a subnormal v0 would make 1/v0 overflow, where the
        // division stays finite.
        val v0 = alpha - beta
        for (i in base + 1 until base + len) buf[i] /= v0
        buf[base] = beta
        return (beta - alpha) / beta
    }

    override fun applyQInto(qr: QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        val m = qr.m
        require(y.size == m) { "applyQ: y length ${y.size} != $m" }
        require(out.size == m) { "applyQ: out length ${out.size} != $m" }
        val x = out
        if (out !== y) y.copyInto(out)
        val k = qr.tau.size
        val buf = qr.qr
        // Q = H_0·H_1···H_{k−1} and each H is symmetric, so Qᵀ applies them in ascending order and
        // Q in descending. Each reflector is a contiguous run, so both halves vectorize.
        val order = if (transpose) 0 until k else k - 1 downTo 0
        for (col in order) {
            val t = qr.tau[col]
            if (t == 0.0) continue
            val len = m - col - 1
            val vBase = col + 1 + col * m
            val s = x[col] + denseDot(buf, vBase, x, col + 1, len)
            val f = t * s
            x[col] -= f
            if (f != 0.0) denseAxpy(x, col + 1, -f, buf, vBase, len)
        }
        return x
    }

    override fun factor(a: DenseMatrix): LuDecomposition {
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        return factorCore(a, LuDecomposition(n, DoubleArray(n * n), IntArray(n), singular = false))
    }

    override fun factorInto(a: DenseMatrix, out: LuDecomposition): LuDecomposition {
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        require(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        return factorCore(a, out)
    }

    /** Doolittle LU with partial pivoting, writing into [out]'s buffers. */
    private fun factorCore(a: DenseMatrix, out: LuDecomposition): LuDecomposition {
        val n = out.n
        val lu = out.lu
        a.data.copyInto(lu)
        val piv = out.piv
        for (i in 0 until n) piv[i] = i
        var singular = false
        for (k in 0 until n) {
            // The pivot column is contiguous, so the search is a linear sweep.
            var p = k
            var max = abs(lu[k + k * n])
            for (i in k + 1 until n) {
                val v = abs(lu[i + k * n])
                if (v > max) {
                    max = v
                    p = i
                }
            }
            if (max == 0.0) {
                singular = true
                continue
            }
            if (p != k) {
                // The one strided step in this layout, and the one dgetrf pays too: a row interchange
                // touches one element per column.
                for (j in 0 until n) {
                    val t = lu[k + j * n]
                    lu[k + j * n] = lu[p + j * n]
                    lu[p + j * n] = t
                }
                val tp = piv[k]
                piv[k] = piv[p]
                piv[p] = tp
            }
            // Multipliers first, into the pivot column; then one axpy per trailing column, each reading
            // that column and writing its own. Divided rather than reciprocal-scaled so a subnormal pivot
            // cannot turn into an infinity.
            val pivot = lu[k + k * n]
            val len = n - k - 1
            val colBase = k + 1 + k * n
            for (i in k + 1 until n) lu[i + k * n] = lu[i + k * n] / pivot
            for (j in k + 1 until n) {
                val ukj = lu[k + j * n]
                if (ukj != 0.0) denseAxpy(lu, k + 1 + j * n, -ukj, lu, colBase, len)
            }
        }
        out.singular = singular
        return out
    }

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray {
        val n = lu.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        require(out.size == n) { "solve: out length ${out.size} != $n" }
        val a = lu.lu
        val piv = lu.piv
        return if (transpose) {
            solveTranspose(n, a, piv, b, out, workspace)
        } else {
            solveNormal(n, a, piv, b, out, workspace)
        }
    }

    // A x = b, with P A = L U: permute b by P, forward-solve unit-lower L, back-solve U.
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    private fun solveNormal(
        n: Int,
        a: DoubleArray,
        piv: IntArray,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray {
        // The permutation may read a slot of b that has already been written when out aliases b, so the
        // gather goes through a staging pass only in that case.
        if (out === b) {
            val staged = workspace?.take(n) ?: DoubleArray(n)
            for (i in 0 until n) staged[i] = b[piv[i]]
            staged.copyInto(out)
            workspace?.release(staged)
        } else {
            for (i in 0 until n) out[i] = b[piv[i]]
        }
        trsvCore(a, n, out, lower = true, transpose = false, unitDiag = true)
        trsvCore(a, n, out, lower = false, transpose = false, unitDiag = false)
        return out
    }

    // Aᵀ x = b, with Aᵀ = Uᵀ Lᵀ P: forward-solve Uᵀ (lower), back-solve unit-upper Lᵀ, then un-permute.
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    private fun solveTranspose(
        n: Int,
        a: DoubleArray,
        piv: IntArray,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray {
        // The triangular solves run on a staged copy; the scatter that follows cannot be done in place.
        val y = workspace?.take(n) ?: DoubleArray(n)
        b.copyInto(y)
        trsvCore(a, n, y, lower = false, transpose = true, unitDiag = false)
        trsvCore(a, n, y, lower = true, transpose = true, unitDiag = true)
        for (i in 0 until n) out[piv[i]] = y[i] // undo the row permutation
        workspace?.release(y)
        return out
    }
}

/**
 * The best backend the current platform provides, or null when only the portable reference is available.
 * On the JVM this discovers providers via `ServiceLoader`, so an optional backend artifact (such as
 * koblas-openblas) activates itself when present on the classpath. Other targets have no runtime
 * discovery and return null; a backend artifact there is activated explicitly through
 * [installLinearAlgebra].
 */
expect fun platformLinearAlgebra(): LinearAlgebra?

/**
 * Registers the backends koblas ships for this platform, once, on the first [koblas] read.
 *
 * The JVM half-registers its host-OpenBLAS backend here; the native targets do it eagerly before `main`
 * instead, and the web targets have nothing to register.
 */
internal expect fun registerPlatformBackends()

/** Runs [registerPlatformBackends] exactly once, on the first [koblas] read. */
private val platformRegistration: Unit by lazy { registerPlatformBackends() }

private val platformDefault: LinearAlgebra by lazy { platformLinearAlgebra() ?: ReferenceLinearAlgebra }

@Volatile
private var installed: LinearAlgebra? = null

@Volatile
private var registeredBlas: Blas? = null

@Volatile
private var registeredLapack: Lapack? = null

/**
 * The composed backend, rebuilt only when a registration changes.
 *
 * [koblas] is read on every dispatched call, so it has to be a field read: composing on demand would
 * allocate per call.
 */
@Volatile
private var resolved: LinearAlgebra? = null

/** The active backend: an [installLinearAlgebra] override when set, else the strongest registered half
 *  of each kind, else the [platformLinearAlgebra] backend, else the portable [ReferenceLinearAlgebra]. */
val koblas: LinearAlgebra
    get() {
        // One volatile read to make sure the platform's own backends had their chance to register. Doing
        // it here rather than inside platformDefault keeps that value meaning "what discovery found", so
        // clearing the registry still falls back to it rather than to a frozen composite.
        platformRegistration
        return installed ?: resolved ?: platformDefault
    }

/**
 * Pairs a [Blas] and a [Lapack] into one backend.
 *
 * Only used when the two halves come from different places; when one backend won both, that object is
 * used directly, so the common case keeps a single virtual call rather than two.
 */
private class ComposedBackend(private val blas: Blas, private val lapack: Lapack) :
    LinearAlgebra,
    Blas by blas,
    Lapack by lapack {
    override val name: String get() = if (blas.name == lapack.name) blas.name else "${blas.name}+${lapack.name}"
    override val priority: Int get() = maxOf(blas.priority, lapack.priority)
}

/** Recomputes [resolved] from the registered halves, falling back to the platform backend for either. */
private fun recompose() {
    val blas = registeredBlas
    val lapack = registeredLapack
    resolved = when {
        blas == null && lapack == null -> null
        blas === lapack && blas is LinearAlgebra -> blas
        else -> ComposedBackend(blas ?: platformDefault, lapack ?: platformDefault)
    }
}

/** What this runtime resolved on both performance seams, for startup logging — e.g.
 *  `"backend=openblas, primitives=simd(8 lanes)"` ([koblas].name and [mathBackend]). */
val koblasInfo: String get() = "backend=${koblas.name}, primitives=$mathBackend"

/**
 * Overrides the backend [koblas] resolves to, taking precedence over every automatic mechanism —
 * registration and platform discovery alike. Passing null restores automatic selection. The switch
 * is visible to subsequent [koblas] reads but is not synchronized with operations in flight.
 */
fun installLinearAlgebra(backend: LinearAlgebra?) {
    installed = backend
}

/**
 * Offers [backend] for automatic selection: it becomes active only while no [installLinearAlgebra]
 * override is set and no previously registered backend has a higher [LinearAlgebra.priority]. This
 * is how backend artifacts activate themselves on platforms without classpath discovery — the
 * koblas-cblas startup registration goes through here — and it keeps the strongest of several
 * candidates: openblas over cblas over the reference.
 */
fun registerLinearAlgebra(backend: LinearAlgebra) {
    registerBlas(backend)
    registerLapack(backend)
}

/**
 * Offers the BLAS half of [backend] for automatic selection, ranked by [Backend.priority] against other
 * registered BLAS backends. Register the halves separately when a host provides one library and not the
 * other; [registerLinearAlgebra] is the shorthand for registering the same object as both.
 */
fun registerBlas(backend: Blas) {
    val current = registeredBlas
    if (current == null || backend.priority > current.priority) {
        registeredBlas = backend
        recompose()
    }
}

/** Offers the LAPACK half of [backend] for automatic selection; see [registerBlas]. */
fun registerLapack(backend: Lapack) {
    val current = registeredLapack
    if (current == null || backend.priority > current.priority) {
        registeredLapack = backend
        recompose()
    }
}

/** Test hook: clears registration state so selection tests are order-independent. */
internal fun resetRegisteredLinearAlgebra() {
    registeredBlas = null
    registeredLapack = null
    recompose()
}

/** LU-factorize this square matrix with the active backend ([koblas]). */
fun DenseMatrix.lu(): LuDecomposition = koblas.factor(this)

/** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for this factorization with the active backend. */
fun LuDecomposition.solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.solve(this, b, transpose)

/** Matrix-matrix product `this · other` with the active backend. */
fun DenseMatrix.matMul(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)
