@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.sparse.SparseVectorKernels
import kotlin.math.abs
import kotlin.math.sqrt

// Arithmetic over [VectorView] / [MatrixView] as free functions; the view types stay read-only.
//
// Iteration goes through [forEachStored], which dispatches dense to all-indices and sparse to stored
// entries. Dense-by-dense paths call the active `VectorKernels` through `koblas.vectorKernels` -- the compiled-in
// SIMD or scalar kernels, or a registered host BLAS for runs long enough to pay for the call.
//
// Naming: mutating functions take the destination first and return [Unit] (`scale`,
// `axpy`, `ger`); allocating functions return a fresh result (`gemv`,
// infix `dot`).
//
// Naming follows the library-wide rule: BLAS routines keep their standard mnemonics, LAPACK routines get
// English names. Two deliberate exceptions live here. `norm2` and `asum` spell out what BLAS calls
// `nrm2` and `asum`, because `norm2` pairs with `norm1` (which is LAPACK `dlange`, not BLAS) and reading
// them side by side matters more than matching four characters exactly. And `iamax` keeps its mnemonic
// rather than becoming `indexOfMaxAbs`, since it is unambiguous to anyone who has met BLAS.

/**
 * Visit each stored entry of [this] as `(index, value)`, in ascending index order for any storage.
 * For [DenseVector] that's every index in `0 until size`; for [SparseVector] that's the entries present
 * in the parallel index/value arrays (which may include numerical zeros); for any other [VectorLike] it is
 * every index, read through [VectorLike.get].
 */
inline fun VectorLike.forEachStored(block: (i: Int, v: Double) -> Unit) {
    when (this) {
        is DenseVector -> {
            val d = data
            for (i in 0 until d.size) block(i, d[i])
        }

        is SparseVector -> {
            val idx = indices
            val vals = values
            for (k in idx.indices) block(idx[k], vals[k])
        }

        // A foreign VectorLike: koblas cannot know which entries it considers stored, so every index is
        // visited, which is what "stored" means for a dense vector anyway.
        else -> for (i in 0 until size) block(i, this[i])
    }
}

/**
 * `aT * b`. Dense×dense routes through the active [com.eignex.koblas.dense.VectorKernels]; a mixed pair
 * walks the sparse side and gathers from the dense one; anything else is read entry by entry.
 *
 * Every combination involving a sparse operand goes through [SparseVectorKernels], where a host sparse
 * BLAS could replace it: `usdot` against a dense vector, and a single-pass merge of the two ascending
 * index lists when both are sparse.
 */
infix fun VectorLike.dot(other: VectorLike): Double {
    requireShape(size == other.size) { "size mismatch: $size vs ${other.size}" }
    if (this is DenseVector && other is DenseVector) {
        return koblas.vectorKernels.dot(data, 0, other.data, 0, size)
    }
    if (this is SparseVector && other is SparseVector) return koblas.sparseVectorKernels.dot(this, other)
    if (this is SparseVector && other is DenseVector) return koblas.sparseVectorKernels.dot(this, other.data)
    if (this is DenseVector && other is SparseVector) return koblas.sparseVectorKernels.dot(other, data)
    // At least one operand is a foreign VectorLike, so neither storage is known: the generic path, reached
    // through `get` on both sides. Unreachable while the view roots were sealed, and reachable since they
    // opened, which is what an adapter from another library arrives as.
    var s = 0.0
    for (i in 0 until size) s += this[i] * other[i]
    return s
}

/**
 * Euclidean norm `||v||₂` (BLAS `dnrm2`). Sparse vectors sum over stored entries only.
 *
 * The fast path is a plain `sqrt(sum of squares)`; when that sum overflows or drowns in underflow
 * (components beyond roughly `1e±150`), a rescaled two-pass recovers the netlib-accurate result, so
 * any finite input yields the correct norm.
 */
fun norm2(v: VectorLike): Double = when (v) {
    is DenseVector -> koblas.vectorKernels.nrm2(v.data, 0, v.size)

    is SparseVector -> koblas.sparseVectorKernels.nrm2(v)

    // A foreign vector has no backing to hand a kernel, so it is materialised first. The rescaling this
    // needs is not worth reimplementing per storage.
    else -> euclideanNorm(v.toDoubleArray(), 0, v.size)
}

/** Sum of absolute values `Sum |v_i|` (BLAS `dasum`). Sparse vectors sum over stored entries only. */
fun asum(v: VectorLike): Double = when (v) {
    is DenseVector -> koblas.vectorKernels.asum(v.data, 0, v.size)

    is SparseVector -> koblas.sparseVectorKernels.asum(v)

    else -> {
        var s = 0.0
        v.forEachStored { _, x -> s += abs(x) }
        s
    }
}

/**
 * Index of the first entry with maximal `|v_i|` (BLAS `idamax`), or `-1` for a zero-length vector.
 * "First" is by index for either storage: a [SparseVector]'s stored entries are ascending, so its
 * storage order *is* index order and the two contracts agree. An all-unstored (zero) vector returns index
 * 0, matching the dense zero vector.
 */
fun iamax(v: VectorLike): Int {
    if (v.size == 0) return -1
    var best = -1
    var bestAbs = -1.0
    v.forEachStored { i, x ->
        val a = abs(x)
        if (a > bestAbs) {
            bestAbs = a
            best = i
        }
    }
    return if (best == -1) 0 else best // no stored entries: the zero vector's max is its first element
}

/** `dst = src` (BLAS `dcopy`). Dense sources bulk-copy; sparse sources zero-fill then scatter. */
fun copy(src: VectorLike, dst: DenseVector) {
    requireShape(src.size == dst.size) { "size mismatch: ${src.size} vs ${dst.size}" }
    when (src) {
        is DenseVector -> src.data.copyInto(dst.data)

        is SparseVector -> {
            dst.data.fill(0.0)
            koblas.sparseVectorKernels.scatter(src, dst.data)
        }

        else -> {
            dst.data.fill(0.0)
            src.forEachStored { i, v -> dst.data[i] = v }
        }
    }
}

/** Exchange the contents of [a] and [b] (BLAS `dswap`). */
fun swap(a: DenseVector, b: DenseVector) {
    requireShape(a.size == b.size) { "size mismatch: ${a.size} vs ${b.size}" }
    val ad = a.data
    val bd = b.data
    for (i in ad.indices) {
        val t = ad[i]
        ad[i] = bd[i]
        bd[i] = t
    }
}

/**
 * A plane rotation: the cosine and sine that [rot] applies, plus the length it collapsed a pair to.
 *
 * @property c the cosine.
 * @property s the sine.
 * @property r the rotated length, `±hypot(a, b)` — what `a` becomes and what `b` becomes zero from.
 */
class Givens internal constructor(val c: Double, val s: Double, val r: Double)

/**
 * Generate the plane rotation that zeroes [b] against [a] (BLAS `drotg`).
 *
 * Scales by the larger magnitude before squaring, so a pair whose squares would overflow or vanish still
 * produces the right rotation — the same reason [norm2] rescales, and not optional: `hypot(1e200, 1e200)`
 * is representable while `1e200²` is not.
 *
 * Follows netlib `drotg`'s sign convention rather than inventing one, since a caller reconstructing the
 * factorization depends on it: `r` takes the sign of whichever input was larger in magnitude. The
 * degenerate all-zero pair yields the identity rotation (`c = 1`, `s = 0`, `r = 0`) rather than a `NaN`.
 *
 * Together with [rot] this is what a factorization *update* needs — QR or Cholesky update and downdate,
 * and the Forrest-Tomlin basis update a simplex uses. koblas has none of those yet; this is the primitive
 * they all require.
 */
fun rotg(a: Double, b: Double): Givens {
    if (b == 0.0 && a == 0.0) return Givens(c = 1.0, s = 0.0, r = 0.0)
    val absA = abs(a)
    val absB = abs(b)
    val scale = maxOf(absA, absB)
    // Factor the larger out before squaring; both ratios are then at most 1.
    val ra = a / scale
    val rb = b / scale
    val magnitude = scale * sqrt(ra * ra + rb * rb)
    val r = if (absA > absB) {
        if (a >= 0.0) magnitude else -magnitude
    } else {
        if (b >= 0.0) magnitude else -magnitude
    }
    return Givens(c = a / r, s = b / r, r = r)
}

/**
 * Apply a plane rotation to a pair of vectors in place (BLAS `drot`): each `(x_i, y_i)` becomes
 * `(c·x_i + s·y_i, c·y_i − s·x_i)`.
 *
 * Deliberately off the [com.eignex.koblas.dense.VectorKernels] seam, for the reason that keeps `copy` and
 * `swap` off it: whether a foreign call beats the loop has to be measured, not assumed. A rotation does more
 * arithmetic per element than a copy, so it plausibly clears the bar where copy does not — but "plausibly" is
 * not a threshold. Adding it to the seam later is additive; guessing wrong now is not.
 */
fun rot(x: DenseVector, y: DenseVector, rotation: Givens) {
    requireShape(x.size == y.size) { "size mismatch: ${x.size} vs ${y.size}" }
    val c = rotation.c
    val s = rotation.s
    if (c == 1.0 && s == 0.0) return
    val xd = x.data
    val yd = y.data
    for (i in xd.indices) {
        val xi = xd[i]
        val yi = yd[i]
        xd[i] = c * xi + s * yi
        yd[i] = c * yi - s * xi
    }
}

/** `y = y + alpha * x`. Dense `x` uses SIMD; sparse `x` walks stored entries. */
fun axpy(y: DenseVector, alpha: Double, x: VectorLike) {
    requireShape(y.size == x.size) { "size mismatch: ${y.size} vs ${x.size}" }
    if (alpha == 0.0) return
    when (x) {
        is DenseVector -> koblas.vectorKernels.axpy(y.data, 0, alpha, x.data, 0, y.size)
        is SparseVector -> koblas.sparseVectorKernels.axpy(y.data, alpha, x)
        else -> x.forEachStored { i, v -> y.data[i] += alpha * v }
    }
}

/** `v = alpha * v`. */
fun scale(v: DenseVector, alpha: Double) {
    if (alpha == 1.0) return
    koblas.vectorKernels.scale(v.data, 0, alpha, v.size)
}

/**
 * Rank-one update `A = A + alpha · x · yᵀ` (BLAS `dger`). Subtract by passing `alpha = -1.0`.
 *
 * Two dense operands dispatch to the installed backend through [LinearAlgebra.ger]. A sparse or mixed
 * pair has no BLAS counterpart and stays here, visiting only the rows and columns where `x_i · y_j`
 * can be non-zero.
 */
fun ger(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix) {
    requireShape(a.rows == x.size && a.cols == y.size) {
        "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
    }
    if (alpha == 0.0) return
    if (x is DenseVector && y is DenseVector) {
        koblas.ger(alpha, x.data, y.data, a)
        return
    }
    // Mixed or sparse: no BLAS routine takes these, so walk the stored entries. Skipping a zero entry of
    // y skips a whole column of updates, which is the point of accepting a sparse operand at all. The
    // column is the outer loop because columns are the contiguous axis.
    val md = a.data
    val rows = a.rows
    y.forEachStored { j, yj ->
        if (yj != 0.0) {
            val col = j * rows
            val scaled = alpha * yj
            x.forEachStored { i, xi -> md[col + i] += scaled * xi }
        }
    }
}

/**
 * Symmetric rank-1 update `A += alpha · x · xᵀ` (BLAS `dsyr`); see [Blas.syr].
 *
 * The symmetric counterpart of [ger], and what accumulating a covariance one observation at a time is.
 */
fun syr(alpha: Double, x: VectorLike, a: DenseMatrix, uplo: Uplo = Uplo.FULL) = koblas.syr(alpha, x, a, uplo)

/** Symmetric rank-2 update `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`); see [Blas.syr2]. */
fun syr2(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix, uplo: Uplo = Uplo.FULL) =
    koblas.syr2(alpha, x, y, a, uplo)

/**
 * Matrix 1-norm: the maximum absolute column sum (LAPACK `dlange` with norm `1`). This is the `anorm`
 * input [LinearAlgebra.rcond] expects, computed on the matrix before factoring, so a solver that
 * estimates conditioning each refactorization calls both.
 *
 * Takes no workspace, unlike [LinearAlgebra.rcond], because it needs no scratch: a column is contiguous,
 * so each column sum completes before the next begins and one accumulator suffices. Row-major storage would
 * force a running total per column and therefore an `n`-wide array, which is the only reason such a routine
 * would take a workspace at all.
 */
fun norm1(a: DenseMatrix): Double {
    val ad = a.data
    val rows = a.rows
    var m = 0.0
    for (j in 0 until a.cols) {
        val base = j * rows
        var s = 0.0
        for (i in 0 until rows) s += abs(ad[base + i])
        if (s > m) m = s
    }
    return m
}

/**
 * Matrix infinity-norm: the maximum absolute row sum (LAPACK `dlange` with norm `I`).
 *
 * The awkward one under column-major storage, and the reason it takes a workspace where [norm1] does not.
 * A column is contiguous, so a column sum finishes before the next begins and one accumulator serves;
 * row sums all advance together, so this needs a `rows`-wide running total. That is exactly the array
 * `norm1` itself needed back when the storage was row-major.
 */
fun normInf(a: DenseMatrix, workspace: Workspace? = null): Double {
    val rows = a.rows
    if (rows == 0 || a.cols == 0) return 0.0
    val sums = workspace?.take(rows) ?: DoubleArray(rows)
    sums.fill(0.0, 0, rows) // take() promises nothing about the contents
    val ad = a.data
    for (j in 0 until a.cols) {
        val base = j * rows
        for (i in 0 until rows) sums[i] += abs(ad[base + i])
    }
    var m = 0.0
    for (i in 0 until rows) if (sums[i] > m) m = sums[i]
    workspace?.release(sums)
    return m
}

/**
 * Frobenius norm: `sqrt(Sum a_ij²)` (LAPACK `dlange` with norm `F`).
 *
 * One pass over the flat backing, and it reuses the shared rescaling rather than growing a third copy of
 * it — a matrix whose entries square out of range is no less likely than a vector's.
 */
fun normFro(a: DenseMatrix): Double = euclideanNorm(a.data, 0, a.data.size)

/**
 * Column [j] as a fresh vector — a contiguous `copyOfRange` of the backing, since a column of a
 * column-major matrix is exactly the run `data[j * rows until (j + 1) * rows]`.
 *
 * A copy rather than a view: [DenseVector] is a bare `DoubleArray` with no offset or stride, so there is
 * nothing for a view to be. Reach into [DenseMatrix.data] directly when the copy is the cost that matters.
 */
fun DenseMatrix.column(j: Int): DenseVector {
    require(j in 0 until cols) { "column $j outside [0,$cols)" }
    val start = j * rows
    return DenseVector.wrap(data.copyOfRange(start, start + rows))
}

/**
 * Row [i] as a fresh vector.
 *
 * The awkward direction under column-major storage, and the reason to prefer [column] where the algorithm
 * allows: consecutive entries of a row sit `rows` apart, so this gathers across the whole backing where
 * [column] copies one contiguous run.
 */
fun DenseMatrix.row(i: Int): DenseVector {
    require(i in 0 until rows) { "row $i outside [0,$rows)" }
    val out = DoubleArray(cols)
    for (j in 0 until cols) out[j] = data[i + j * rows]
    return DenseVector.wrap(out)
}

/**
 * Fresh transposed matrix `Aᵀ`. Always materializes; for products against a transposed operand,
 * prefer the `transpose` flags on [LinearAlgebra.gemv] / [LinearAlgebra.gemm], which read the
 * original storage without copying.
 */
fun DenseMatrix.transpose(): DenseMatrix {
    val t = DenseMatrix(cols, rows)
    val td = t.data
    // Reads down a contiguous source column and writes across a stride in the destination.
    for (j in 0 until cols) {
        val base = j * rows
        for (i in 0 until rows) td[j + i * cols] = data[base + i]
    }
    return t
}

/**
 * Fresh transposed matrix `Aᵀ`, still CSC — which makes this the CSC-to-CSR conversion as well.
 *
 * Worth more than symmetry with [DenseMatrix.transpose]: transposing compressed-sparse-column *is*
 * compressed-sparse-row of the original, so this is what a host library wanting row-major sparse input
 * needs (MKL's inspector-executor in CSR mode, some CHOLMOD paths). Groundwork for a host sparse backend
 * as much as a missing routine.
 *
 * Two counting passes, no searching: tally the entries per row of `this` to lay out the result's `colPtr`,
 * then scatter each entry to its place. `O(nnz + rows + cols)`.
 *
 * Explicitly stored zeros survive, because equality on [SparseMatrix] distinguishes them — dropping them
 * here would make `transpose().transpose()` a different matrix from the original.
 */
fun SparseMatrix.transpose(): SparseMatrix {
    val outPtr = IntArray(rows + 1)
    // Pass one: outPtr[i + 1] counts the entries in row i, then a prefix sum turns counts into offsets.
    for (k in rowIdx.indices) outPtr[rowIdx[k] + 1]++
    for (i in 0 until rows) outPtr[i + 1] += outPtr[i]
    val outIdx = IntArray(values.size)
    val outVal = DoubleArray(values.size)
    // Pass two: walking source columns in order means each destination column receives its row indices
    // ascending, which is the invariant the constructor requires.
    val next = outPtr.copyOf()
    for (j in 0 until cols) {
        for (k in colPtr[j] until colPtr[j + 1]) {
            val slot = next[rowIdx[k]]++
            outIdx[slot] = j
            outVal[slot] = values[k]
        }
    }
    return SparseMatrix(cols, rows, outPtr, outIdx, outVal)
}

/**
 * Matrix-vector product `A · x` into a fresh dense result (BLAS `dgemv` with `alpha = 1`, `beta = 0`).
 *
 * The view-taking overload of [Blas.gemv], for the same reason [ger] has one: a [SparseVector] operand
 * has no BLAS counterpart, and walking only its stored entries is the point of passing one. Two dense
 * operands dispatch to the backend, so this is a shape adapter rather than a second implementation.
 *
 * Every combination of the two storages resolves to a loop over stored entries only. That matters most
 * for a [SparseMatrix] operand: the generic fallback below reads through [MatrixView.get], which on CSC
 * is a search per entry, so a sparse matrix taking that path would cost `rows × cols` searches instead of
 * the `nnz` the representation exists to deliver.
 *
 * No `transpose` flag: for a sparse matrix use [com.eignex.koblas.sparse.gemv], which takes one.
 */
fun gemv(A: MatrixLike, x: VectorLike): DenseVector {
    requireShape(A.cols == x.size) { "gemv shape mismatch: A is ${A.rows}x${A.cols}, x size ${x.size}" }
    if (A is DenseMatrix && x is DenseVector) return DenseVector.wrap(koblas.gemv(A, x.data))
    val out = DenseVector(A.rows)
    val od = out.data
    if (A is DenseMatrix) {
        // One axpy per stored entry of x, down a contiguous column — so a sparse x touches only the
        // columns it has entries in, which is the reason this overload exists.
        val ad = A.data
        val rows = A.rows
        x.forEachStored { j, v ->
            if (v != 0.0) koblas.vectorKernels.axpy(od, 0, v, ad, j * rows, rows)
        }
    } else if (A is SparseMatrix) {
        // Column j of A scaled by x_j, accumulated: only the stored entries of both operands are read,
        // whichever storage x has.
        x.forEachStored { j, v ->
            if (v != 0.0) A.forEachInColumn(j) { i, aij -> od[i] += aij * v }
        }
    } else {
        // A foreign MatrixLike: every entry through `get`, which is why handing koblas its own storage
        // matters when the data is large.
        for (i in 0 until A.rows) {
            var s = 0.0
            x.forEachStored { j, v -> s += A[i, j] * v }
            od[i] = s
        }
    }
    return out
}
