@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.operations.*
import com.eignex.koblas.internal.scheduling.dense.*
import com.eignex.koblas.internal.scheduling.sparse.*
import com.eignex.koblas.internal.storage.*

/**
 * Dense and sparse BLAS through one immutable engine.
 *
 * BLAS routines are typed members. The portable engine runs the shared schedule over its own kernels; a
 * composed engine such as [koblas] routes each call to the provider its profile selects and otherwise runs the
 * same schedule over its own kernels; a vendor engine from [BuiltinEngines.vendor] calls its library for the
 * routines it exports and fails for the rest. [BuiltinEngines] names exact engines, while [koblas] selects its
 * standard platform configuration once.
 *
 * Operands are typed containers; dense views retain their offsets and strides. Routines validate logical
 * dimensions and preserve BLAS no-read rules. Copies, AXPY and matrix operations stage destructive aliases,
 * borrowing from a [Scratch] when one is given. Rotations and swaps have their own overlap requirements.
 * Primitive array windows belong to the internal kernel interface. Rounding, reduction order and unspecified
 * exceptional results belong to the selected arithmetic.
 */
public interface KoblasEngine {

    // Engine identity and provenance.

    /** Implementation identifier for diagnostics. */
    public val name: String

    /** Stable identifier used by routing profiles. */
    public val id: String get() = vendor?.vendorName?.lowercase() ?: name

    /** Callbacks supplied by this provider, trusted by the composer. Shared matrix schedules select arithmetic directly from leaf [providers]. */
    public val implementations: Set<EngineImplementation> get() = emptySet()

    /** Library provenance, when applicable. */
    public val vendor: Vendor? get() = null

    /** Resolved host library, when applicable. */
    public val libraryPath: String get() = ""

    /** Host library version, when applicable. */
    public val version: String get() = ""

    /** Evidence for the host library's compute thread count. */
    public val threadEvidence: ThreadEvidence get() = ThreadEvidence.Unconfirmed

    /** Leaf providers in routing order, including their own capabilities and host provenance. */
    public val providers: List<KoblasEngine> get() = listOf(this)

    // Typed vector and matrix operations.

    /** Dot product of equally sized vectors; sparse operands evaluate only stored positions. */
    public fun dot(a: DenseVector, b: DenseVector): Double = dotVector(a, b)

    /** Copies [a] into equally sized [b], preserving sparse source structure and staging destructive aliases. */
    public fun copy(a: DenseVector, b: DenseVector): Unit = copyIntoVector(a, b)

    /** Adds `alpha * x` into equally sized [y]; zero [alpha] reads no entries and aliases are staged. */
    public fun axpy(alpha: Double, x: DenseVector, y: DenseVector): Unit = axpyVector(alpha, x, y)

    /** Dot product of equally sized vectors; sparse operands evaluate only stored positions. */
    public fun dot(a: DenseVector, b: SparseVector): Double = dotVector(a, b)

    /** Copies [a] into equally sized [b], preserving sparse source structure and staging destructive aliases. */
    public fun copy(a: DenseVector, b: SparseVector): Unit = copyIntoVector(a, b)

    /** Adds `alpha * x` into equally sized [y]; zero [alpha] reads no entries and aliases are staged. */
    public fun axpy(alpha: Double, x: DenseVector, y: SparseVector): Unit = axpyVector(alpha, x, y)

    /** Dot product of equally sized vectors; sparse operands evaluate only stored positions. */
    public fun dot(a: SparseVector, b: DenseVector): Double = dotVector(a, b)

    /** Copies [a] into equally sized [b], preserving sparse source structure and staging destructive aliases. */
    public fun copy(a: SparseVector, b: DenseVector): Unit = copyIntoVector(a, b)

    /** Adds `alpha * x` into equally sized [y]; zero [alpha] reads no entries and aliases are staged. */
    public fun axpy(alpha: Double, x: SparseVector, y: DenseVector): Unit = axpyVector(alpha, x, y)

    /** Dot product of equally sized vectors; sparse operands evaluate only stored positions. */
    public fun dot(a: SparseVector, b: SparseVector): Double = dotVector(a, b)

    /** Copies [a] into equally sized [b], preserving sparse source structure and staging destructive aliases. */
    public fun copy(a: SparseVector, b: SparseVector): Unit = copyIntoVector(a, b)

    /** Adds `alpha * x` into equally sized [y]; zero [alpha] reads no entries and aliases are staged. */
    public fun axpy(alpha: Double, x: SparseVector, y: SparseVector): Unit = axpyVector(alpha, x, y)

    /** Scales stored entries in place, preserving views and sparse structure. Non-finite behavior follows the selected kernel. */
    public fun scale(v: DenseVector, alpha: Double): Unit = scaleVector(alpha, v)

    /** Rescaled Euclidean norm; sparse storage reduces only its stored values. An empty vector has norm zero. */
    public fun nrm2(v: DenseVector): Double = normVector(v, Norm.Two)

    /** Sum of absolute values of stored entries, or zero for an empty vector. */
    public fun asum(v: DenseVector): Double = absoluteSumVector(v)

    /** Plain sum of stored entries. Reduction order and rounding follow the selected kernel. */
    public fun sum(v: DenseVector): Double = sumVector(v)

    /** Scales stored entries in place, preserving views and sparse structure. Non-finite behavior follows the selected kernel. */
    public fun scale(v: SparseVector, alpha: Double): Unit = scaleVector(alpha, v)

    /** Rescaled Euclidean norm; sparse storage reduces only its stored values. An empty vector has norm zero. */
    public fun nrm2(v: SparseVector): Double = normVector(v, Norm.Two)

    /** Sum of absolute values of stored entries, or zero for an empty vector. */
    public fun asum(v: SparseVector): Double = absoluteSumVector(v)

    /** Plain sum of stored entries. Reduction order and rounding follow the selected kernel. */
    public fun sum(v: SparseVector): Double = sumVector(v)

    /** Logical index of a maximum-magnitude entry, or minus one when empty. Ties and NaNs follow the selected kernel. */
    public fun iamax(v: DenseVector): Int = kernels.iamax(v.values, v.offset, v.size, v.stride)

    /** Plane rotation of equally sized vectors. Equal mappings are safe; callers must stage other destructive overlaps. */
    public fun rot(x: DenseVector, y: DenseVector, c: Double, s: Double) {
        requireSameSize(x.size, y.size, "rot")
        kernels.rot(x.values, x.offset, y.values, y.offset, x.size, c, s, x.stride, y.stride)
    }

    /** Exchanges equally sized vectors. Callers must stage destructive overlaps, as for [rot]. */
    public fun swap(a: DenseVector, b: DenseVector) {
        requireSameSize(a.size, b.size, "swap")
        kernels.swap(a.values, a.offset, b.values, b.offset, a.size, a.stride, b.stride)
    }

    /** Writes stored [source] entries into [destination], leaving absent positions untouched. Aliased sources are staged. */
    public fun scatter(source: SparseVector, destination: DenseVector): Unit = scatterVector(source, destination)

    /** Reads [source] at [destination]'s stored indices. The vectors must have equal sizes and distinct coefficient buffers. */
    public fun gather(source: DenseVector, destination: SparseVector): Unit = gatherVector(source, destination, false)

    /** Reads and zeroes [source] at [destination]'s stored indices, with the size and buffer requirements of [gather]. */
    public fun gatherZero(source: DenseVector, destination: SparseVector): Unit = gatherVector(
        source,
        destination,
        true,
    )

    /**
     * `y = alpha * op(A) * x + beta * y`. Views retain their offsets and strides; aliases are staged.
     * A zero [alpha] reads neither matrix nor input vector, and a zero [beta] overwrites [y] without reading it.
     * [transpose] selects the transposed matrix.
     */
    public fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        transpose: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        kernels.gemv(alpha, a, x.values, x.offset, x.stride, beta, y.values, y.offset, y.stride, transpose, scratch)
    }

    /**
     * `y = alpha * op(A) * x + beta * y`. Views retain their offsets and strides; aliases are staged.
     * A zero [alpha] reads neither matrix nor input vector, and a zero [beta] overwrites [y] without reading it.
     * [transpose] selects the transposed matrix.
     */
    public fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        transpose: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        sparseMatrixVector(alpha, a, x, beta, y, transpose, scratch, symmetric = false)
    }

    /**
     * `y = alpha * A * x + beta * y`. Views retain their offsets and strides; aliases are staged.
     * A zero [alpha] reads neither matrix nor input vector, and a zero [beta] overwrites [y] without reading it.
     * [lower] selects the stored triangle of the symmetric matrix.
     */
    public fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        requireSymvOperands(a, x.size, y.size)
        kernels.symv(alpha, a, x.values, x.offset, x.stride, beta, y.values, y.offset, y.stride, lower, scratch)
    }

    /**
     * `y = alpha * A * x + beta * y`. Views retain their offsets and strides; aliases are staged.
     * A zero [alpha] reads neither matrix nor input vector, and a zero [beta] overwrites [y] without reading it.
     * [lower] selects the stored triangle of the symmetric matrix.
     */
    public fun symv(
        alpha: Double,
        a: SparseMatrix,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        requireSymvOperands(a, x.size, y.size)
        sparseMatrixVector(alpha, a, x, beta, y, lower, scratch, symmetric = true)
    }

    /** Adds `alpha * x * yᵀ` to [a], requiring matching logical dimensions and staging aliased inputs. */
    public fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion, scratch: Scratch? = null) {
        requireGerOperands(x.size, y.size, a)
        kernels.ger(alpha, x.values, x.offset, x.stride, y.values, y.offset, y.stride, a, scratch)
    }

    /** Adds `alpha * x * xᵀ` to the selected triangle of square [a]. Aliased inputs are staged. */
    public fun syr(
        alpha: Double,
        x: DenseVector,
        a: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        requireSyrOperands(a, x.size, "syr")
        kernels.syr(alpha, x.values, x.offset, x.stride, a, lower, scratch)
    }

    /** Adds `alpha * (x * yᵀ + y * xᵀ)` to the selected triangle of square [a]. Aliased inputs are staged. */
    public fun syr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        requireSyr2Operands(a, x.size, y.size, "syr2")
        kernels.syr2(alpha, x.values, x.offset, x.stride, y.values, y.offset, y.stride, a, lower, scratch)
    }

    /**
     * Solves `op(A) * x = b` in place using the selected triangle. [transpose] transposes it and [unitDiag] assumes a unit diagonal.
     * Views preserve their addressing and aliased matrix entries are staged. No singularity check is performed.
     */
    public fun trsv(
        a: DenseMatrixRegion,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        kernels.trsv(a, x.values, x.offset, x.stride, lower, transpose, unitDiag, scratch)
    }

    /**
     * Solves `op(A) * x = b` in place using the selected triangle. [transpose] transposes it and [unitDiag] assumes a unit diagonal.
     * Views preserve their addressing and aliased matrix entries are staged. No singularity check is performed.
     */
    public fun trsv(
        a: SparseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        sparseTriangularVector(a, x, lower, transpose, unitDiag, scratch, solve = true)
    }

    /**
     * Multiplies `x = op(A) * x` in place using the selected triangle. [transpose] transposes it and [unitDiag] assumes a unit diagonal.
     * Views preserve their addressing and aliased matrix entries are staged. No singularity check is performed.
     */
    public fun trmv(
        a: DenseMatrixRegion,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        kernels.trmv(a, x.values, x.offset, x.stride, lower, transpose, unitDiag, scratch)
    }

    /**
     * Multiplies `x = op(A) * x` in place using the selected triangle. [transpose] transposes it and [unitDiag] assumes a unit diagonal.
     * Views preserve their addressing and aliased matrix entries are staged. No singularity check is performed.
     */
    public fun trmv(
        a: SparseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        sparseTriangularVector(a, x, lower, transpose, unitDiag, scratch, solve = false)
    }

    /** Dot product of [column] with equally sized [dense]. Only stored matrix entries participate, including stored zeros. */
    public fun dotColumn(matrix: SparseMatrix, column: Int, dense: DenseVector): Double {
        requireIndex(column in 0 until matrix.cols) { "column $column is outside [0, ${matrix.cols})" }
        requireSameSize(matrix.rows, dense.size, "dotColumn")
        val start = matrix.colPointers[column]
        val count = matrix.colPointers[column + 1] - start
        return if (dense.isWholeArray) {
            kernels.dot(matrix.rowIndices, start, matrix.values, start, count, dense.values)
        } else {
            com.eignex.koblas.internal.kernels.scalar.sparse.indexedDotDense(
                matrix.rowIndices,
                start,
                matrix.values,
                start,
                count,
                dense.values,
                dense.offset,
                dense.stride,
            )
        }
    }

    // Dense matrix products and triangular operations.

    /**
     * `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *
     * An input that shares [c]'s storage is snapshotted first. A zero [alpha] or an empty shared dimension
     * scales [c] and reads no operand; a zero [beta] overwrites [c] without reading it.
     *
     * Scaling placement and accumulation order depend on the selected implementation, including whether
     * [alpha] scales operand entries or partial sums, so rounding, overflow and non-finite results may differ
     * between implementations and transpose modes.
     */
    public fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        scratch: Scratch? = null,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val routed = routedWholeCall(EngineImplementation.Gemm, c.rows, c.cols, depth, borrowed)
        if (routed != null) return routed.gemm(alpha, a, transposeA, b, transposeB, beta, c, scratch)
        requireGemmOperands(a, transposeA, b, transposeB, c)
        prepareProduct(alpha, a, b, beta, c, depth, null, null, null, scratch) { sa, sb ->
            scheduleGemm(alpha, sa, transposeA, sb, transposeB, beta, c, depth, scratch)
        }
    }

    /**
     * `C = alpha · op(A) · op(B) + beta · C` in only the selected triangle (Netlib `GEMMTR`, commonly exposed as
     * `gemmt`). `op(A)` is `n×k`, `op(B)` is `k×n`, and [c] is `n×n`. The opposite triangle is neither read nor
     * written. Overlap and multipliers follow [gemm].
     */
    public fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val routed = routedWholeCall(EngineImplementation.Gemmt, c.rows, c.cols, depth, borrowed)
        if (routed != null) return routed.gemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, scratch)
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { sa, sb ->
            scheduleGemmt(alpha, sa, transposeA, sb, transposeB, beta, c, lower, scratch)
        }
    }

    /**
     * `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
     * [lower] or upper triangle of the symmetric [a] is read. Overlap and multipliers follow [gemm].
     */
    public fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        right: Boolean = false,
        scratch: Scratch? = null,
    ) {
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val routed = routedWholeCall(EngineImplementation.Symm, c.rows, c.cols, a.rows, borrowed)
        if (routed != null) return routed.symm(alpha, a, b, beta, c, lower, right, scratch)
        requireSymmOperands(a, b, c, right)
        // One operand serving both sides needs a full snapshot, since the second reads the triangle the first skips.
        val aTriangle = if (sameRegion(a, b)) null else lower
        prepareProduct(alpha, a, b, beta, c, a.rows, null, aTriangle, this, scratch) { sa, sb ->
            scheduleSymm(alpha, sa, sb, beta, c, lower, right, scratch)
        }
    }

    /**
     * `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`). Only the
     * [lower] or upper triangle is written. One product, so [alpha] and the accumulation order carry the
     * latitude [gemm] describes.
     */
    public fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val depth = if (transpose) a.rows else a.cols
        val routed = routedWholeCall(
            EngineImplementation.Syrk,
            c.rows,
            c.cols,
            depth,
            a !is DenseMatrix || c !is DenseMatrix,
        )
        if (routed != null) return routed.syrk(alpha, a, transpose, beta, c, lower, scratch)
        requireSyrkOperands(a, transpose, c)
        prepareProduct(alpha, a, a, beta, c, depth, lower, null, null, scratch) { sa, _ ->
            scheduleSyrk(alpha, sa, transpose, beta, c, lower, scratch)
        }
    }

    /**
     * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     * [transpose]. Only the [lower] or upper triangle is written. The implementation may accumulate the two
     * products separately or in one traversal, with the rounding differences [gemm] describes.
     */
    public fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val depth = if (transpose) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val routed = routedWholeCall(EngineImplementation.Syr2k, c.rows, c.cols, depth, borrowed)
        if (routed != null) return routed.syr2k(alpha, a, b, transpose, beta, c, lower, scratch)
        requireSyr2kOperands(a, b, transpose, c)
        prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { sa, sb ->
            scheduleSyr2k(alpha, sa, sb, transpose, beta, c, lower, scratch)
        }
    }

    /**
     * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`). Flags follow
     * [trsv]; the right-hand sides are the columns of [b] from the left and its rows from the right.
     */
    public fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(
            EngineImplementation.Trsm,
            b.rows,
            b.cols,
            a.rows,
            a !is DenseMatrix || b !is DenseMatrix,
        )
        if (routed != null) return routed.trsm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
        prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trsm", scratch) { sa ->
            scheduleTriangularMatrix(alpha, sa, triangle(lower, unitDiag), transpose, b, right, true, scratch)
        }
    }

    /** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of [trsm]. */
    public fun trmm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(
            EngineImplementation.Trmm,
            b.rows,
            b.cols,
            a.rows,
            a !is DenseMatrix || b !is DenseMatrix,
        )
        if (routed != null) return routed.trmm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
        prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trmm", scratch) { sa ->
            scheduleTriangularMatrix(alpha, sa, triangle(lower, unitDiag), transpose, b, right, false, scratch)
        }
    }

    // Sparse matrix operations over validated CSC storage.

    /**
     * `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right], interpreting the square
     * sparse [a] as symmetric from exactly its selected [lower] or upper triangle. A zero [alpha] reads neither
     * input and a zero [beta] does not read [c]. Shared input and destination storage is staged.
     */
    public fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        right: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireSymmOperands(a, b, c, right)
        // Return before staging or loans: an empty axis may be paired with an enormous one.
        if (c.rows == 0 || c.cols == 0) return
        if (alpha == 0.0) return scaleRegion(c, beta, vectors = this)
        staged(scratch, a, a.values === c.values) { stableA ->
            stagedRegion(scratch, b, c) { stableB ->
                scaleRegion(c, beta, vectors = this)
                if (right) {
                    multiplySymmetricFromTheRight(this, alpha, stableA, stableB, c, lower)
                } else {
                    multiplySymmetricFromTheLeft(this, alpha, stableA, stableB, c, lower, scratch)
                }
            }
        }
    }

    /**
     * `C = alpha · op(A) · op(B) + beta · C` (Sparse BLAS `usmm`) for a sparse [a] against a dense [b], or
     * `C = alpha · op(B) · op(A) + beta · C` when [right]. A zero [beta] overwrites [c] without reading it.
     *
     * [right] exists because [a] is distinguished by being the sparse operand and cannot be swapped, so without a
     * side the product of a dense matrix by a sparse one would only be reachable by transposing the result.
     */
    public fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        right: Boolean = false,
        scratch: Scratch? = null,
    ) {
        // Right-side multiplication uses the same derivation with operands reversed.
        if (right) {
            requireGemmOperands(
                b,
                transposeB,
                a,
                transposeA,
                c,
            )
        } else {
            requireGemmOperands(a, transposeA, b, transposeB, c)
        }
        if (c.rows == 0 || c.cols == 0) return
        if (alpha == 0.0) return scaleRegion(c, beta, vectors = this)
        val depth = if (right) (if (transposeB) b.rows else b.cols) else (if (transposeA) a.rows else a.cols)
        staged(scratch, a, a.values === c.values) { stableA ->
            stagedRegion(scratch, b, c) { stableB ->
                scaleRegion(c, beta, vectors = this)
                if (right) {
                    multiplyFromTheRight(this, alpha, stableA, transposeA, stableB, transposeB, c, c.rows, scratch)
                } else {
                    multiplyFromTheLeft(
                        this, alpha, stableA, transposeA, stableB, transposeB, c, c.rows, c.cols, depth, scratch,
                    )
                }
            }
        }
    }

    /**
     * Dense selected-triangle `C = alpha · op(A) · op(A)ᵀ + beta · C` for a sparse [a]. Only the [lower] or upper
     * triangle of [c] is read or written, and products involving implicit zeros are not evaluated.
     */
    public fun syrk(
        alpha: Double,
        a: SparseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val n = if (transpose) a.cols else a.rows
        requireSyrkOperands(a, transpose, c)
        // An empty pattern needs only triangle scaling, avoiding potentially oversized row-indexed adjacency.
        if (alpha == 0.0 || a.nnz == 0 || n == 0) return scaleRegion(c, beta, lower)
        staged(scratch, a, a.values === c.values) { stableA ->
            scaleRegion(c, beta, lower)
            withSymmetricRankScratch(scratch, n, stableA.rows, stableA.nnz) {
                    sums,
                    touchedAt,
                    touched,
                    rowPointers,
                    adjacentColumns,
                    adjacentPositions,
                    rowCursor,
                ->
                symmetricRankInto(
                    alpha, stableA, transpose, c, lower, sums, touchedAt, touched,
                    rowPointers, adjacentColumns, adjacentPositions, rowCursor,
                )
            }
        }
    }

    /**
     * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (Sparse BLAS `ussm`), with the
     * flags of the sparse [trsv]. A missing diagonal entry behaves as zero and singularity is not reported. From
     * the right a stored zero off the diagonal is skipped, as reference `dtrsm` skips a zero coefficient.
     */
    public fun trsm(
        a: SparseMatrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ): Unit = sparseTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true, scratch)

    /**
     * `B = alpha · op(T) · B` in place, or `B = alpha · B · op(T)` when [right]. Zero or missing diagonal entries
     * participate normally. From the right a stored zero off the diagonal is skipped, as reference `dtrmm` skips
     * a zero coefficient.
     */
    public fun trmm(
        a: SparseMatrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ): Unit = sparseTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false, scratch)
}
