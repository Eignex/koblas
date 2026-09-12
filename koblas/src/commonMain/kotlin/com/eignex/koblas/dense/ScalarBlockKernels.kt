package com.eignex.koblas.dense

/** Selected output entries; [diagonalOffset] is the global row origin minus the global column origin. */
public class BlockOutput private constructor(private val lower: Boolean?, public val diagonalOffset: Int) {
    internal fun contains(row: Int, column: Int): Boolean = when (lower) {
        null -> true
        true -> row.toLong() + diagonalOffset >= column
        false -> row.toLong() + diagonalOffset <= column
    }

    /** Full or triangular output selections. */
    public companion object {
        /** Write every logical output entry. */
        public val Full: BlockOutput = BlockOutput(null, 0)

        /** Write the lower triangle, including the diagonal. */
        public fun lower(diagonalOffset: Int = 0): BlockOutput = BlockOutput(true, diagonalOffset)

        /** Write the upper triangle, including the diagonal. */
        public fun upper(diagonalOffset: Int = 0): BlockOutput = BlockOutput(false, diagonalOffset)
    }
}

/** Arithmetic order is part of a product contract, independently of the input's physical packing. */
public enum class ProductEvaluation {
    /** Increasing-depth dot product, followed by alpha and addition of beta-scaled C. */
    DotThenScale,

    /** Increasing-depth updates C += A(i, p) * (alpha * B(p, j)), skipping raw zero B coefficients. */
    OrderedUpdates,

    /** Increasing-depth updates as above, evaluating zero B coefficients as well. */
    ArithmeticUpdates,
}

/** Whether this block begins or continues a depth-split product. */
public enum class DepthContribution {
    /** Apply the supplied beta. */
    First,

    /** Accumulate onto C without reapplying beta. */
    Subsequent,
}

/**
 * Scalar reference for direct, left-packed, right-packed, both-packed, and retained matrix products. Operands
 * supply logical m, n, k; no packing group determines a compute tile or solve order. Every packed input must
 * have the matching left/right role. Layout and context rejection, scratch validation, and allocation occur
 * before output mutation. There is no retry after writes start.
 *
 * Alpha zero or depth zero scales only selected C entries and does not read A or B. Beta zero never loads
 * original C. Unselected output, array gaps, and padding are untouched. Nonalias calls need no scratch;
 * aliases stage the selected result in caller scratch or one temporary array, then commit it. No workspace
 * or native state survives the call. Exact consumers validate a retained SVL restriction on every call.
 */
public object ScalarBlockKernels {
    /**
     * Number of scratch doubles needed for an alias-safe call. A conservative shared-array check avoids an
     * address-set allocation. The supplied scratch must not share any operand's backing array.
     */
    public fun scratchSize(a: MatrixOperand, b: MatrixOperand, c: MatrixWindow, alpha: Double = 1.0): Int {
        requireShape(a, b, c)
        return if (alpha != 0.0 && a.columns != 0 && (a.buffer === c.buffer || b.buffer === c.buffer)) {
            checkedStorageSize(c.rows.toLong() * c.columns)
        } else {
            0
        }
    }

    /**
     * Computes the selected entries of alpha * A * B + beta * C with explicit [evaluation]. Baked operand
     * transforms are already represented in their values. [contribution] applies beta only to the first depth
     * block; splitting a dot into blocks can round differently, so ordered fallback callers keep depth order.
     * [svlBytes] is the consumer's current checked context, with zero meaning unknown. Scalar arithmetic itself
     * is length independent, but an operand's declared restriction remains enforced.
     */
    @Suppress("LongParameterList")
    public fun product(
        a: MatrixOperand,
        b: MatrixOperand,
        c: MatrixWindow,
        alpha: Double = 1.0,
        beta: Double = 0.0,
        output: BlockOutput = BlockOutput.Full,
        contribution: DepthContribution = DepthContribution.First,
        evaluation: ProductEvaluation = ProductEvaluation.DotThenScale,
        scratch: DoubleArray? = null,
        svlBytes: Int = 0,
    ) {
        requireShape(a, b, c)
        requirePacked(a, PackedRole.Left, svlBytes)
        requirePacked(b, PackedRole.Right, svlBytes)
        val required = scratchSize(a, b, c, alpha)
        if (scratch != null) {
            require(scratch.size >= required) { "insufficient product scratch" }
            require(scratch !== a.buffer && scratch !== b.buffer && scratch !== c.buffer) { "scratch aliases operand" }
        }
        if (c.rows == 0 || c.columns == 0) return
        val staged = if (required == 0) null else scratch ?: DoubleArray(required)
        val effectiveBeta = if (contribution == DepthContribution.First) beta else 1.0
        val hasProduct = alpha != 0.0 && a.columns != 0
        if (!hasProduct && effectiveBeta == 1.0) return
        for (j in 0 until c.columns) {
            for (i in 0 until c.rows) {
                if (!output.contains(i, j)) continue
                val original = when (effectiveBeta) {
                    0.0 -> 0.0
                    1.0 -> c.data[c.index(i, j)]
                    else -> effectiveBeta * c.data[c.index(i, j)]
                }
                val result = if (hasProduct) evaluate(a, b, i, j, alpha, original, evaluation) else original
                if (staged == null) c.data[c.index(i, j)] = result else staged[i + j * c.rows] = result
            }
        }
        if (staged != null) {
            for (j in 0 until c.columns) {
                for (i in 0 until c.rows) {
                    if (output.contains(i, j)) c.data[c.index(i, j)] = staged[i + j * c.rows]
                }
            }
        }
    }

    @Suppress("LongParameterList")
    private fun evaluate(
        a: MatrixOperand,
        b: MatrixOperand,
        row: Int,
        column: Int,
        alpha: Double,
        original: Double,
        evaluation: ProductEvaluation,
    ): Double {
        if (evaluation == ProductEvaluation.DotThenScale) {
            var sum = 0.0
            for (p in 0 until a.columns) sum += a.value(row, p) * b.value(p, column)
            return alpha * sum + original
        }
        var result = original
        for (p in 0 until a.columns) {
            val right = b.value(p, column)
            if (evaluation == ProductEvaluation.ArithmeticUpdates || right != 0.0) {
                result += a.value(row, p) * (alpha * right)
            }
        }
        return result
    }

    private fun requirePacked(operand: MatrixOperand, role: PackedRole, svlBytes: Int) {
        if (operand is PackedMatrix) {
            require(operand.layout.role == role) { "packed operand has wrong product role" }
            operand.layout.requireCompatible(PackedMatrixLayout.GROUPED_FP64, 1, svlBytes)
        }
    }

    private fun requireShape(a: MatrixOperand, b: MatrixOperand, c: MatrixWindow) {
        require(a.rows == c.rows && b.columns == c.columns && a.columns == b.rows) { "product shape mismatch" }
        require(c.structure == MatrixStructure.General) { "output window must be general" }
    }
}
