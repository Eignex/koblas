package com.eignex.koblas.dense

import com.eignex.koblas.*

/**
 * A symmetric indefinite factorization `A = L·D·Lᵀ` in LAPACK `dsytrf` lower packed form. [ldl] is a live
 * buffer, not a copy, so treat it as read-only. Use [pivotBlocks] for the factorization's 0-based pivot
 * structure; [rawLapackIpiv] exposes the raw signed, 1-based buffer that both koblas's own portable solve
 * and a native LAPACK binding read directly.
 *
 * @property n the matrix dimension.
 * @property ldl the packed factors, column-major, length `n * n`; only the lower triangle is meaningful.
 * @param ipiv the live raw LAPACK `IPIV` buffer, exposed as [rawLapackIpiv]. It contains LAPACK `dsytrf`'s
 * 1-based pivots: positive marks a 1×1 block, a negative pair a 2×2 block.
 * @param failedAt the zero pivot position, or [NOT_SINGULAR]; readable afterwards through [failedAt].
 */
public class F64PivotedSymmetricIndefiniteDecomposition @UnsafeKoblasApi constructor(
    public val n: Int,
    @property:UnsafeKoblasApi public val ldl: DoubleArray,
    ipiv: IntArray,
    failedAt: Int = NOT_SINGULAR,
) {
    /**
     * The live signed, 1-based LAPACK `dsytrf` `IPIV` buffer. koblas's own portable solve reads this
     * directly, not just a native LAPACK binding, so treat "raw" as "LAPACK's encoding," not "external only."
     */
    @UnsafeKoblasApi
    public val rawLapackIpiv: IntArray = ipiv

    /** The position whose pivot was exactly zero, or [NOT_SINGULAR]. This is `dsytrf`'s positive `info` made
     *  0-based. */
    public var failedAt: Int = failedAt
        internal set

    /** Whether a zero pivot was encountered. */
    public val singular: Boolean get() = failedAt != NOT_SINGULAR

    /** The Bunch-Kaufman pivot blocks in factor order, with all positions and interchanges 0-based. */
    public val pivotBlocks: List<F64PivotedSymmetricIndefinitePivotBlock>
        get() {
            val blocks = ArrayList<F64PivotedSymmetricIndefinitePivotBlock>()
            var position = 0
            while (position < n) {
                val raw = rawLapackIpiv[position]
                if (raw > 0) {
                    blocks += F64PivotedSymmetricIndefinitePivotBlock.OneByOne(position, raw - 1)
                    position++
                } else {
                    blocks += F64PivotedSymmetricIndefinitePivotBlock.TwoByTwo(position, -raw - 1)
                    position += 2
                }
            }
            return blocks
        }

    init {
        requireShape(ldl.size == n * n) { "ldl length ${ldl.size} != ${n * n}" }
        requireShape(rawLapackIpiv.size == n) { "ipiv length ${rawLapackIpiv.size} != $n" }
    }
}

/** A 0-based Bunch-Kaufman pivot block in an [F64PivotedSymmetricIndefiniteDecomposition]. */
public sealed interface F64PivotedSymmetricIndefinitePivotBlock {
    /** The first 0-based factor position in this block. */
    public val position: Int

    /** The 0-based factor position swapped with this block's LAPACK pivot position. */
    public val interchangedWith: Int

    /** The 0-based factor position LAPACK exchanges with [interchangedWith]. */
    public val interchangePosition: Int

    /** A 1×1 pivot at [position]. */
    public data class OneByOne(override val position: Int, override val interchangedWith: Int) :
        F64PivotedSymmetricIndefinitePivotBlock {
        override val interchangePosition: Int get() = position
    }

    /** A 2×2 pivot covering [position] and `position + 1`. */
    public data class TwoByTwo(override val position: Int, override val interchangedWith: Int) :
        F64PivotedSymmetricIndefinitePivotBlock {
        override val interchangePosition: Int get() = position + 1
    }
}
