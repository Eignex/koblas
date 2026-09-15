package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow

/**
 * How each matrix operand of a call actually reaches BLAS, once the call's layout is settled.
 *
 * A CBLAS call carries one layout for all of its matrices, so an operand whose own addressing disagrees with
 * it has to reconcile somehow. Most can: a transpose flag turns a column-major block into the row-major
 * reading of its transpose at no cost, and a symmetric or triangular operand reconciles by flipping which
 * triangle it names. Two cannot, because their routine gives them no flag of their own: the `B` of `symm` and
 * the `B` of `syr2k`, which shares one transpose flag with `A`. Those are copied.
 *
 * The layout is chosen so that a copy is only ever needed under the column-major one. A call runs row-major
 * only when every one of its matrices is already row-major, and then nothing needs reconciling; anything else
 * runs column-major, which is what a staged block is packed as. Without that rule a packed block would be
 * handed to a row-major call and read as its own transpose.
 *
 * Execution and [VendorBlas.routeOf] both read this, so a route cannot claim direct addressing for a call that
 * copies.
 */
internal fun effectiveAddressing(operation: VendorOperation, matrices: List<MatrixWindow>): List<Addressing> {
    val layout = layoutOf(matrices)
    val addressing = matrices.mapIndexed { index, window ->
        addressingUnder(window, layout, absorbsDisagreement(operation, index))
    }
    return reconcileSharedFlag(operation, addressing)
}

/**
 * Puts `syr2k`'s two inputs on one presentation, because one transpose flag describes both.
 *
 * The flag is read off `A`, so `B` is not reconciled by agreeing with the call's layout; it has to present
 * itself exactly as `A` does. Agreeing with the layout is enough for every other operand, which is why the
 * general rule cannot settle this one: a column-major `B` under a column-major call looks reconciled and is
 * read as its own transpose when `A` carried a transpose flag with it. Where the two disagree, whichever
 * presents row-major is packed, so both arrive column-major and the shared flag describes both.
 */
private fun reconcileSharedFlag(operation: VendorOperation, addressing: List<Addressing>): List<Addressing> {
    if (operation != VendorOperation.Syr2k || addressing.size < 2) return addressing
    if (presentedLayout(addressing[0]) == presentedLayout(addressing[1])) return addressing
    return addressing.mapIndexed { index, value ->
        if (index < 2 && value == Addressing.RowMajor) Addressing.Staged else value
    }
}

/**
 * The layout a call runs under: row-major only when every matrix is already row-major, column-major otherwise.
 */
internal fun layoutOf(matrices: List<MatrixWindow>): Int =
    if (matrices.isNotEmpty() && matrices.all { addressingOf(it) == Addressing.RowMajor }) {
        Cblas.ROW_MAJOR
    } else {
        Cblas.COL_MAJOR
    }

/** The addressing one operand ends up with under [layout], copying only when it cannot reconcile. */
internal fun addressingUnder(window: MatrixWindow, layout: Int, absorbs: Boolean): Addressing {
    val natural = addressingOf(window)
    if (natural == Addressing.Staged) return Addressing.Staged
    if (presentedLayout(natural) == layout) return natural
    return if (absorbs) natural else Addressing.Staged
}

/**
 * Whether the operand at [index] can reconcile a layout disagreement without being copied.
 *
 * `symm`'s `B` has no flag at all, so it is copied whenever it does not already present the call's layout.
 * Everything else carries its own transpose or triangle flag, and an output matrix never disagrees because it
 * is what the layout was chosen from. `syr2k`'s `B` has a flag but shares it with `A`; that is settled by
 * [reconcileSharedFlag] afterwards rather than here, because agreeing with the layout does not settle it.
 */
private fun absorbsDisagreement(operation: VendorOperation, index: Int): Boolean =
    operation != VendorOperation.Symm || index != 1
