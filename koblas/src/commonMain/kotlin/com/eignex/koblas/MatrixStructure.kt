@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

/** Structure of a stored matrix; unstored entries and unit diagonals are never loaded. */
public enum class MatrixStructure {
    /** Every entry is stored. */
    General,

    /** The lower triangle is mirrored. */
    SymmetricLower,

    /** The upper triangle is mirrored. */
    SymmetricUpper,

    /** Entries above the diagonal are positive zero. */
    TriangularLower,

    /** Entries below the diagonal are positive zero. */
    TriangularUpper,

    /** Lower triangular with implicit diagonal ones. */
    UnitLower,

    /** Upper triangular with implicit diagonal ones. */
    UnitUpper,
    ;

    /** Whether the stored triangle is lower. */
    internal val isLower: Boolean get() = this == SymmetricLower || this == TriangularLower || this == UnitLower

    /** Whether the diagonal is implicit. */
    internal val isUnitDiagonal: Boolean get() = this == UnitLower || this == UnitUpper
}

/**
 * Makes the stored entries match what [structure] promises, in place.
 *
 * A window's structure tells BLAS what to read; it says nothing about what the array holds. The two are the
 * same question only until that buffer reaches a call taking it as a general matrix. A factor declared
 * [MatrixStructure.UnitLower] may carry anything on its diagonal and above it, because a triangular call reads
 * neither; hand the same buffer to `syrk`, which takes its operand as general, and that residue becomes
 * arithmetic. This is how a caller makes the promise true before the hand-off.
 *
 * The triangle the structure declares unstored is filled with positive zero, and a unit diagonal is written as
 * ones. A matrix need not be square: in one wider than it is tall, every column past the last row lies
 * entirely above the diagonal and is zeroed whole.
 *
 * Only the triangular and unit structures are masked. [MatrixStructure.General] stores everything and has
 * nothing to mask; the symmetric structures mirror their stored triangle rather than zeroing the other, which
 * writes a transpose rather than zeros and is a different operation under its own name.
 */
public fun DenseMatrix.maskTo(structure: MatrixStructure) {
    val lower = when (structure) {
        MatrixStructure.TriangularLower, MatrixStructure.UnitLower -> true

        MatrixStructure.TriangularUpper, MatrixStructure.UnitUpper -> false

        MatrixStructure.General, MatrixStructure.SymmetricLower, MatrixStructure.SymmetricUpper ->
            throw IllegalArgumentException("maskTo: $structure declares nothing unstored to zero")
    }
    for (j in 0 until cols) {
        val base = j * rows
        // Masked entries form one contiguous fill per column in column-major storage.
        if (lower) {
            values.fill(0.0, base, base + minOf(j, rows))
        } else {
            values.fill(0.0, base + minOf(j + 1, rows), base + rows)
        }
    }
    if (structure.isUnitDiagonal) {
        for (d in 0 until minOf(rows, cols)) values[d + d * rows] = 1.0
    }
}
