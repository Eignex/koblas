package com.eignex.koblas.internal.operands

/** Structure of a stored matrix; unstored entries and unit diagonals are never loaded. */
internal enum class MatrixStructure {
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
