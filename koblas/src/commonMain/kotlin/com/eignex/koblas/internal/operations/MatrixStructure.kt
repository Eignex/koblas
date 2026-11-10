package com.eignex.koblas.internal.operations

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

/**
 * Structure for triangular operands only. Symmetric destinations use [symmetricStructure]:
 * their unstored half is mirrored, whereas a triangular operand declares it zero.
 */
internal fun triangle(lower: Boolean, unitDiag: Boolean): MatrixStructure = when {
    unitDiag && lower -> MatrixStructure.UnitLower
    unitDiag -> MatrixStructure.UnitUpper
    lower -> MatrixStructure.TriangularLower
    else -> MatrixStructure.TriangularUpper
}

/** The stored triangle a symmetric operand declares, shared with the callers that state it themselves. */
internal fun symmetricStructure(lower: Boolean): MatrixStructure =
    if (lower) MatrixStructure.SymmetricLower else MatrixStructure.SymmetricUpper
