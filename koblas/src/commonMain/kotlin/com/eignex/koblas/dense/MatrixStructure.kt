package com.eignex.koblas.dense

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
}

/**
 * The triangle and diagonal a genuinely triangular operand declares.
 *
 * Only the four triangular routines use this. A symmetric destination takes [symmetricStructure] instead, even
 * though both spellings reach the same CBLAS `uplo`: a rank update's destination is stored as a symmetric
 * matrix is, and calling it triangular would declare the other half to be zeros it is not.
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

/** Symmetric bindings have no unit-diagonal flag, so they require a stored triangle and diagonal. */
internal fun requireStructured(structure: MatrixStructure, what: String) {
    val stored = structure != MatrixStructure.General &&
        structure != MatrixStructure.UnitLower &&
        structure != MatrixStructure.UnitUpper
    require(stored) { "$what requires a stored triangle with a stored diagonal" }
}

/** A triangular structure, stored or with an implicit unit diagonal. */
internal fun requireTriangular(structure: MatrixStructure, what: String) {
    val triangular = structure == MatrixStructure.TriangularLower ||
        structure == MatrixStructure.TriangularUpper ||
        structure == MatrixStructure.UnitLower ||
        structure == MatrixStructure.UnitUpper
    require(triangular) { "$what requires a triangular matrix" }
}
