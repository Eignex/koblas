@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.MatrixStructure

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
        // Each column's masked entries are one contiguous run in column-major storage, so this is one fill
        // per column rather than an indexed walk.
        if (lower) {
            data.fill(0.0, base, base + minOf(j, rows))
        } else {
            data.fill(0.0, base + minOf(j + 1, rows), base + rows)
        }
    }
    if (structure == MatrixStructure.UnitLower || structure == MatrixStructure.UnitUpper) {
        for (d in 0 until minOf(rows, cols)) data[d + d * rows] = 1.0
    }
}
