package com.eignex.koblas

/** The `failedAt` value of a sparse factorization that succeeded. */
public const val NOT_SINGULAR: Int = -1

/** A singular factorization whose backend cannot identify the failed pivot. */
public const val SINGULAR_POSITION_UNKNOWN: Int = -2

internal fun requireFactored(failedAt: Int, routine: String) {
    if (failedAt != NOT_SINGULAR) throw hfactorSingularFailure(failedAt, routine)
}

internal fun hfactorSingularFailure(failedAt: Int, routine: String): SingularMatrix = SingularMatrix(
    failedAt,
    buildString {
        append(routine)
        if (failedAt == SINGULAR_POSITION_UNKNOWN) {
            append(": the factorization is singular")
        } else {
            append(": the factorization is singular at pivot ").append(failedAt)
        }
        append(", so the system has no unique solution. ")
        append("Check `singular` before solving, or factor a repaired matrix.")
    },
)

internal fun requireHfactorShape(condition: Boolean, message: () -> String) {
    if (!condition) throw DimensionMismatch(message())
}

internal fun requireHfactorIndex(condition: Boolean, message: () -> String) {
    if (!condition) throw IndexOutOfBoundsException(message())
}

internal fun requireHfactorSolveShapes(rows: Int, cols: Int, b: DoubleArray, out: DoubleArray) {
    requireHfactorShape(b.size == rows) { "solve: b size ${b.size}, expected $rows" }
    requireHfactorShape(out.size == cols) { "solve: out size ${out.size}, expected $cols" }
}

internal fun requireHfactorSolveShapes(rows: Int, cols: Int, b: DenseMatrix, out: DenseMatrix) {
    requireHfactorShape(b.rows == rows) { "solve: B has ${b.rows} rows, expected $rows" }
    requireHfactorShape(out.rows == cols && out.cols == b.cols) {
        "solve: out is ${out.rows}x${out.cols}, expected ${cols}x${b.cols}"
    }
}
