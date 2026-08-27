package com.eignex.koblas.dense

/** Whether a BLAS operand is read in its stored orientation or transposed. */
public enum class Transpose {
    /** Read the operand without transposing it. */
    NO_TRANSPOSE,

    /** Read the transpose of the operand without materializing a copy. */
    TRANSPOSE,
}

/** Whether a triangular matrix stores an ordinary or implicit unit diagonal. */
public enum class Diag {
    /** Read and use the stored diagonal. */
    NON_UNIT,

    /** Treat every diagonal entry as one without reading it. */
    UNIT,
}

/** Side on which a distinguished symmetric or triangular operand acts. */
public enum class Side {
    /** The distinguished operand acts on the left. */
    LEFT,

    /** The distinguished operand acts on the right. */
    RIGHT,
}

internal val Transpose.enabled: Boolean get() = this == Transpose.TRANSPOSE

internal val Diag.isUnit: Boolean get() = this == Diag.UNIT

internal val Side.isRight: Boolean get() = this == Side.RIGHT

internal fun Uplo.lowerTriangle(operation: String): Boolean = when (this) {
    Uplo.LOWER -> true
    Uplo.UPPER -> false
    Uplo.FULL -> throw IllegalArgumentException("$operation requires Uplo.LOWER or Uplo.UPPER, not Uplo.FULL")
}
