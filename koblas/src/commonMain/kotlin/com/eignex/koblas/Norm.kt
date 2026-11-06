package com.eignex.koblas

/** Norm choices shared by [Vector.norm] and [Matrix.norm]; each operation documents its supported choices. */
public enum class Norm {
    /** Sum of absolute vector entries, or the largest absolute column sum of a matrix. */
    One,

    /** Euclidean vector norm. The spectral matrix norm is not supported. */
    Two,

    /** Maximum absolute vector entry, or the largest absolute row sum of a matrix. */
    Infinity,

    /** Square root of the sum of squared matrix entries. Not supported for vectors. */
    Frobenius,

    /** Maximum absolute entry. Equivalent to [Infinity] for vectors; distinct from it for matrices. */
    MaxAbs,
}
