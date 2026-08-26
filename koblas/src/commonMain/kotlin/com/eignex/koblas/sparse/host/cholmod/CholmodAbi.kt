package com.eignex.koblas.sparse.host.cholmod

/**
 * CHOLMOD's ABI as offsets and codes, shared by the bindings that reach it.
 *
 * CHOLMOD takes its operands as structs rather than as loose arrays, so unlike the KLU and UMFPACK bindings
 * this one has to lay them out itself. `cholmod_sparse` and `cholmod_dense` are documented public structs
 * and are laid out in full; `cholmod_factor` is read only for the three fields below, and `cholmod_common`
 * is never read at all.
 */
internal val CHOLMOD_SONAMES: List<String> = listOf(
    "libcholmod.so.5",
    "libcholmod.so.4",
    "libcholmod.so.3",
    "libcholmod.so",
    "libcholmod.5.dylib",
    "libcholmod.dylib",
    "/opt/homebrew/opt/suite-sparse/lib/libcholmod.dylib", // Homebrew is keg-only
    "/opt/homebrew/opt/suite-sparse/lib/libcholmod.5.dylib",
    "/usr/local/opt/suite-sparse/lib/libcholmod.dylib",
)

/**
 * Room for a `cholmod_common`, which is started and handed to every call. Deliberately larger than any
 * release has needed, since the size is what the binding has to get right and the struct grows between
 * versions where its leading fields do not move.
 */
internal const val CHOLMOD_COMMON_BYTES = 16384L

/**
 * `final_ll`. CHOLMOD leaves a simplicial factor as `L·D·Lᵀ` by default, and that factorization exists for
 * an indefinite matrix: it comes back reporting success with a negative entry in `D`, where this seam
 * promises `A = L·Lᵀ` and a raise at the first non-positive pivot. Asking for `L·Lᵀ` is what makes CHOLMOD
 * stop at that pivot and name it, which is the column the portable Cholesky names too.
 */
internal const val CHOLMOD_COMMON_FINAL_LL = 60L

/** `print`, the print level, set to nothing so a refused matrix does not write to stderr behind a caller. */
internal const val CHOLMOD_COMMON_PRINT = 144L

internal const val CHOLMOD_SPARSE_BYTES = 88L
internal const val CHOLMOD_SPARSE_NROW = 0L
internal const val CHOLMOD_SPARSE_NCOL = 8L
internal const val CHOLMOD_SPARSE_NZMAX = 16L
internal const val CHOLMOD_SPARSE_P = 24L
internal const val CHOLMOD_SPARSE_I = 32L
internal const val CHOLMOD_SPARSE_NZ = 40L
internal const val CHOLMOD_SPARSE_X = 48L
internal const val CHOLMOD_SPARSE_Z = 56L
internal const val CHOLMOD_SPARSE_STYPE = 64L
internal const val CHOLMOD_SPARSE_ITYPE = 68L
internal const val CHOLMOD_SPARSE_XTYPE = 72L
internal const val CHOLMOD_SPARSE_DTYPE = 76L
internal const val CHOLMOD_SPARSE_SORTED = 80L
internal const val CHOLMOD_SPARSE_PACKED = 84L

internal const val CHOLMOD_DENSE_BYTES = 56L
internal const val CHOLMOD_DENSE_NROW = 0L
internal const val CHOLMOD_DENSE_NCOL = 8L
internal const val CHOLMOD_DENSE_NZMAX = 16L
internal const val CHOLMOD_DENSE_D = 24L
internal const val CHOLMOD_DENSE_X = 32L
internal const val CHOLMOD_DENSE_Z = 40L
internal const val CHOLMOD_DENSE_XTYPE = 48L
internal const val CHOLMOD_DENSE_DTYPE = 52L

/** Enough of a `cholmod_factor` to read the three fields below, which are the only ones this reads. */
internal const val CHOLMOD_FACTOR_BYTES = 208L
internal const val CHOLMOD_FACTOR_N = 0L
internal const val CHOLMOD_FACTOR_MINOR = 8L
internal const val CHOLMOD_FACTOR_NZMAX = 40L
internal const val CHOLMOD_FACTOR_IS_LL = 180L

/** The lower triangle is stored and the strict upper one is ignored, which is what this seam's Cholesky reads. */
internal const val CHOLMOD_STYPE_LOWER = -1

/** Every stored entry is the matrix, which is what a routine that is not reading a symmetric half wants. */
internal const val CHOLMOD_STYPE_GENERAL = 0

internal const val CHOLMOD_INT = 0
internal const val CHOLMOD_REAL = 1
internal const val CHOLMOD_DOUBLE = 0

/** `cholmod_solve` system code for `A x = b`. */
internal const val CHOLMOD_A = 0

internal const val CHOLMOD_TRUE = 1
