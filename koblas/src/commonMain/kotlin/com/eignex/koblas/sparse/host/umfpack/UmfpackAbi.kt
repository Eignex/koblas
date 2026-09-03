package com.eignex.koblas.sparse.host.umfpack

// UMFPACK's ABI constants and the names its library goes by, shared by the two bindings. The JVM binding
// and the native one have no common source set below commonMain, so this is the only place both can read.
// Only plain values live here, since the calls themselves need each binding's own FFI.

/** The sys selector for `Ax = b`. */
internal const val SYS_A = 0

/** The sys selector for `Aᵀx = b`, which for a real matrix is the plain transpose. */
internal const val SYS_AT = 1

/** Info array length, UMFPACK_INFO. */
internal const val INFO = 90

/** Control array length, UMFPACK_CONTROL. */
internal const val CONTROL = 20

/** Control index of the iterative-refinement step count, UMFPACK_IRSTEP. UMFPACK defaults it to 2. */
internal const val IRSTEP = 7

/** Control index of the threshold-pivoting tolerance, UMFPACK_PIVOT_TOLERANCE. */
internal const val PIVOT_TOLERANCE = 3

/** Control index selecting native row scaling. */
internal const val SCALE = 16

/** Info index UMFPACK_LNZ, nonzeros in L with the diagonal included. */
internal const val INFO_LNZ = 43

/** Info index UMFPACK_UNZ, nonzeros in U with the diagonal included. */
internal const val INFO_UNZ = 44

/** Info index UMFPACK_RCOND, the cheap reciprocal pivot-condition estimate. */
internal const val INFO_RCOND = 67

/** Success. */
internal const val OK = 0

/** UMFPACK_WARNING_singular_matrix, a factorization was produced but the matrix is singular. */
internal const val WARNING_SINGULAR = 1

/** The symbol whose presence stands for a usable UMFPACK. */
internal const val KEY_SYMBOL = "umfpack_di_symbolic"

/** The SuiteSparse UMFPACK ABI 5 and 6 names koblas supports. */
internal val UMFPACK_SONAMES = listOf(
    "libumfpack.so.6",
    "libumfpack.so.5",
    "libumfpack.dylib",
    "/opt/homebrew/opt/suite-sparse/lib/libumfpack.dylib", // Homebrew is keg-only
    "/usr/local/opt/suite-sparse/lib/libumfpack.dylib",
)
