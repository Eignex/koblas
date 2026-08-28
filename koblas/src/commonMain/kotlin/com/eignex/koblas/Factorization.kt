package com.eignex.koblas

/** The `failedAt` value of a factorization that succeeded, dense or sparse. */
public const val NOT_SINGULAR: Int = -1

/**
 * A `failedAt` meaning "singular, but this backend cannot say where". Exists so a host solver like UMFPACK,
 * which counts zero pivots without locating them, need not invent a position or report [NOT_SINGULAR].
 */
public const val SINGULAR_POSITION_UNKNOWN: Int = -2

/**
 * Translates a LAPACK `info` return into a `failedAt` position. A positive `info` is the 1-based pivot
 * index; a negative one is an illegal-argument report and maps to [NOT_SINGULAR].
 */
internal fun lapackFailedAt(info: Int): Int = if (info > 0) info - 1 else NOT_SINGULAR

/**
 * Rejects a solve against a singular factorization, naming the pivot that failed.
 *
 * @param failedAt the position from the factorization, or [NOT_SINGULAR] when it succeeded.
 * @param routine the routine to name in the message.
 */
internal fun requireFactored(failedAt: Int, routine: String) {
    if (failedAt != NOT_SINGULAR) throw singularFailure(failedAt, routine)
}

/** The failure [requireFactored] throws, exposed for the caller that already knows it is singular. */
internal fun singularFailure(failedAt: Int, routine: String): SingularMatrix = SingularMatrix(
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
