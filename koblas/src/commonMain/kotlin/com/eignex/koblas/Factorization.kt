package com.eignex.koblas

/**
 * The [com.eignex.koblas.dense.LuDecomposition.failedAt] /
 * [com.eignex.koblas.sparse.SparseFactorization.failedAt] value of a factorization that succeeded.
 *
 * Every factorization koblas produces, dense or sparse, reports singularity the same way: the position of
 * the pivot that had no acceptable candidate, or this when there wasn't one. Deriving `singular` from it
 * rather than carrying a separate flag means the two cannot disagree - which they could when the flag was
 * the source of truth and each backend set it independently.
 *
 * It lives in the root package because both the dense and the sparse factorization types need it and
 * neither owns the other. Whether they should share a `Factorization` interface outright is a separate
 * question, deliberately still open: the sparse factors are opaque handles while the dense ones are a
 * shared LAPACK format, and the two have different reasons to exist.
 */
public const val NOT_SINGULAR: Int = -1

/**
 * A `failedAt` meaning "singular, but this backend cannot say where".
 *
 * koblas's own factorizations always know the position: they pick pivots themselves, so the step that ran out
 * of candidates is exactly where they stopped. A host solver need not tell you. UMFPACK reports
 * `UMFPACK_WARNING_singular_matrix` and, in `Info`, how many zero pivots there were — but not which, and
 * recovering it would mean extracting `U` to hunt for a zero diagonal, which is absurd for a diagnostic.
 *
 * So this exists rather than having a host backend invent a position or, worse, report [NOT_SINGULAR] for a
 * matrix it just called singular. `singular` stays true because the value is not [NOT_SINGULAR], and a caller
 * that switches on the position has one more case to handle honestly.
 */
public const val SINGULAR_POSITION_UNKNOWN: Int = -2

/**
 * Translates a LAPACK `info` return into a `failedAt` position.
 *
 * A positive `info` from `dgetrf` or `dsytrf` is the 1-based index of the pivot that came out exactly zero,
 * so it becomes that index 0-based; anything else means the factorization succeeded. Public because any
 * LAPACK-backed [com.eignex.koblas.dense.Lapack] implementation needs the same translation, including ones
 * outside this repository.
 *
 * Negative `info` is an illegal-argument report rather than a singularity, and callers check for it
 * separately — it maps to [NOT_SINGULAR] here so a backend that forgot the check reports "not singular"
 * rather than a nonsense position.
 */
public fun lapackFailedAt(info: Int): Int = if (info > 0) info - 1 else NOT_SINGULAR

/**
 * Rejects a solve against a singular factorization, naming the pivot that failed.
 *
 * One contract for the whole library: a solve against a factorization that has no inverse throws rather
 * than producing infinities. The sparse side already refuses, and so does dense `invert`; the dense solves
 * divided by a zero pivot and returned `Inf`/`NaN`, so a caller who did not consult
 * [com.eignex.koblas.dense.LuDecomposition.singular] carried the garbage onward instead of finding out
 * where it came from.
 *
 * Shared rather than repeated per backend so the message and the reported position cannot drift; every
 * `solveInto` calls it before touching the factors. A field read and a branch, against an `O(n²)` solve.
 *
 * @param failedAt the position from the factorization, or [NOT_SINGULAR] when it succeeded.
 * @param routine the routine to name in the message.
 */
internal fun requireFactored(failedAt: Int, routine: String) {
    if (failedAt != NOT_SINGULAR) throw singularFailure(failedAt, routine)
}

/**
 * The failure [requireFactored] throws, for the one caller that knows it is singular without checking:
 * `SingularSparseFactorization` exists only for that case, so it throws this directly rather than
 * testing a field it already knows the value of. Shared so the wording stays identical either way.
 */
internal fun singularFailure(failedAt: Int, routine: String): SingularMatrix = SingularMatrix(
    failedAt,
    buildString {
        append(routine)
        // A host backend may report singularity without a position; saying "pivot -2" would be worse
        // than saying nothing, so the two cases are worded apart.
        if (failedAt == SINGULAR_POSITION_UNKNOWN) {
            append(": the factorization is singular")
        } else {
            append(": the factorization is singular at pivot ").append(failedAt)
        }
        append(", so the system has no unique solution. ")
        append("Check `singular` before solving, or factor a repaired matrix.")
    },
)
