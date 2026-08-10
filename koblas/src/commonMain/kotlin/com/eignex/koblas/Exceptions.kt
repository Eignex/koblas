package com.eignex.koblas

/**
 * What koblas throws when a routine cannot do what was asked.
 *
 * Every one of these is an [IllegalArgumentException], which is what koblas threw before there were types
 * for them, so an existing `catch (e: IllegalArgumentException)` keeps working and nothing has to change
 * to keep compiling. The types are here for the callers who want to tell the cases apart.
 *
 * The distinction that matters is between a shape that was wrong when the call was made and a matrix that
 * turned out to be numerically unusable. [DimensionMismatch] is a programming error: the caller passed
 * operands that do not fit, and the fix is in the code. [SingularMatrix] and [NotPositiveDefinite] are
 * properties of the data, discovered partway through a factorization, and a caller who knows their problem
 * can often act on them — regularize and retry, fall back from Cholesky to `ldl`, drop to a least-squares
 * solve. Catching `IllegalArgumentException` cannot separate the second kind from the first, so the
 * recovery path also swallowed genuine bugs.
 *
 * Sealed, so the set is closed and a `when` over it is exhaustive. Any backend may throw them — the
 * constructors are public, and a third-party [com.eignex.koblas.dense.Lapack] should report a singular
 * factorization the same way koblas's own does.
 *
 * Not every failure has a type. Malformed CSC arrays, a negative buffer size, an index outside a matrix:
 * those stay plain `require`, because there is nothing to recover from and no second call that would
 * succeed.
 */
public sealed class KoblasException(message: String) : IllegalArgumentException(message)

/**
 * Operands whose shapes do not fit the routine: a vector of the wrong length, a non-square matrix where a
 * square one is required, a product whose inner dimensions disagree.
 *
 * The common case, and the one a caller almost never recovers from — it means the call site is wrong.
 * Typed anyway so that a `catch` meant for a numerical failure does not silently absorb it.
 */
public class DimensionMismatch(message: String) : KoblasException(message)

/**
 * A factorization met an exactly zero pivot, so the matrix it came from has no inverse.
 *
 * @property position the 0-based pivot index where the factorization failed, matching
 *   [com.eignex.koblas.dense.LuDecomposition.failedAt] and LAPACK's positive `info` less one.
 * @param message what failed, naming the routine and the position.
 */
public class SingularMatrix(public val position: Int, message: String) : KoblasException(message)

/**
 * A matrix asserted to be positive definite turned out not to be: a Cholesky or a strict `L·D·Lᵀ` met a
 * pivot that was zero, negative or `NaN`.
 *
 * The recoverable one, and the reason these types exist. An estimate that has drifted slightly indefinite
 * is a normal thing for a covariance or a Hessian to do, and the caller's options — regularize the pivot,
 * factor as symmetric indefinite instead — are decisions only they can make. Catching this is how they
 * make it after the fact rather than having to predict it with a policy up front.
 *
 * @property position the 0-based column whose pivot failed.
 * @property pivot the offending diagonal value.
 * @param message what failed, naming the position, the pivot and the policy that would continue past it.
 */
public class NotPositiveDefinite(public val position: Int, public val pivot: Double, message: String) :
    KoblasException(
        message,
    )

/**
 * `require` for a shape check: throws [DimensionMismatch] rather than a bare `IllegalArgumentException`.
 *
 * Same shape as `kotlin.require` so a call site converts by its name alone, and inline so the message is
 * still only built when the check fails.
 */
internal inline fun requireShape(condition: Boolean, message: () -> String) {
    if (!condition) throw DimensionMismatch(message())
}
