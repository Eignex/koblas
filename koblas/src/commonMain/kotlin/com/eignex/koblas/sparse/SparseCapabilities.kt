package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi

/** General pivoting sparse LU for unrelated matrix patterns. */
public interface GeneralSparseLu : Backend {
    /** Factorizes the square [a] for general solves. */
    public fun factor(a: SparseMatrix): SparseLuFactorization
}

/** Sparse LU optimized for repeated numeric factorizations of one structural pattern. */
public interface RepeatedSparseLu : Backend {
    /**
     * Analyzes the structure of [a] for repeated numeric factorizations. The caller owns the returned
     * analysis and must close it after every factor produced from it has been closed.
     */
    public fun analyze(a: SparseMatrix): SparseLuAnalysis = RefactoringSparseLuAnalysis(this, a)

    /** Creates the initial factorization of [a]. */
    public fun factor(a: SparseMatrix): SparseLuFactorization

    /** Refactorizes [a], reusing [previous] when its structure is compatible. */
    public fun refactor(previous: SparseLuFactorization, a: SparseMatrix): SparseLuFactorization
}

/**
 * A caller-owned symbolic analysis of one immutable sparse pattern.
 *
 * [factor] creates a numeric factor for the analyzed pattern. [refactor] replaces its previous factor with a
 * numeric factor for new values and closes it when the provider returns another object. Numeric factors are
 * owned by the caller and must be closed before this analysis. Calls after [close] throw
 * [IllegalStateException]; close is idempotent. A caller sharing an analysis between threads must serialize
 * factorization and close calls.
 */
public interface SparseLuAnalysis : AutoCloseable {
    /** Creates numeric factors for [a].
     *  @throws IllegalArgumentException when [a] does not have the analyzed structure. */
    public fun factor(a: SparseMatrix): SparseLuFactorization

    /** Refactorizes [a] through this analysis, superseding [previous].
     *  @throws IllegalArgumentException when [a] does not have the analyzed structure. */
    public fun refactor(previous: SparseLuFactorization, a: SparseMatrix): SparseLuFactorization

    /** Releases this symbolic analysis. */
    override fun close()
}

/** An immutable snapshot of a validated CSC structure, without its numeric values. */
internal class SparsePattern private constructor(
    private val rows: Int,
    private val cols: Int,
    private val columnPointers: IntArray,
    private val rowIndices: IntArray,
) {
    @OptIn(UnsafeKoblasApi::class)
    private fun matches(a: SparseMatrix): Boolean = rows == a.rows && cols == a.cols &&
        columnPointers.contentEquals(a.colPtr) && rowIndices.contentEquals(a.rowIdx)

    fun requireMatch(a: SparseMatrix) {
        require(matches(a)) {
            "sparse pattern ${a.rows}x${a.cols} with ${a.nnz} entries does not match " +
                "${rows}x$cols with ${rowIndices.size} entries"
        }
    }

    companion object {
        fun of(a: SparseMatrix): SparsePattern = SparsePattern(
            a.rows,
            a.cols,
            a.copyColumnPointers(),
            a.copyRowIndices(),
        )
    }
}

/**
 * The lifecycle both analyses owe their caller: one immutable pattern, one closed flag, and the guard every
 * call passes before it reaches the numeric half.
 *
 * Written once because the promise is one promise. [what] names the analysis in the failure, which is the
 * only part the two kinds do not share.
 */
private class AnalysisGuard(a: SparseMatrix, private val what: String) {
    private val pattern: SparsePattern = SparsePattern.of(a)
    private var closed = false

    /** Rejects a call after [close], then a matrix of another pattern. */
    fun admit(a: SparseMatrix) {
        check(!closed) { "$what is closed" }
        pattern.requireMatch(a)
    }

    /** Idempotent, as the analysis contract promises. */
    fun close() {
        closed = true
    }
}

private class RefactoringSparseLuAnalysis(private val provider: RepeatedSparseLu, a: SparseMatrix) :
    SparseLuAnalysis {
    private val guard = AnalysisGuard(a, "sparse LU analysis")

    override fun factor(a: SparseMatrix): SparseLuFactorization {
        guard.admit(a)
        return provider.factor(a)
    }

    override fun refactor(previous: SparseLuFactorization, a: SparseMatrix): SparseLuFactorization {
        guard.admit(a)
        val next = provider.refactor(previous, a)
        if (next !== previous) previous.close()
        return next
    }

    override fun close() {
        guard.close()
    }
}

/**
 * A caller-owned symbolic analysis of one sparse pattern, reusable across numeric factorizations of it.
 *
 * The symmetric factorizations and the QR each begin by deriving an elimination structure from the pattern
 * alone. A caller refactorizing one structure with new values, which is what an interior-point or a
 * sequential-QP iteration does, pays for that derivation on every call unless it is held here.
 *
 * What the reuse is worth follows the fill, since the derivation costs a pass over the pattern whatever the
 * numeric half then does with it. `SparseRefactorBenchmark` measures 1.4x on a banded pattern from order 256
 * to 4096, where there is almost no fill; `SparseSymmetricHostBenchmark` measures nothing outside its error
 * bars on a pattern of one percent density, where the numeric sweep is doing far more work than the analysis.
 *
 * [factor] rejects a matrix of another pattern rather than filling the factor from the wrong bounds. Numeric
 * factors are owned by the caller and must be closed before this analysis. Calls after [close] throw
 * [IllegalStateException]; close is idempotent. A caller sharing an analysis between threads must serialize
 * its calls.
 *
 * @param F the kind of factorization this analysis produces.
 */
public interface SparseSymbolicAnalysis<out F : AutoCloseable> : AutoCloseable {
    /** Numerically factorizes [a], which must have the analyzed pattern.
     *  @throws IllegalArgumentException when [a] does not have the analyzed structure. */
    public fun factor(a: SparseMatrix): F

    /** Releases this symbolic analysis. */
    override fun close()
}

/**
 * The pattern check every [SparseSymbolicAnalysis] owes its caller, over whatever numeric half it holds.
 *
 * A provider whose analysis is nothing but the check passes its ordinary factorization as [numeric], which is
 * what a binding that keeps no reusable structure of its own does. A provider that has one closes over it.
 */
internal class PatternOnlyAnalysis<F : AutoCloseable>(a: SparseMatrix, private val numeric: (SparseMatrix) -> F) :
    SparseSymbolicAnalysis<F> {
    private val guard = AnalysisGuard(a, "sparse symbolic analysis")

    override fun factor(a: SparseMatrix): F {
        guard.admit(a)
        return numeric(a)
    }

    override fun close() {
        guard.close()
    }
}

/** Symmetric positive-definite sparse Cholesky factorization. */
public interface SparseCholesky : Backend {
    /** Factorizes the lower triangle of [a] as `L * L^T`. */
    public fun cholesky(a: SparseMatrix): SparseCholeskyFactorization

    /** Analyzes the pattern of [a] for repeated factorizations of that structure. */
    public fun analyzeCholesky(a: SparseMatrix): SparseSymbolicAnalysis<SparseCholeskyFactorization> =
        PatternOnlyAnalysis(a) { cholesky(it) }
}

/** Sparse QR factorization of a tall or square matrix, for least-squares solves. */
public interface SparseQr : Backend {
    /** Factorizes [a], which must have at least as many rows as columns, as `Q * R`. */
    public fun qr(a: SparseMatrix): SparseQrFactorization

    /** Analyzes the pattern of [a] for repeated factorizations of that structure. */
    public fun analyzeQr(a: SparseMatrix): SparseSymbolicAnalysis<SparseQrFactorization> =
        PatternOnlyAnalysis(a) { qr(it) }
}

/**
 * Numerically unpivoted sparse `L * D * L^T` factorization for quasi-definite systems.
 *
 * The ordering may reduce fill, but no numerical pivoting occurs. Use [GeneralSparseLu] for a general
 * indefinite system that needs numerical pivoting.
 */
public interface QuasiDefiniteLdl : Backend {
    /** Factorizes [a]'s lower triangle as quasi-definite `L * D * L^T`. */
    public fun quasiDefiniteLdl(a: SparseMatrix): QuasiDefiniteLdlFactorization

    /** Analyzes the pattern of [a] for repeated factorizations of that structure. */
    public fun analyzeQuasiDefiniteLdl(a: SparseMatrix): SparseSymbolicAnalysis<QuasiDefiniteLdlFactorization> =
        PatternOnlyAnalysis(a) { quasiDefiniteLdl(it) }
}

/** Sparse factorization of a simplex basis that supports column replacement. */
public interface BasisFactorizations : Backend {
    /** Factorizes [basis] for subsequent column replacements. */
    public fun factorBasis(basis: SparseMatrix): BasisFactorization
}

/**
 * Compatibility composition of independently selected sparse factorization roles.
 *
 * @property generalLu provider for ordinary sparse LU.
 * @property choleskyProvider provider for positive-definite symmetric factorization.
 * @property quasiDefiniteLdlProvider provider for quasi-definite symmetric factorization.
 * @property qrProvider provider for least-squares QR.
 */
internal class SparseDecompositionRoles(
    val generalLu: GeneralSparseLu,
    val choleskyProvider: SparseCholesky,
    val quasiDefiniteLdlProvider: QuasiDefiniteLdl,
    val qrProvider: SparseQr,
) : SparseLapack {
    override val name: String
        get() = listOf(generalLu.name, choleskyProvider.name, quasiDefiniteLdlProvider.name, qrProvider.name)
            .distinct()
            .joinToString("+")
    override val priority: Int
        get() = maxOf(
            generalLu.priority,
            choleskyProvider.priority,
            quasiDefiniteLdlProvider.priority,
            qrProvider.priority,
        )
    override val isPortable: Boolean
        get() = generalLu.isPortable && choleskyProvider.isPortable && quasiDefiniteLdlProvider.isPortable &&
            qrProvider.isPortable
    override val isAvailable: Boolean
        get() = generalLu.isAvailable && choleskyProvider.isAvailable && quasiDefiniteLdlProvider.isAvailable &&
            qrProvider.isAvailable

    override fun factor(a: SparseMatrix): SparseLuFactorization = generalLu.factor(a)
    override fun cholesky(a: SparseMatrix): SparseCholeskyFactorization = choleskyProvider.cholesky(a)
    override fun quasiDefiniteLdl(a: SparseMatrix): QuasiDefiniteLdlFactorization =
        quasiDefiniteLdlProvider.quasiDefiniteLdl(a)
    override fun qr(a: SparseMatrix): SparseQrFactorization = qrProvider.qr(a)
}
