package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64SparseMatrix

/** General pivoting sparse LU for unrelated matrix patterns. */
public interface F64GeneralSparseLu : Backend {
    /** Factorizes the square [a] for general solves. */
    public fun factor(a: F64SparseMatrix): F64SparseLuFactorization
}

/** Sparse LU optimized for repeated numeric factorizations of one structural pattern. */
public interface F64RepeatedSparseLu : Backend {
    /**
     * Analyzes the structure of [a] for repeated numeric factorizations. The caller owns the returned
     * analysis and must close it after every factor produced from it has been closed.
     */
    public fun analyze(a: F64SparseMatrix): F64SparseLuAnalysis = RefactoringSparseLuAnalysis(this, a)

    /** Creates the initial factorization of [a]. */
    public fun factor(a: F64SparseMatrix): F64SparseLuFactorization

    /** Refactorizes [a], reusing [previous] when its structure is compatible. */
    public fun refactor(previous: F64SparseLuFactorization, a: F64SparseMatrix): F64SparseLuFactorization
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
public interface F64SparseLuAnalysis : AutoCloseable {
    /** Creates numeric factors for [a].
     *  @throws IllegalArgumentException when [a] does not have the analyzed structure. */
    public fun factor(a: F64SparseMatrix): F64SparseLuFactorization

    /** Refactorizes [a] through this analysis, superseding [previous].
     *  @throws IllegalArgumentException when [a] does not have the analyzed structure. */
    public fun refactor(previous: F64SparseLuFactorization, a: F64SparseMatrix): F64SparseLuFactorization

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
    private fun matches(a: F64SparseMatrix): Boolean = rows == a.rows && cols == a.cols &&
        columnPointers.contentEquals(a.colPtr) && rowIndices.contentEquals(a.rowIdx)

    fun requireMatch(a: F64SparseMatrix) {
        require(matches(a)) {
            "sparse pattern ${a.rows}x${a.cols} with ${a.nnz} entries does not match " +
                "${rows}x$cols with ${rowIndices.size} entries"
        }
    }

    companion object {
        fun of(a: F64SparseMatrix): SparsePattern = SparsePattern(
            a.rows,
            a.cols,
            a.copyColumnPointers(),
            a.copyRowIndices(),
        )
    }
}

private class RefactoringSparseLuAnalysis(private val provider: F64RepeatedSparseLu, a: F64SparseMatrix) :
    F64SparseLuAnalysis {
    private val pattern: SparsePattern = SparsePattern.of(a)
    private var closed = false

    override fun factor(a: F64SparseMatrix): F64SparseLuFactorization {
        checkOpen()
        pattern.requireMatch(a)
        return provider.factor(a)
    }

    override fun refactor(previous: F64SparseLuFactorization, a: F64SparseMatrix): F64SparseLuFactorization {
        checkOpen()
        pattern.requireMatch(a)
        val next = provider.refactor(previous, a)
        if (next !== previous) previous.close()
        return next
    }

    override fun close() {
        closed = true
    }

    private fun checkOpen() {
        check(!closed) { "sparse LU analysis is closed" }
    }
}

/** Symmetric positive-definite sparse Cholesky factorization. */
public interface F64SparseCholesky : Backend {
    /** Factorizes the lower triangle of [a] as `L * L^T`. */
    public fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization
}

/** Sparse QR factorization of a tall or square matrix, for least-squares solves. */
public interface F64SparseQr : Backend {
    /** Factorizes [a], which must have at least as many rows as columns, as `Q * R`. */
    public fun qr(a: F64SparseMatrix): F64SparseQrFactorization
}

/**
 * Numerically unpivoted sparse `L * D * L^T` factorization for quasi-definite systems.
 *
 * The ordering may reduce fill, but no numerical pivoting occurs. Use [F64GeneralSparseLu] for a general
 * indefinite system that needs numerical pivoting.
 */
public interface F64QuasiDefiniteLdl : Backend {
    /** Factorizes [a]'s lower triangle as quasi-definite `L * D * L^T`. */
    public fun quasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization
}

/** Sparse factorization of a simplex basis that supports column replacement. */
public interface F64BasisFactorizations : Backend {
    /** Factorizes [basis] for subsequent column replacements. */
    public fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization
}

/**
 * Compatibility composition of independently selected sparse factorization roles.
 *
 * @property generalLu provider for ordinary sparse LU.
 * @property choleskyProvider provider for positive-definite symmetric factorization.
 * @property quasiDefiniteLdlProvider provider for quasi-definite symmetric factorization.
 * @property qrProvider provider for least-squares QR.
 */
internal class F64SparseDecompositionRoles(
    val generalLu: F64GeneralSparseLu,
    val choleskyProvider: F64SparseCholesky,
    val quasiDefiniteLdlProvider: F64QuasiDefiniteLdl,
    val qrProvider: F64SparseQr,
) : F64SparseDecompositions {
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

    override fun factor(a: F64SparseMatrix): F64SparseLuFactorization = generalLu.factor(a)
    override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization = choleskyProvider.cholesky(a)
    override fun quasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization =
        quasiDefiniteLdlProvider.quasiDefiniteLdl(a)
    override fun qr(a: F64SparseMatrix): F64SparseQrFactorization = qrProvider.qr(a)
}
