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
    public fun factor(a: F64SparseMatrix): F64SparseFactorization

    /** Refactorizes [a], reusing [previous] when its structure is compatible. */
    public fun refactor(previous: F64SparseFactorization, a: F64SparseMatrix): F64SparseFactorization
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
    /** The exact CSC structure accepted by this analysis. */
    public val pattern: F64SparsePattern

    /** Creates numeric factors for [a], whose structure must match [pattern]. */
    public fun factor(a: F64SparseMatrix): F64SparseFactorization

    /** Refactorizes [a] through this analysis, superseding [previous]. */
    public fun refactor(previous: F64SparseFactorization, a: F64SparseMatrix): F64SparseFactorization

    /** Releases this symbolic analysis. */
    override fun close()
}

/** An immutable snapshot of a validated CSC structure, without its numeric values. */
public class F64SparsePattern private constructor(
    /** Rows in matrices matching this pattern. */
    public val rows: Int,
    /** Columns in matrices matching this pattern. */
    public val cols: Int,
    internal val columnPointers: IntArray,
    internal val rowIndices: IntArray,
) {
    /** Stored positions in this pattern. */
    public val nnz: Int get() = rowIndices.size

    /** A copy of the CSC column pointers retained by this pattern. */
    public fun copyColumnPointers(): IntArray = columnPointers.copyOf()

    /** A copy of the CSC row indices retained by this pattern. */
    public fun copyRowIndices(): IntArray = rowIndices.copyOf()

    /** Whether [a] has exactly this shape and ordered CSC structure. */
    @OptIn(UnsafeKoblasApi::class)
    public fun matches(a: F64SparseMatrix): Boolean = rows == a.rows && cols == a.cols &&
        columnPointers.contentEquals(a.colPtr) && rowIndices.contentEquals(a.rowIdx)

    /** Throws [IncompatibleSparsePatternException] unless [a] has exactly this structure. */
    public fun requireMatch(a: F64SparseMatrix) {
        if (!matches(a)) throw IncompatibleSparsePatternException(this, a)
    }

    override fun equals(other: Any?): Boolean = other is F64SparsePattern && rows == other.rows && cols == other.cols &&
        columnPointers.contentEquals(other.columnPointers) && rowIndices.contentEquals(other.rowIndices)

    override fun hashCode(): Int {
        var hash = 31 * rows + cols
        hash = 31 * hash + columnPointers.contentHashCode()
        return 31 * hash + rowIndices.contentHashCode()
    }

    override fun toString(): String = "F64SparsePattern(${rows}x$cols, nnz=$nnz)"

    /** Pattern factories. */
    public companion object {
        /** Copies the shape and structural arrays of [a], excluding its values. */
        public fun of(a: F64SparseMatrix): F64SparsePattern = F64SparsePattern(
            a.rows,
            a.cols,
            a.copyColumnPointers(),
            a.copyRowIndices(),
        )
    }
}

/** Raised before numeric work when a matrix does not match a symbolic analysis. */
public class IncompatibleSparsePatternException(
    /** Pattern required by the analysis. */
    public val expected: F64SparsePattern,
    /** Matrix rejected by the analysis. */
    public val actual: F64SparseMatrix,
) : IllegalArgumentException(
    "sparse pattern ${actual.rows}x${actual.cols} with ${actual.nnz} entries does not match " +
        "${expected.rows}x${expected.cols} with ${expected.nnz} entries",
)

private class RefactoringSparseLuAnalysis(private val provider: F64RepeatedSparseLu, a: F64SparseMatrix) :
    F64SparseLuAnalysis {
    override val pattern: F64SparsePattern = F64SparsePattern.of(a)
    private var closed = false

    override fun factor(a: F64SparseMatrix): F64SparseFactorization {
        checkOpen()
        pattern.requireMatch(a)
        return provider.factor(a)
    }

    override fun refactor(previous: F64SparseFactorization, a: F64SparseMatrix): F64SparseFactorization {
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

/** Unpivoted symmetric sparse `L * D * L^T` factorization. */
public interface F64SparseLdl : Backend {
    /** Factorizes the lower triangle of [a] as `L * D * L^T`. */
    public fun ldl(a: F64SparseMatrix): F64SparseLdlFactorization
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
 * @property ldlProvider provider for quasi-definite symmetric factorization.
 * @property qrProvider provider for least-squares QR.
 */
internal class F64SparseDecompositionRoles(
    val generalLu: F64GeneralSparseLu,
    val choleskyProvider: F64SparseCholesky,
    val ldlProvider: F64SparseLdl,
    val qrProvider: F64SparseQr,
) : F64SparseDecompositions {
    override val name: String
        get() = listOf(generalLu.name, choleskyProvider.name, ldlProvider.name, qrProvider.name)
            .distinct()
            .joinToString("+")
    override val priority: Int
        get() = maxOf(generalLu.priority, choleskyProvider.priority, ldlProvider.priority, qrProvider.priority)
    override val isPortable: Boolean
        get() = generalLu.isPortable && choleskyProvider.isPortable && ldlProvider.isPortable &&
            qrProvider.isPortable
    override val isAvailable: Boolean
        get() = generalLu.isAvailable && choleskyProvider.isAvailable && ldlProvider.isAvailable &&
            qrProvider.isAvailable

    override fun factor(a: F64SparseMatrix): F64SparseLuFactorization = generalLu.factor(a)
    override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization = choleskyProvider.cholesky(a)
    override fun ldl(a: F64SparseMatrix): F64SparseLdlFactorization = ldlProvider.ldl(a)
    override fun qr(a: F64SparseMatrix): F64SparseQrFactorization = qrProvider.qr(a)
}
