package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.core.F64SparseMatrix

/** General pivoting sparse LU for unrelated matrix patterns. */
public interface F64GeneralSparseLu : Backend {
    /** Factorizes the square [a] for general solves. */
    public fun factor(a: F64SparseMatrix): F64SparseFactorization
}

/** Sparse LU optimized for repeated numeric factorizations of one structural pattern. */
public interface F64RepeatedSparseLu : Backend {
    /** Creates the initial factorization of [a]. */
    public fun factor(a: F64SparseMatrix): F64SparseFactorization

    /** Refactorizes [a], reusing [previous] when its structure is compatible. */
    public fun refactor(previous: F64SparseFactorization, a: F64SparseMatrix): F64SparseFactorization
}

/** Symmetric positive-definite sparse Cholesky factorization. */
public interface F64SparseCholesky : Backend {
    /** Factorizes the lower triangle of [a] as `L * L^T`. */
    public fun cholesky(a: F64SparseMatrix): F64SparseFactorization
}

/** Unpivoted symmetric sparse `L * D * L^T` factorization. */
public interface F64SparseLdl : Backend {
    /** Factorizes the lower triangle of [a] as `L * D * L^T`. */
    public fun ldl(a: F64SparseMatrix): F64SparseFactorization
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
 */
internal class F64SparseDecompositionRoles(
    val generalLu: F64GeneralSparseLu,
    val choleskyProvider: F64SparseCholesky,
    val ldlProvider: F64SparseLdl,
) : F64SparseDecompositions {
    override val name: String
        get() = listOf(generalLu.name, choleskyProvider.name, ldlProvider.name).distinct().joinToString("+")
    override val priority: Int get() = maxOf(generalLu.priority, choleskyProvider.priority, ldlProvider.priority)
    override val isPortable: Boolean
        get() = generalLu.isPortable && choleskyProvider.isPortable && ldlProvider.isPortable
    override val isAvailable: Boolean
        get() = generalLu.isAvailable && choleskyProvider.isAvailable && ldlProvider.isAvailable

    override fun factor(a: F64SparseMatrix): F64SparseFactorization = generalLu.factor(a)
    override fun cholesky(a: F64SparseMatrix): F64SparseFactorization = choleskyProvider.cholesky(a)
    override fun ldl(a: F64SparseMatrix): F64SparseFactorization = ldlProvider.ldl(a)
}

internal class LegacyGeneralSparseLu(private val delegate: F64SparseDecompositions) : F64GeneralSparseLu {
    override val name: String get() = delegate.name
    override val priority: Int get() = delegate.priority
    override val isPortable: Boolean get() = delegate.isPortable
    override val isAvailable: Boolean get() = delegate.isAvailable
    override fun factor(a: F64SparseMatrix): F64SparseFactorization = delegate.factor(a)
}

internal class LegacySparseCholesky(private val delegate: F64SparseDecompositions) : F64SparseCholesky {
    override val name: String get() = delegate.name
    override val priority: Int get() = delegate.priority
    override val isPortable: Boolean get() = delegate.isPortable
    override val isAvailable: Boolean get() = delegate.isAvailable
    override fun cholesky(a: F64SparseMatrix): F64SparseFactorization = delegate.cholesky(a)
}

internal class LegacySparseLdl(private val delegate: F64SparseDecompositions) : F64SparseLdl {
    override val name: String get() = delegate.name
    override val priority: Int get() = delegate.priority
    override val isPortable: Boolean get() = delegate.isPortable
    override val isAvailable: Boolean get() = delegate.isAvailable
    override fun ldl(a: F64SparseMatrix): F64SparseFactorization = delegate.ldl(a)
}
