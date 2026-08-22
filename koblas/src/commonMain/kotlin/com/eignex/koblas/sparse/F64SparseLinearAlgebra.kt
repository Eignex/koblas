package com.eignex.koblas.sparse

import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.factorization.ldl.*
import com.eignex.koblas.sparse.factorization.lu.*
import com.eignex.koblas.sparse.symbolic.*

/** Both sparse halves in one type. */
public interface F64SparseLinearAlgebra :
    F64SparseBlas,
    F64SparseLapack

/** Factorize this sparse matrix with the active backend ([koblas]), the counterpart of `F64DenseMatrix.lu`. */
public fun F64SparseMatrix.lu(equilibrate: Boolean = false, dropTolerance: Double = NO_DROP): F64SparseFactorization =
    koblas.factor(this, equilibrate, dropTolerance)

/**
 * Factorize this symmetric matrix as `L·D·Lᵀ` with the active backend ([koblas]). Indefinite by default.
 * See [F64SparseLapack.ldl].
 */
public fun F64SparseMatrix.ldl(
    policy: SparseLdlPolicy = SparseLdlPolicy.Indefinite,
    ordering: SparseOrdering = SparseOrdering.MinimumDegree,
): F64SparseFactorization = koblas.ldl(this, policy, ordering)

/**
 * Factorize this symmetric positive-definite matrix with the active backend ([koblas]). Throws on a
 * non-positive pivot. See [F64SparseLapack.cholesky].
 */
public fun F64SparseMatrix.cholesky(
    policy: SparseLdlPolicy = SparseLdlPolicy.Strict,
    ordering: SparseOrdering = SparseOrdering.MinimumDegree,
): F64SparseFactorization = koblas.cholesky(this, policy, ordering)

/**
 * Analyse this symmetric matrix's pattern with the active backend ([koblas]), for factorizing a sequence of
 * matrices that share it. See [F64SparseLapack.analyze].
 */
public fun F64SparseMatrix.analyze(ordering: SparseOrdering = SparseOrdering.MinimumDegree): SparseSymbolic =
    koblas.analyze(this, ordering)

/** `this · x`, or `thisᵀ · x` when [transpose], with the active backend ([koblas]). */
public fun F64SparseMatrix.gemv(x: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.gemv(
    this,
    x,
    transpose,
)

/**
 * Solve `op(T) · x = b` in place against this matrix's [lower] or upper triangle, with the active backend
 * ([koblas]). See [F64SparseBlas.trsv].
 */
public fun F64SparseMatrix.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trsv(this, x, lower, transpose, unitDiag)
