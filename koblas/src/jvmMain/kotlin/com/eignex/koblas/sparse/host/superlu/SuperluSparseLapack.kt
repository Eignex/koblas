package com.eignex.koblas.sparse.host.superlu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLapack
import com.eignex.koblas.sparse.factorization.lu.F64SparseLu
import com.eignex.koblas.sparse.factorization.lu.NO_DROP
import java.lang.foreign.Arena

/** SuperLU's factorization driver. */
public enum class SuperluDriver {
    /** The lean factorization and triangular-solve routines. */
    Standard,

    /** The expert driver with SuperLU-native equilibration and diagnostics. */
    Expert,
}

/** Sparse factorizations backed by a host SuperLU bridge. */
public class SuperluSparseLapack(
    /** The SuperLU routine family that factors and solves each matrix. */
    public val driver: SuperluDriver = SuperluDriver.Expert,
) : F64SparseLapack {
    override val name: String get() = BackendNames.SUPERLU

    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1

    override val isAvailable: Boolean get() = SuperluCalls.available

    override val isPortable: Boolean get() = false

    override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): F64SparseFactorization {
        requireSquare(a, "factor")
        if (!SuperluCalls.available || dropTolerance != NO_DROP || a.rows == 0 || a.nnz == 0) {
            return F64SparseLu.factorCsc(a, equilibrate, dropTolerance)
        }
        val arena = Arena.ofShared()
        val factor =
            (if (driver == SuperluDriver.Expert) SuperluCalls::factorizeExpert else SuperluCalls::factorize).invoke(
                a.rows,
                a.values.copyOf(),
                a.rowIdx,
                a.colPtr,
                equilibrate,
                arena,
            ) ?: run {
                arena.close()
                return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
            }
        return SuperluFactorization(a.rows, NOT_SINGULAR, factor, arena, SuperluCalls.fill(factor))
    }
}
