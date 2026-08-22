package com.eignex.koblas.sparse.host.superlu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLapack
import com.eignex.koblas.sparse.factorization.ldl.SparseLdlPolicy
import com.eignex.koblas.sparse.factorization.lu.F64SparseLu
import com.eignex.koblas.sparse.factorization.lu.NO_DROP
import com.eignex.koblas.sparse.symbolic.SparseOrdering
import com.eignex.koblas.sparse.symbolic.SparseSymbolic
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_INT

/** Sparse factorizations backed by a host SuperLU bridge. */
public class SuperluSparseLapack : F64SparseLapack {
    override val name: String get() = BackendNames.SUPERLU

    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1

    override val isAvailable: Boolean get() = SuperluCalls.available

    override val isPortable: Boolean get() = false

    override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): F64SparseFactorization {
        requireSquare(a, "factor")
        if (!SuperluCalls.available || equilibrate || dropTolerance != NO_DROP || a.rows == 0 || a.nnz == 0) {
            return F64SparseLu.factorCsc(a, equilibrate, dropTolerance)
        }
        Arena.ofConfined().use { scratch ->
            val status = scratch.allocate(JAVA_INT)
            val lnz = scratch.allocate(JAVA_INT)
            val unz = scratch.allocate(JAVA_INT)
            val handle = SuperluCalls.factorize(
                a.rows,
                MemorySegment.ofArray(a.values),
                MemorySegment.ofArray(a.rowIdx),
                MemorySegment.ofArray(a.colPtr),
                status,
                lnz,
                unz,
            )
            return when (status.get(JAVA_INT, 0)) {
                0 -> SuperluFactorization(a.rows, NOT_SINGULAR, handle, lnz.get(JAVA_INT, 0) + unz.get(JAVA_INT, 0))
                1 -> F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
                else -> F64SparseLu.factorCsc(a)
            }
        }
    }

    override fun analyze(a: F64SparseMatrix, ordering: SparseOrdering): SparseSymbolic =
        F64ReferenceSparseLinearAlgebra.analyze(a, ordering)

    override fun ldl(a: F64SparseMatrix, policy: SparseLdlPolicy, ordering: SparseOrdering): F64SparseFactorization =
        F64ReferenceSparseLinearAlgebra.ldl(a, policy, ordering)
}
