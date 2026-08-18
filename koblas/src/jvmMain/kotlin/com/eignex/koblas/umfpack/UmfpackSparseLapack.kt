package com.eignex.koblas.umfpack

import com.eignex.koblas.BackendNames
import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLapack
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.NO_DROP
import com.eignex.koblas.sparse.SparseLdlPolicy
import com.eignex.koblas.sparse.SparseOrdering
import com.eignex.koblas.sparse.SparseSymbolic
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE

/** Sparse factorizations backed by a host UMFPACK. */
public class UmfpackSparseLapack : F64SparseLapack {
    override val name: String get() = BackendNames.UMFPACK

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** UMFPACK ships separately from OpenBLAS, so a host can have one without the other. */
    override val isAvailable: Boolean get() = UmfpackCalls.available

    /** Stated rather than delegated: the delegate is portable and this calls out to UMFPACK. */
    override val isPortable: Boolean get() = false

    /**
     * Factorizes [a], symbolic analysis then numeric factorization. [equilibrate] and [dropTolerance] fall
     * back to the portable path, since UMFPACK offers neither koblas's scaling nor a drop threshold.
     */
    override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): F64SparseFactorization {
        requireSquare(a, "factor")
        if (!UmfpackCalls.available) return F64SparseLu.factorCsc(a, equilibrate, dropTolerance)
        // Any tolerance but the default goes to the portable path, which owns validating the rest.
        if (equilibrate || dropTolerance != NO_DROP) return F64SparseLu.factorCsc(a, equilibrate, dropTolerance)
        // Nothing to factor, and the portable path reports the pivot position where UMFPACK reports only
        // that it failed, so both bindings route this the same way.
        if (a.rows == 0 || a.nnz == 0) return F64SparseLu.factorCsc(a)

        val colPtr = MemorySegment.ofArray(a.colPtr)
        val rowIdx = MemorySegment.ofArray(a.rowIdx)
        val values = MemorySegment.ofArray(a.values)

        // A confined arena for the symbolic analysis, which is scratch: Numeric keeps what it needs.
        Arena.ofConfined().use { scratch ->
            val symbolicHolder = scratch.allocate(ADDRESS)
            val info = scratch.allocate(JAVA_DOUBLE, INFO.toLong())
            val symbolicStatus = UmfpackCalls.symbolic(a.rows, colPtr, rowIdx, values, symbolicHolder, info)
            if (symbolicStatus != OK) {
                // A rejected pattern goes to the portable path.
                return F64SparseLu.factorCsc(a)
            }

            // The numeric factors outlive this call, so their holder gets an arena the factorization owns.
            val arena = Arena.ofShared()
            val numericHolder = arena.allocate(ADDRESS)
            val numericStatus = UmfpackCalls.numeric(
                colPtr,
                rowIdx,
                values,
                symbolicHolder.get(ADDRESS, 0),
                numericHolder,
                info,
            )
            UmfpackCalls.freeSymbolic(symbolicHolder)

            if (numericStatus != OK && numericStatus != WARNING_SINGULAR) {
                // Freed rather than only closing the arena. UMFPACK nulls the handle on an error return, so
                // this is a no-op today, but relying on that is what left the native binding freeing an
                // uninitialized holder, and the arena owns the holder rather than the factors.
                UmfpackCalls.freeNumeric(numericHolder)
                arena.close()
                return F64SparseLu.factorCsc(a)
            }
            if (numericStatus == WARNING_SINGULAR) {
                // The partial factors UMFPACK hands back cannot be solved against, so they are released and
                // the canonical singular result keeps `nnz == 0` true of every singular factorization.
                UmfpackCalls.freeNumeric(numericHolder)
                arena.close()
                return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
            }
            val handles = UmfpackFactorization.Handles(arena, numericHolder)
            val factorization = UmfpackFactorization(a, NOT_SINGULAR, handles)
            val (lnz, unz) = UmfpackFactorization.fillOf(info)
            factorization.lnz = lnz
            factorization.unz = unz
            return factorization
        }
    }

    // UMFPACK provides no symmetric factorization, so these run the portable versions. Forwarded explicitly
    // rather than by class delegation, which would route the convenience overloads to the portable factor
    // instead of this one, since a delegated member calls back into the delegate.
    override fun analyze(a: F64SparseMatrix, ordering: SparseOrdering): SparseSymbolic =
        F64ReferenceSparseLinearAlgebra.analyze(a, ordering)

    override fun ldl(a: F64SparseMatrix, policy: SparseLdlPolicy, ordering: SparseOrdering): F64SparseFactorization =
        F64ReferenceSparseLinearAlgebra.ldl(a, policy, ordering)
}
