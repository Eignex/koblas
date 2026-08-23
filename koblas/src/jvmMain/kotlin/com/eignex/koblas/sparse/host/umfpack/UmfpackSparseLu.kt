package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization
import com.eignex.koblas.sparse.factorization.lu.NO_DROP
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE

/** Sparse factorizations backed by a host UMFPACK. */
public class UmfpackSparseLu : F64SparseLu {
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
        if (!UmfpackCalls.available) {
            return F64SparseLuFactorization.factorCsc(a, equilibrate, dropTolerance)
        }
        // Any tolerance but the default goes to the portable path, which owns validating the rest.
        if (equilibrate || dropTolerance != NO_DROP) {
            return F64SparseLuFactorization.factorCsc(
                a,
                equilibrate,
                dropTolerance,
            )
        }
        // Nothing to factor, and the portable path reports the pivot position where UMFPACK reports only
        // that it failed, so both bindings route this the same way.
        if (a.rows == 0 || a.nnz == 0) return F64SparseLuFactorization.factorCsc(a)

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
                return F64SparseLuFactorization.factorCsc(a)
            }

            // The numeric factors outlive this call, so their holder gets an arena the factorization owns.
            val arena = Arena.ofShared()
            val numericHolder = arena.allocate(ADDRESS)
            // Until a factorization holds them, this call owns the factors and the arena, and nothing else
            // would ever free them: the cleaner is registered with the factorization, so an escape before
            // that point strands both for the life of the process. Freed rather than only closed, because
            // UMFPACK nulls the handle on an error return but the arena owns the holder, not the factors.
            var owned = true
            try {
                val numericStatus = try {
                    UmfpackCalls.numeric(
                        colPtr,
                        rowIdx,
                        values,
                        symbolicHolder.get(ADDRESS, 0),
                        numericHolder,
                        info,
                    )
                } finally {
                    // Scratch either way; UMFPACK keeps what it needs inside Numeric.
                    UmfpackCalls.freeSymbolic(symbolicHolder)
                }

                if (numericStatus != OK && numericStatus != WARNING_SINGULAR) {
                    return F64SparseLuFactorization.factorCsc(a)
                }
                // The partial factors UMFPACK hands back for a singular matrix cannot be solved against, so
                // the canonical singular result keeps `nnz == 0` true of every singular factorization.
                if (numericStatus == WARNING_SINGULAR) {
                    return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
                }
                val handles = UmfpackFactorization.Handles(arena, numericHolder)
                val factorization = UmfpackFactorization(a, NOT_SINGULAR, handles)
                val (lnz, unz) = UmfpackFactorization.fillOf(info)
                factorization.lnz = lnz
                factorization.unz = unz
                factorization.rcond = UmfpackFactorization.rcondOf(info)
                owned = false
                return factorization
            } finally {
                if (owned) {
                    UmfpackCalls.freeNumeric(numericHolder)
                    arena.close()
                }
            }
        }
    }
}
