package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.NO_DROP
import com.eignex.koblas.sparse.SingularSparseFactorization
import com.eignex.koblas.sparse.SparseFactorization
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLu
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE

/** Sparse factorizations backed by a host UMFPACK. */
public class UmfpackSparseLapack : SparseLapack {
    override val name: String get() = "umfpack"

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /**
     * Factorizes [a], symbolic analysis then numeric factorization. [equilibrate] and [dropTolerance] fall
     * back to the portable path, since UMFPACK offers neither koblas's scaling nor a drop threshold.
     */
    override fun factor(a: SparseMatrix, equilibrate: Boolean, dropTolerance: Double): SparseFactorization {
        requireShape(a.rows == a.cols) { "factor requires a square matrix; got ${a.rows}x${a.cols}" }
        // Any tolerance but the default goes to the portable path, which owns validating the rest.
        if (equilibrate || dropTolerance != NO_DROP) return SparseLu.factorCsc(a, equilibrate, dropTolerance)
        if (a.rows == 0) return SparseLu.factorCsc(a)

        val colPtr = MemorySegment.ofArray(a.colPtr)
        val rowIdx = MemorySegment.ofArray(a.rowIdx)
        val values = MemorySegment.ofArray(a.values)

        // A confined arena for the symbolic analysis, which is scratch: Numeric keeps what it needs.
        Arena.ofConfined().use { scratch ->
            val symbolicHolder = scratch.allocate(ADDRESS)
            val info = scratch.allocate(JAVA_DOUBLE, UmfpackCalls.INFO.toLong())
            val symbolicStatus = UmfpackCalls.symbolic(a.rows, colPtr, rowIdx, values, symbolicHolder, info)
            if (symbolicStatus != UmfpackCalls.OK) {
                // A rejected pattern goes to the portable path.
                return SparseLu.factorCsc(a)
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

            if (numericStatus != UmfpackCalls.OK && numericStatus != UmfpackCalls.WARNING_SINGULAR) {
                arena.close()
                return SparseLu.factorCsc(a)
            }
            if (numericStatus == UmfpackCalls.WARNING_SINGULAR) {
                // The partial factors UMFPACK hands back cannot be solved against, so they are released and
                // the canonical singular result keeps `nnz == 0` true of every singular factorization.
                UmfpackCalls.freeNumeric(numericHolder)
                arena.close()
                return SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
            }
            val handles = UmfpackFactorization.Handles(arena, numericHolder)
            val factorization = UmfpackFactorization(a, NOT_SINGULAR, handles)
            val (lnz, unz) = UmfpackFactorization.fillOf(info)
            factorization.lnz = lnz
            factorization.unz = unz
            return factorization
        }
    }
}
