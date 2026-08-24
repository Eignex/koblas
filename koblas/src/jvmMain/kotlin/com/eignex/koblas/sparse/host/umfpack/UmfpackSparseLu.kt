package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.F64SparseLuAdapter
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE

/** Sparse factorizations backed by a host UMFPACK. */
public class UmfpackSparseLu(
    /** Policy for this backend instance and the factors it produces. */
    public val config: UmfpackConfig = UmfpackConfig(),
) : F64SparseLuAdapter(config.factorizeMin) {
    private val calls = UmfpackCalls(config)

    override val name: String get() = BackendNames.UMFPACK

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** UMFPACK ships separately from OpenBLAS, so a host can have one without the other. */
    override val nativeAvailable: Boolean get() = calls.available

    internal val refinementSteps: Double? get() = calls.refinementSteps

    internal val pivotTolerance: Double? get() = calls.pivotTolerance

    /** Factorizes [a] through UMFPACK's symbolic analysis and numeric factorization. */
    override fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization {
        val colPtr = MemorySegment.ofArray(a.colPtr)
        val rowIdx = MemorySegment.ofArray(a.rowIdx)
        val values = MemorySegment.ofArray(a.values)
        val control = calls.control(equilibrate)

        // A confined arena for the symbolic analysis, which is scratch: Numeric keeps what it needs.
        Arena.ofConfined().use { scratch ->
            val symbolicHolder = scratch.allocate(ADDRESS)
            val info = scratch.allocate(JAVA_DOUBLE, INFO.toLong())
            val symbolicStatus = calls.symbolic(a.rows, colPtr, rowIdx, values, symbolicHolder, control, info)
            check(symbolicStatus == OK) { "umfpack_di_symbolic failed with status $symbolicStatus" }

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
                    calls.numeric(
                        colPtr,
                        rowIdx,
                        values,
                        symbolicHolder.get(ADDRESS, 0),
                        numericHolder,
                        control,
                        info,
                    )
                } finally {
                    // Scratch either way; UMFPACK keeps what it needs inside Numeric.
                    calls.freeSymbolic(symbolicHolder)
                }

                check(numericStatus == OK || numericStatus == WARNING_SINGULAR) {
                    "umfpack_di_numeric failed with status $numericStatus"
                }
                // The partial factors UMFPACK hands back for a singular matrix cannot be solved against, so
                // the canonical singular result keeps `nnz == 0` true of every singular factorization.
                if (numericStatus == WARNING_SINGULAR) {
                    return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
                }
                val factorization = UmfpackFactorization(a, NOT_SINGULAR, arena, numericHolder, calls, control)
                val (lnz, unz) = UmfpackFactorization.fillOf(info)
                factorization.lnz = lnz
                factorization.unz = unz
                factorization.rcond = UmfpackFactorization.rcondOf(info)
                owned = false
                return factorization
            } finally {
                if (owned) {
                    calls.freeNumeric(numericHolder)
                    arena.close()
                }
            }
        }
    }
}
