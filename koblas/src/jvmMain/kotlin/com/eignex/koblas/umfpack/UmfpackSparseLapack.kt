package com.eignex.koblas.umfpack

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.SingularSparseFactorization
import com.eignex.koblas.sparse.SparseFactorization
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLu
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE

/**
 * The sparse factorization half, backed by SuiteSparse's UMFPACK.
 *
 * The first host sparse backend koblas has, and the one that shows the sparse seams were drawn correctly:
 * `SparseLapack.factor` returns a `SparseFactorization` interface, so UMFPACK's opaque `void *Numeric` fits
 * behind it without the public model changing at all. Registered at priority 100, matching the dense
 * host-OpenBLAS backend, so it wins over the portable `SparseLu` wherever SuiteSparse is installed.
 *
 * A `SparseMatrix` crosses to `umfpack_di_*` with **no repacking**: koblas's CSC invariant is UMFPACK's
 * stated precondition, verified against the headers rather than assumed. That is the whole reason this
 * binding is short.
 *
 * Two things it deliberately does not do.
 *
 * It does not implement `SparseBlas`. A sparse `gemv` is `O(nnz)` memory-bound work, so a foreign call
 * has very little to amortize, and UMFPACK does not offer one anyway — the products would have to come
 * from CXSparse or CHOLMOD, each wanting its own struct built per call. koblas's own `gemv` walks the
 * same arrays with no call overhead at all.
 *
 * It does not gate on size. Every other host binding in koblas consults a dispatch threshold, because
 * for dense work there is a size below which the foreign call costs more than the arithmetic. Here the
 * arithmetic is a whole sparse LU with a fill-reducing ordering, so the crossover is expected to sit at
 * a very small `n`. "Expected" is not measured: see the threshold task before treating it as settled.
 */
class UmfpackSparseLapack : SparseLapack {
    override val name: String get() = "umfpack"

    override val priority: Int get() = 100

    /**
     * Factorize [a] with UMFPACK: symbolic analysis, then numeric factorization.
     *
     * [equilibrate] is accepted and ignored, which is the honest mapping rather than a silent lie. koblas's
     * `SparseLu` takes it to mean "scale rows by a power of two before pivoting"; UMFPACK scales rows by
     * default as part of its own strategy (`UMFPACK_SCALE_SUM`) and offers no way to ask for koblas's
     * specific variant. Requesting it therefore falls back to the portable factorization, so a caller who
     * asked for that scaling gets it rather than something else with the same name.
     */
    override fun factor(a: SparseMatrix, equilibrate: Boolean): SparseFactorization {
        require(a.rows == a.cols) { "factor requires a square matrix; got ${a.rows}x${a.cols}" }
        if (equilibrate) return SparseLu.factorCsc(a, equilibrate = true)
        if (a.rows == 0) return SparseLu.factorCsc(a)

        val colPtr = MemorySegment.ofArray(a.colPtr)
        val rowIdx = MemorySegment.ofArray(a.rowIdx)
        val values = MemorySegment.ofArray(a.values)

        // The symbolic analysis is scratch: UMFPACK keeps what it needs inside Numeric, so it is freed as
        // soon as the numeric factorization exists.
        Arena.ofConfined().use { scratch ->
            val symbolicHolder = scratch.allocate(ADDRESS)
            val info = scratch.allocate(JAVA_DOUBLE, UmfpackCalls.INFO.toLong())
            val symbolicStatus = UmfpackCalls.symbolic(a.rows, colPtr, rowIdx, values, symbolicHolder, info)
            if (symbolicStatus != UmfpackCalls.OK) {
                // A rejected pattern is not something to guess about; hand it to the portable path, which
                // will either factor it or report a position.
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
                // UMFPACK does hand back partial factors here, but koblas forbids solving against a singular
                // factorization, so they can never be used — and keeping them would hold native memory for
                // nothing. Returning the canonical singular result also keeps `nnz == 0` true of every
                // singular factorization whatever produced it, which is what SparseFactorization promises.
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
