package com.eignex.koblas

/**
 * Maintains a basis factorization across rank-1 basis updates using the **product-form of the inverse**
 * (PFI): a base sparse LU [SparseLu] of the basis `B₀` at the last refactorization, plus a chain of
 * elementary "eta" transforms — one appended per update. [ftran] (`B x = b`) and [btran] (`Bᵀ x = b`)
 * apply the base LU solve and then walk the eta chain, so an update costs one `O(m)` [update] instead of
 * a full `O(nnz)` refactorization.
 *
 * Each eta records the pivot position `p` (a basis slot) and the spike `η = B⁻¹ A_q` — the entering
 * column transformed by the factorization in effect *just before* the update, i.e. the FTRAN result the
 * caller already computed. After basis slot `p` is replaced by column `q`, the new basis is `B·E` where
 * `E` is the identity with column `p` set to `η`; its inverse differs from the identity only in column
 * `p`, so applying it (and its transpose) is `O(m)` per eta.
 *
 * The chain lengthens fill and accumulates rounding error, so the caller refactorizes once [etaCount]
 * reaches its limit (rebuilding `B₀` and dropping the chain). Spikes are stored densely (length `m`);
 * index spaces match [SparseLu]'s (basis-slot in, original-row out).
 */
class EtaBasis private constructor(private val m: Int, private val base: SparseLu) {
    private val etaRow = ArrayList<Int>()
    private val etaSpike = ArrayList<DoubleArray>()

    /** Number of updates folded into the chain since the base factorization. */
    val etaCount: Int get() = etaRow.size

    /** Solve `B x = b` (FTRAN): base LU solve, then forward through the eta chain in update order.
     *  Each eta applies over the two contiguous runs around the pivot via [denseAxpy]. */
    fun ftran(b: DoubleArray): DoubleArray {
        val x = base.ftran(b)
        for (j in etaSpike.indices) {
            val p = etaRow[j]
            val eta = etaSpike[j]
            val xp = x[p] / eta[p]
            if (xp != 0.0) {
                denseAxpy(x, 0, -xp, eta, 0, p)
                denseAxpy(x, p + 1, -xp, eta, p + 1, m - p - 1)
            }
            x[p] = xp
        }
        return x
    }

    /** Solve `Bᵀ x = b` (BTRAN): the eta chain transposed in reverse update order, then the base LU.
     *  Each eta gathers over the two contiguous runs around the pivot via [denseDot]. */
    fun btran(b: DoubleArray): DoubleArray {
        val z = b.copyOf()
        for (j in etaSpike.indices.reversed()) {
            val p = etaRow[j]
            val eta = etaSpike[j]
            val s = z[p] - denseDot(eta, 0, z, 0, p) - denseDot(eta, p + 1, z, p + 1, m - p - 1)
            z[p] = s / eta[p]
        }
        return base.btran(z)
    }

    /** Append the eta for an update replacing basis slot [pivotRow]; [spike] must be this object's
     *  [ftran] of the entering column, computed *before* this call (pivot magnitude already checked). */
    fun update(pivotRow: Int, spike: DoubleArray) {
        etaRow.add(pivotRow)
        etaSpike.add(spike.copyOf())
    }

    /** Factory for an [EtaBasis]. */
    companion object {
        /** Wrap an already-factorized basis [lu] (`m × m`) as a fresh, empty eta chain. */
        fun of(lu: SparseLu, m: Int): EtaBasis = EtaBasis(m, lu)
    }
}
