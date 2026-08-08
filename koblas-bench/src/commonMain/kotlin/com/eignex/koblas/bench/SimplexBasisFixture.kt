package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix
import kotlin.random.Random

/**
 * A matrix shaped like a simplex basis rather than like noise.
 *
 * The sparse numbers koblas has been measured on until now came from a random matrix at one per cent density,
 * which is close to the worst case for any fill-reducing method — there is no structure to exploit, so every
 * ordering is as bad as every other and the factorization fills toward dense. Nothing klause factorizes looks
 * like that, and a comparison on it says more about the fixture than about the code.
 *
 * A basis matrix from a revised simplex looks like this instead. Most of its columns are slack columns, which
 * are unit vectors; the structural columns that have entered the basis are sparse, and the whole thing is
 * *near-triangular* — a permutation exists that makes it triangular except for a small number of **spike**
 * columns, which is exactly what makes a basis cheap to factorize and what refactorization schemes are built
 * around. [spikeFraction] is the knob that matters: at zero the matrix is triangular and factorizes with no
 * fill at all, and every spike adds a column that has to be eliminated against the ones after it.
 *
 * Deliberately not claimed to be a real basis. It is a structural stand-in built from the properties a basis
 * has — triangularity, slacks, a few spikes, entries near unit magnitude — so it exercises the same code paths
 * with the same shapes. A basis dumped from an actual klause solve, or a Netlib LP, is still worth measuring
 * against when one is available; this is what makes the comparison meaningful without one.
 *
 * @param n the basis dimension.
 * @param slackFraction how many columns are unit vectors, as a fraction of [n].
 * @param spikeFraction how many columns violate the triangular order, as a fraction of [n].
 * @param columnNonzeros entries in a structural column, before the triangular restriction.
 */
internal fun simplexBasis(
    n: Int,
    rng: Random,
    slackFraction: Double = 0.55,
    spikeFraction: Double = 0.08,
    columnNonzeros: Int = 6,
): SparseMatrix {
    val slacks = (n * slackFraction).toInt()
    val spikes = (n * spikeFraction).toInt()
    val isSpike = BooleanArray(n)
    repeat(spikes) { isSpike[rng.nextInt(n)] = true }
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        // The diagonal is always present and dominant, which is what keeps the basis non-singular without
        // having to check: a simplex basis is invertible by construction, and this stands in for that.
        entries.add(j to (1.0 + rng.nextDouble()))
        when {
            j < slacks -> Unit // a slack column is the unit vector, and stays one
            isSpike[j] -> {
                // A spike reaches *below* the diagonal, which is what forces elimination against later
                // columns; a triangular column never does.
                repeat(columnNonzeros) {
                    val i = rng.nextInt(n)
                    if (i != j) entries.add(i to rng.nextDouble(-1.0, 1.0))
                }
            }

            else -> {
                // A structural column in triangular position: entries only above the diagonal.
                repeat(columnNonzeros) {
                    val i = rng.nextInt(j + 1)
                    if (i != j) entries.add(i to rng.nextDouble(-1.0, 1.0))
                }
            }
        }
        entries
    }
    return SparseMatrix.ofColumns(n, n, columns)
}
