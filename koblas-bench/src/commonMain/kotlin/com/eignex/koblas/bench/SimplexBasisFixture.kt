package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix
import kotlin.random.Random

/**
 * A structural stand-in for a simplex basis: near-triangular, mostly slack columns, a few spikes.
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
        entries.add(j to (1.0 + rng.nextDouble()))
        when {
            j < slacks -> Unit // a slack column is the unit vector, and stays one
            isSpike[j] -> {
                repeat(columnNonzeros) {
                    val i = rng.nextInt(n)
                    if (i != j) entries.add(i to rng.nextDouble(-1.0, 1.0))
                }
            }

            else -> {
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
