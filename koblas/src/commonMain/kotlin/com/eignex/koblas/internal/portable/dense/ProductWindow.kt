@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.internal.portable.dense

/**
 * Whether copying both operands into tiles pays for itself at these extents.
 *
 * The copy is `rows · depth + depth · columns` entries and the arithmetic `rows · columns · depth` products,
 * so what decides it is how much arithmetic each copied value takes part in. A product with an extent below
 * one tile cannot fill the tiles it would pack into, and a small product pays the copy against too little
 * arithmetic to hide it behind; both run as panel work over the operands where they are instead.
 *
 * The threshold reflects a measured crossover on one host. Sampled shapes favored packing above it,
 * while winners below it varied within the run-to-run band. Thin shapes can still lose above it;
 * no measured secondary condition separated those losses reliably.
 */
internal fun packsProductByWork(rows: Int, columns: Int, depth: Int, tileRows: Int, tileColumns: Int): Boolean =
    rows >= tileRows && columns >= tileColumns &&
        rows.toLong() * columns * depth >= PACKED_PRODUCT_MINIMUM_WORK

/** Products at or above this many multiply-adds are packed. */
internal const val PACKED_PRODUCT_MINIMUM_WORK: Long = 32L * 32L * 32L
