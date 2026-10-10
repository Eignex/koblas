package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngineApi

/** Runs [body] against every product backend this platform has, which is what decides the block geometry. */
@OptIn(KoblasEngineApi::class)
internal fun withProducts(body: (DenseProductKernels) -> Unit) {
    body(PortableProductKernels)
    BuiltinEngines.simd?.let { body(it.productKernels) }
}

/** The product backend of the engine whose scheduling the conformance checks above run under. */
@OptIn(KoblasEngineApi::class)
internal fun defaultProducts(): DenseProductKernels = BuiltinEngines.simd?.productKernels ?: PortableProductKernels

/**
 * A destination order past several of this backend's tiles and one row short of a whole number of them,
 * read from the backend because a fixed order fills a narrow tile exactly and leaves a wide one short.
 */
internal fun blockedOrder(products: DenseProductKernels): Int = 4 * maxOf(products.tileRows, products.tileColumns) + 1

/** The smallest depth at which a triangle-selected product of this order is packed into tiles. */
internal fun blockedDepth(products: DenseProductKernels, order: Int): Int {
    var depth = 8
    while (depth < MAXIMUM_FIXTURE_DEPTH && !packsWindow(products, order, order, depth, OutputTriangle.Lower)) {
        depth *= 2
    }
    return depth
}

/** A bound on the fixtures above, so a backend that never packs produces a test that fails rather than hangs. */
private const val MAXIMUM_FIXTURE_DEPTH = 4096

/**
 * A square order at which a product of it by itself is packed and which leaves a row edge; a rank update's
 * operand can share its destination only at one shape, so the packed alias case cannot use a rectangle.
 */
internal fun packedSquareOrder(products: DenseProductKernels): Int {
    var order = blockedOrder(products)
    while (order < MAXIMUM_FIXTURE_ORDER && !packsWindow(products, order, order, order, OutputTriangle.Lower)) {
        order += products.tileRows
    }
    return order
}

/** A bound on the search above, for the reason [MAXIMUM_FIXTURE_DEPTH] is one. */
private const val MAXIMUM_FIXTURE_ORDER = 512
