package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.EngineImplementation.*
import com.eignex.koblas.internal.engine.combine
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

internal fun engineWithKernels(
    vectors: KoblasEngine = ScalarEngine,
    panels: KoblasEngine = ScalarEngine,
    products: KoblasEngine = ScalarEngine,
    triangles: KoblasEngine = ScalarEngine,
    indexed: KoblasEngine = ScalarEngine,
): KoblasEngine = combine(
    EngineProfile(
        "test",
        listOf(
            vectors to EngineImplementation.entries.filter { it.vector && !it.name.startsWith("Indexed") }.toSet(),
            indexed to setOf(IndexedDot),
            panels to setOf(Panels),
            products to setOf(Products),
            triangles to setOf(Triangular),
        ).flatMap { (engine, operations) ->
            engine.providers.map { EngineRule(it.id, operations) }
        },
    ),
    panels,
    products,
    triangles,
    indexed,
    vectors,
)

internal fun engineWithHost(
    portable: KoblasEngine,
    host: KoblasEngine,
    eligible: Boolean = true,
    borrowed: Boolean = true,
): KoblasEngine = combine(
    EngineProfile(
        "host-test",
        buildList {
            if (eligible && host.implementations.isNotEmpty()) {
                add(EngineRule(host.id, host.implementations, borrowed = if (borrowed) null else false))
            }
            portable.providers.filter { it.implementations.isNotEmpty() }.forEach {
                add(EngineRule(it.id, it.implementations))
            }
        },
    ),
    host,
    portable,
)
