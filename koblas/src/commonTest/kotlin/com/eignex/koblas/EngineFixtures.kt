package com.eignex.koblas

import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.internal.dense.*

internal fun engineWithKernels(
    vectors: KoblasEngine = PortableEngine,
    panels: KoblasEngine = PortablePanelKernels,
    products: KoblasEngine = PortableProductKernels,
    triangles: KoblasEngine = PortableTriangularKernels,
    indexed: KoblasEngine = PortableEngine,
): KoblasEngine = KoblasEngine.combine(
    EngineProfile(
        "test",
        listOf(
            EngineRule(vectors.id, EngineImplementation.entries.takeWhile { it != IndexedDot }.toSet()),
            EngineRule(indexed.id, setOf(IndexedDot)),
            EngineRule(panels.id, setOf(Panels)),
            EngineRule(products.id, setOf(Products)),
            EngineRule(triangles.id, setOf(Triangular)),
        ),
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
): KoblasEngine = KoblasEngine.combine(
    EngineProfile(
        "host-test",
        buildList {
            if (eligible && host.implementations.isNotEmpty()) {
                add(EngineRule(host.id, host.implementations, borrowed = if (borrowed) null else false))
            }
            add(EngineRule(portable.id, EngineImplementation.entries.toSet()))
        },
    ),
    host,
    portable,
)
