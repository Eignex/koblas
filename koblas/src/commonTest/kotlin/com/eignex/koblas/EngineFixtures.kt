package com.eignex.koblas

import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

internal fun engineWithKernels(
    vectors: KoblasEngine = ScalarEngine,
    panels: KoblasEngine = ScalarEngine,
    products: KoblasEngine = ScalarEngine,
    triangles: KoblasEngine = ScalarEngine,
    indexed: KoblasEngine = ScalarEngine,
): BuiltinEngine = BuiltinEngine(
    "test",
    KernelSet(
        (vectors as? BuiltinEngine)?.kernels?.vectors ?: vectors as EngineKernels,
        (indexed as? BuiltinEngine)?.kernels?.indexed ?: indexed as EngineKernels,
        panels.panelProvider,
        products.productProvider,
        triangles.triangularProvider,
    ),
)

internal fun engineWithHost(portable: KoblasEngine, host: KoblasEngine, eligible: Boolean = true): BuiltinEngine =
    BuiltinEngine(
        "host-test",
        (portable as? BuiltinEngine)?.kernels ?: KernelSet(),
        (host as EngineKernels).takeIf { eligible },
    )
