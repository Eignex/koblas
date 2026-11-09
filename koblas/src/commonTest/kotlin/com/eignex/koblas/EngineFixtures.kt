package com.eignex.koblas

import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.KernelProvider
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
        (vectors as? BuiltinEngine)?.roles?.vectors ?: vectors as KernelProvider,
        (indexed as? BuiltinEngine)?.roles?.indexed ?: indexed as KernelProvider,
        panels.panelProvider,
        products.productProvider,
        triangles.triangularProvider,
    ),
)

internal fun engineWithHost(portable: KoblasEngine, host: KoblasEngine, eligible: Boolean = true): BuiltinEngine =
    BuiltinEngine(
        "host-test",
        (portable as? BuiltinEngine)?.roles ?: KernelSet(),
        (host as KernelProvider).takeIf { eligible },
    )
