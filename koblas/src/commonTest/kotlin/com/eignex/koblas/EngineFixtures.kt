package com.eignex.koblas

import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

internal fun engineWithKernels(
    vectors: VectorKernels = ScalarEngine,
    panels: PanelKernels = ScalarEngine,
    products: PackedProductKernels = ScalarEngine,
    triangles: TriangularKernels = ScalarEngine,
    indexed: IndexedKernels = ScalarEngine,
): KoblasEngine = BuiltinEngine(
    "test",
    KernelSet(
        (vectors as? BuiltinEngine)?.roles?.vectors ?: vectors,
        (indexed as? BuiltinEngine)?.roles?.indexed ?: indexed,
        panels,
        products,
        triangles,
    ),
).asEngine()

internal fun engineWithHost(portable: KoblasEngine, host: KernelProvider, eligible: Boolean = true): KoblasEngine =
    BuiltinEngine(
        "host-test",
        (portable.kernels as? BuiltinEngine)?.roles ?: KernelSet(),
        host.takeIf { eligible },
    ).asEngine()

internal fun engineWithHost(portable: KoblasEngine, host: KoblasEngine, eligible: Boolean = true): KoblasEngine =
    engineWithHost(portable, host.kernels, eligible)
