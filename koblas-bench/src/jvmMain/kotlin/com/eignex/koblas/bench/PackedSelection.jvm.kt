package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.PackedKernels

internal actual fun benchmarkPackedKernels(engine: KoblasEngine): PackedKernels =
    engine.nativeVariant?.let { BuiltinEngines.exactC(it).packedKernels } ?: engine.packedKernels
