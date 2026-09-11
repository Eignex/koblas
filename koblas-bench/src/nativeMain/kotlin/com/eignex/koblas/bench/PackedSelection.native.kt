package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.PackedKernels

internal actual fun benchmarkPackedKernels(engine: KoblasEngine): PackedKernels = engine.packedKernels
