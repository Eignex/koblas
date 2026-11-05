package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.kernels.EngineKernels

/** Platform kernels with eligible whole-call host composition and portable fallbacks. */
internal expect fun platformEngine(): KoblasEngine

/** JVM system property before the environment variable; Native uses the environment variable. */
internal expect fun platformEngineSetting(): String?

internal expect fun simdBackend(): List<EngineKernels>?

internal expect fun vendorEngine(vendor: Vendor): KoblasEngine?
