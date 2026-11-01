package com.eignex.koblas

internal actual fun simdBackend(): EngineBackend? = null

/** Select an applicable measured profile once, with portable coverage when none matches. */
internal actual fun platformEngine(): KoblasEngine = standardEngine("native")
