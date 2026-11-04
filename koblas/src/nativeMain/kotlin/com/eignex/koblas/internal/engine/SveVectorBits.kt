package com.eignex.koblas.internal.engine

/** Linux reports the current thread's SVE width without changing its vector configuration. */
internal expect fun sveVectorBits(): Int?
