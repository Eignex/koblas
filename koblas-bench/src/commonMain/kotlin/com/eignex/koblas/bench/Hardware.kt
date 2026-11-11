package com.eignex.koblas.bench

/** CPU model reported by Linux, or null when the platform supplies no model. */
internal expect fun cpuModel(): String?

/** Active SVE width for benchmark provenance, without changing the vector configuration. */
internal expect fun sveVectorBits(): Int?
