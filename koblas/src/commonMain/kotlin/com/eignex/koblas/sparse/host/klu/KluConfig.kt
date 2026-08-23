package com.eignex.koblas.sparse.host.klu

/** Policy for one KLU backend instance. KLU's remaining Common fields retain its native defaults. */
public data class KluConfig(
    /** An absolute KLU library path, or the deployment lookup chain when null. */
    val libraryPath: String? = null,
)
