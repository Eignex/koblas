package com.eignex.koblas.internal.host

/** One environment variable, or null on a target that cannot read the environment. */
internal expect fun environmentVariableOrNull(name: String): String?
