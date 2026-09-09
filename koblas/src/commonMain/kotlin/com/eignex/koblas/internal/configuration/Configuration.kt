package com.eignex.koblas.internal.configuration

/** One JVM system property, or null off the JVM, where there are none. */
internal expect fun systemPropertyOrNull(name: String): String?

/** One environment variable, or null on a target that cannot read the environment. */
internal expect fun environmentVariableOrNull(name: String): String?
