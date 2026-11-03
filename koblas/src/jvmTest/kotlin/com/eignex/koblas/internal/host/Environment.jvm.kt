package com.eignex.koblas.internal.host

internal actual fun environmentVariableOrNull(name: String): String? = System.getenv(name)
