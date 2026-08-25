package com.eignex.koblas.internal.backend

internal actual fun systemPropertyOrNull(name: String): String? = System.getProperty(name)

internal actual fun environmentVariableOrNull(name: String): String? = System.getenv(name)
