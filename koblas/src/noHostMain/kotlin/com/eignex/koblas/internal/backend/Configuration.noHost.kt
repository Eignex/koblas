package com.eignex.koblas.internal.backend

/** No host library to point anywhere on these targets, so nothing reads configuration either. */
internal actual fun systemPropertyOrNull(name: String): String? = null

internal actual fun environmentVariableOrNull(name: String): String? = null
