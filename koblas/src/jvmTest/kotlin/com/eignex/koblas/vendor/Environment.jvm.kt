package com.eignex.koblas.vendor

internal actual fun environmentVariableOrNull(name: String): String? = System.getenv(name)
