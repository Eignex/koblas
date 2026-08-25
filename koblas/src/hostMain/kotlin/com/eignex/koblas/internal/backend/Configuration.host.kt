package com.eignex.koblas.internal.backend

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.getenv

/** Kotlin/Native has no system properties, so a native deployment configures koblas through the environment. */
internal actual fun systemPropertyOrNull(name: String): String? = null

@OptIn(ExperimentalForeignApi::class)
internal actual fun environmentVariableOrNull(name: String): String? = getenv(name)?.toKString()
