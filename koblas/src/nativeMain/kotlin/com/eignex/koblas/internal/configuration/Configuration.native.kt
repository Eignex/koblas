package com.eignex.koblas.internal.configuration

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.getenv

/** Kotlin/Native has no system properties, so native deployments use environment variables. */
internal actual fun systemPropertyOrNull(name: String): String? = null

@OptIn(ExperimentalForeignApi::class)
internal actual fun environmentVariableOrNull(name: String): String? = getenv(name)?.toKString()
