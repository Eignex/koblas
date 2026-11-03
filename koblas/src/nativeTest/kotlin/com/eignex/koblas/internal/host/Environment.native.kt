package com.eignex.koblas.internal.host

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.getenv

@OptIn(ExperimentalForeignApi::class)
internal actual fun environmentVariableOrNull(name: String): String? = getenv(name)?.toKString()
