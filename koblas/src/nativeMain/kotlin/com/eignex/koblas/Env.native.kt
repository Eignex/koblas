@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString

internal actual fun readEnv(name: String): String? = platform.posix.getenv(name)?.toKString()
