@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString

/** Environment only, since there are no system properties outside the JVM. */
internal actual fun dispatchOverride(level: DispatchLevel): Int? =
    platform.posix.getenv(ConfigurationKeys.DISPATCH_ENV_PREFIX + level.name)?.toKString()?.toIntOrNull()
