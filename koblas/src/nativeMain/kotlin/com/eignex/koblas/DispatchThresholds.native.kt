@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString

/** Environment only, since there are no system properties outside the JVM. */
internal actual fun dispatchOverride(element: ElementType, level: DispatchLevel): Int? =
    platform.posix.getenv(ConfigurationKeys.dispatchEnv(element, level))?.toKString()?.toIntOrNull()
