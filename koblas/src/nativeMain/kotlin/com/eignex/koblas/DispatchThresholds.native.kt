@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString

/**
 * Environment only; there are no system properties outside the JVM.
 *
 * Read straight from `getenv` rather than through a `readEnv` indirection. Sharing this one line with the
 * web targets, which have no environment at all, would take an `expect` inside a non-common source set with
 * an actual per target — two source sets and a nested expect/actual pair to express "the web cannot do this".
 */
internal actual fun dispatchOverride(name: String): Int? =
    platform.posix.getenv("KOBLAS_DISPATCH_${name.uppercase()}")?.toKString()?.toIntOrNull()
