@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString

/**
 * Environment only; there are no system properties outside the JVM.
 *
 * Read straight from `getenv` rather than through a `readEnv` indirection. That indirection used to be an
 * `expect` declared inside a non-common source set, with an actual per target, purely so this one line
 * could be shared with the web targets that have no environment at all — two source sets and a nested
 * expect/actual pair to express "the web cannot do this".
 */
internal actual fun dispatchOverride(name: String): Int? =
    platform.posix.getenv("KOBLAS_DISPATCH_${name.uppercase()}")?.toKString()?.toIntOrNull()
