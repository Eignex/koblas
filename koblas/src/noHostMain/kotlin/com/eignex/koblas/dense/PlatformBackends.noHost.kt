package com.eignex.koblas.dense

/**
 * Nothing to discover on a target with no host library to reach: JS, both Wasm flavours, mingw and iOS.
 * Linux and macOS get the real implementation from the source set that can see the bindings.
 */
internal actual fun registerPlatformBackends() = Unit
