package com.eignex.koblas.dense

/**
 * Nothing to discover on a target with no host library to reach.
 *
 * That is JS and both Wasm flavours, which have no dynamic loader at all, and the native targets koblas does
 * not bind libraries on: mingw, where there is no `dlopen`, and iOS, where there is no host library to find
 * and an App Store binary should not carry the call. Linux and macOS get the real implementation instead,
 * from the source set that can see the bindings.
 */
internal actual fun registerPlatformBackends() = Unit
