package com.eignex.koblas

/** No environment to read from a browser or a Wasm sandbox, and no native backend to tune either. */
internal actual fun readEnv(name: String): String? = null
