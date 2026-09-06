package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.F64BundledBackend

/** A provider that is nothing but a name, which is all the pin and offer logic reads of one. */
internal fun namedProvider(provider: String): Backend = object : Backend {
    override val name: String get() = provider
}

/** The same carrying a bundled build of [canonical], the name a deployment configures it under. */
internal fun bundledProvider(provider: String, canonical: String): F64BundledBackend = object : F64BundledBackend {
    override val name: String get() = provider
    override val canonicalName: String get() = canonical
}
