package com.eignex.koblas.vendor

/**
 * Where an optional vendor runtime is packaged, and when it is preferred over an installed one.
 *
 * Koblas never ships a vendor library in its own artifact. A separate optional module carries them, and this
 * file is the convention that module publishes into and the loader reads back, fixed here so that the two
 * cannot be designed independently and disagree.
 *
 * **Installed libraries win.** A library the operator installed is the one they chose, built or tuned for the
 * host, and possibly the one the rest of their stack already links; a bundled payload is a convenience for a
 * host that has none. Preferring the bundle would silently override that choice, and the failure mode is the
 * quiet kind, where everything still computes and only the performance and the version are not what was meant.
 * The order is fixed rather than configurable, so a report naming a vendor means the same thing everywhere.
 */
internal object Bundle {
    /** The directory a payload for [host] is packaged under, or null where nothing is ever bundled. */
    fun platform(host: HostPlatform): String? = when {
        host.operatingSystem == OperatingSystem.Linux && host.architecture == Architecture.X86_64 -> "linux-x86_64"
        host.operatingSystem == OperatingSystem.Linux && host.architecture == Architecture.Arm64 -> "linux-arm64"
        host.operatingSystem == OperatingSystem.MacOs && host.architecture == Architecture.Arm64 -> "macosx-arm64"
        else -> null
    }

    /**
     * The path a bundled [vendor] library occupies for [host], or null when that combination is never bundled.
     *
     * The same string addresses a JVM classpath resource and a file laid out beside a Native executable, so one
     * convention describes both and the optional module publishes one layout rather than two.
     */
    fun path(vendor: Vendor, host: HostPlatform): String? {
        val platform = platform(host) ?: return null
        val file = vendor.bundledFile ?: return null
        return "$ROOT/$platform/$file"
    }

    /** The prefix every bundled payload sits under, on the classpath and on disk alike. */
    const val ROOT: String = "com/eignex/koblas/vendor"
}
