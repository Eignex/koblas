package com.eignex.koblas.vendor

/**
 * Where an optional vendor runtime is packaged, and when it is preferred over an installed one.
 *
 * Koblas never ships a vendor library in its own artifact. A separate optional module carries them, and this
 * file is the convention that module publishes into and the loader reads back, fixed here so that the two
 * cannot be designed independently and disagree.
 *
 * **A payload is a directory, not a file.** oneMKL's entry point is a dispatcher that opens its threading
 * layer, its interface layer and an instruction-set layer by name once it is running, so a payload that
 * carried only the name [Vendor.bundledFile] would open and then fail at the first call. Each vendor therefore
 * occupies a directory of its own, and [manifest] inside it lists what that directory holds. The list is
 * written by the packaging build rather than fixed here, because which files a runtime needs belongs to the
 * version that was packaged and would otherwise have to be edited here every time that version moved.
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
     * The directory a bundled [vendor] payload occupies for [host], or null when that combination is never
     * bundled.
     *
     * The same string addresses a JVM classpath resource and a directory laid out beside a Native executable,
     * so one convention describes both and the optional module publishes one layout rather than two. One
     * directory per vendor, so that a host reaching for AOCL unpacks AOCL and not the 227 megabytes of oneMKL
     * that share its platform.
     */
    fun directory(vendor: Vendor, host: HostPlatform): String? {
        val platform = platform(host) ?: return null
        if (vendor.bundledFile == null) return null
        return "$ROOT/$platform/${vendor.name.lowercase()}"
    }

    /** The file within [directory] that is opened as the library, or null when [vendor] is never bundled. */
    fun entry(vendor: Vendor, host: HostPlatform): String? {
        val directory = directory(vendor, host) ?: return null
        return "$directory/${vendor.bundledFile}"
    }

    /** The list of file names the payload holds, one per line, or null when [vendor] is never bundled. */
    fun manifest(vendor: Vendor, host: HostPlatform): String? = directory(vendor, host)?.let { "$it/$MANIFEST" }

    /**
     * The names [text] lists, or an empty list when any of them is not a plain file name.
     *
     * A payload is flat by construction. Reading a name with a separator or a parent reference in it would
     * mean writing outside the directory the loader created for it, so the whole manifest is rejected instead:
     * a payload that does not describe itself the way the convention says is not one to unpack halfway.
     */
    fun names(text: String): List<String> {
        val names = text.lines().map { it.trim() }.filter { it.isNotEmpty() }
        val flat = names.all { '/' !in it && '\\' !in it && it != "." && it != ".." }
        return if (flat) names else emptyList()
    }

    /**
     * The prefix of the directory the JVM loader extracts a payload into.
     *
     * Here rather than beside the extraction, because it is the second half of the same convention: a payload
     * on a Native host is read where it lies, and on the JVM it is a resource that has to become a file first.
     * Both forms are what [isPayload] recognises.
     */
    fun extraction(vendor: Vendor): String = "koblas-vendor-${vendor.name.lowercase()}-"

    /**
     * Whether the library at [path] is [vendor]'s bundled payload rather than a library installed on [host].
     *
     * The distinction is what makes the precedence rule checkable: a test that finds a payload answering has
     * learnt nothing about which of the two the loader prefers, because on that host there was only one.
     */
    fun isPayload(path: String, vendor: Vendor, host: HostPlatform): Boolean {
        val entry = entry(vendor, host) ?: return false
        return path.endsWith(entry) || extraction(vendor) in path
    }

    /** The prefix every bundled payload sits under, on the classpath and on disk alike. */
    const val ROOT: String = "com/eignex/koblas/vendor"

    /** The file each payload directory holds listing its own contents. */
    const val MANIFEST: String = "payload"
}
