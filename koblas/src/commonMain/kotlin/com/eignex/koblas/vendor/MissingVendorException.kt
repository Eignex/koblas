package com.eignex.koblas.vendor

/**
 * Raised by a dense Level 2 or 3 call on a host where no supported BLAS library was found.
 *
 * Containers, Level 1 and the generic primitives do not raise this: they are implemented here and do not need
 * a library. Only the operations a vendor owns do, and they raise rather than compute, because the alternative
 * is a portable substitute reported under a vendor's name.
 *
 * The message lists what was searched, because on a host with no library the useful answer is which file to
 * install rather than that something was absent. It is built when the failure happens and never on the calling
 * path of a successful operation.
 */
public class MissingVendorException internal constructor(private val host: HostPlatform = hostPlatform()) :
    IllegalStateException() {
    override val message: String
        get() = buildString {
            val vendors = Vendor.select(host)
            append("no supported BLAS library was found for ")
            append(host.operatingSystem.name.lowercase()).append('/').append(host.architecture.name.lowercase())
            append(", so dense Level 2 and 3 operations cannot run. ")
            if (vendors.isEmpty()) {
                append("No vendor is supported on this platform.")
                return@buildString
            }
            append("Install one of: ")
            // The raw candidate patterns rather than the resolved paths: {home} says which prefix is meant
            // without this message needing a platform seam of its own to expand it.
            vendors.joinTo(this, "; ") { vendor ->
                "${vendor.vendorName} (tried ${vendor.candidates.joinToString(", ")})"
            }
            append(". Containers, Level 1 and the sparse primitives work without one.")
        }
}
