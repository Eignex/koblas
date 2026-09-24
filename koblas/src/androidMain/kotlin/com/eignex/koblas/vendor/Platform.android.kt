package com.eignex.koblas.vendor

/** Resolved once; none of these facts change while the process runs. */
private val host: HostPlatform by lazy {
    HostPlatform(OperatingSystem.Other, architecture(), CpuVendor.Unknown)
}

/**
 * The platform this process is running on.
 *
 * Android reports its kernel as Linux, but none of the Linux vendors are installed on it, so it is
 * [OperatingSystem.Other] and [Vendor.select] offers nothing; [openBlas] reaches the bundled build directly.
 */
public actual fun hostPlatform(): HostPlatform = host

private fun architecture(): Architecture = when (System.getProperty("os.arch").orEmpty()) {
    "amd64", "x86_64" -> Architecture.X86_64
    "aarch64", "arm64" -> Architecture.Arm64
    else -> Architecture.Other
}

/**
 * The OpenBLAS this library bundles, or null where it did not load.
 *
 * Android ships no BLAS of its own, so there is nothing to search: the bundled build is the only candidate,
 * and naming any other vendor in [only] finds nothing.
 */
public actual fun openBlas(only: Vendor?): Blas? =
    if (only == null || only == Vendor.OpenBlas) AndroidVendorBlas.open() else null
