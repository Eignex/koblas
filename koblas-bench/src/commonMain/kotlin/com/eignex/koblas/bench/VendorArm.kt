package com.eignex.koblas.bench

import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.*

/**
 * The work for one benchmark case on one arm, or the reason there is none to time.
 *
 * Rejected arms leave the row untimed so measurements remain attributable to the requested implementation.
 */
internal sealed class ArmChoice {
    open val work: CaseWork? get() = null
    open val reason: String? get() = null

    class Accepted(override val work: CaseWork) : ArmChoice()
    class Declined(override val reason: String) : ArmChoice()
}

/** The prefix a vendor mode carries for each runtime, which is also which entry point may run it. */
internal const val JVM_VENDOR_PREFIX: String = "jvm-vendor-"

/** The Native runtime's vendor-mode prefix. */
internal const val NATIVE_VENDOR_PREFIX: String = "native-vendor-"

/** The vendor a benchmark mode names, or null when the mode is not a vendor arm. */
internal fun vendorFromMode(mode: String): Vendor? {
    val name = when {
        mode.startsWith(JVM_VENDOR_PREFIX) -> mode.removePrefix(JVM_VENDOR_PREFIX)
        mode.startsWith(NATIVE_VENDOR_PREFIX) -> mode.removePrefix(NATIVE_VENDOR_PREFIX)
        else -> return null
    }
    return Vendor.entries.singleOrNull { it.name.lowercase() == name }
        ?: error("unknown vendor in mode $mode; expected one of ${Vendor.entries.map { it.name.lowercase() }}")
}

/**
 * The vendor a mode names for [prefix]'s runtime, or null when the mode is not that runtime's vendor arm.
 *
 * The runtime prefix prevents a runner from publishing another runtime's vendor mode.
 */
internal fun vendorForRuntime(mode: String, prefix: String): Vendor? =
    if (mode.startsWith(prefix)) vendorFromMode(mode) else null

/** A vendor-isolated automatic policy, with SIMD withheld by the launching process. */
internal fun automaticVendorFromMode(mode: String): Vendor? {
    if (!mode.startsWith("jvm-auto-")) return null
    require(mode.endsWith("-no-simd")) { "Automatic vendor modes must name the SIMD state" }
    val name = mode.removePrefix("jvm-auto-").removeSuffix("-no-simd")
    return requireNotNull(Vendor.entries.singleOrNull { it.vendorName.lowercase() == name }) {
        "Unknown vendor in automatic mode $mode"
    }
}

/**
 * Opens the vendor a mode names, failing loudly rather than quietly measuring something else.
 *
 * The description records the resolved file, version and single-thread evidence. Libraries that cannot
 * hold to one thread are rejected; those without a thread-count entry point cannot claim confirmation.
 */
internal fun openVendorForMode(mode: String): Pair<KoblasEngine, String> {
    val vendor = vendorFromMode(mode) ?: error("$mode is not a vendor mode")
    val blas = KoblasEngine.vendor(vendor)
        ?: error("requested vendor ${vendor.vendorName} is not installed; it cannot be benchmarked on this host")
    return blas to engineIdentity(mode, blas)
}

/** Exact vendor calls use the same fixtures, validation and timed bodies as the other engine arms. */
internal fun vendorArm(case: BenchCase, blas: KoblasEngine): ArmChoice {
    val operation = if (case.operation == "ger-shrinking") EngineImplementation.Ger else case.implementation
    if (case.operation.startsWith("sp") || operation == null || operation in setOf(EngineImplementation.Panels, EngineImplementation.Products, EngineImplementation.Triangular) || operation.name.startsWith("Indexed") || operation == EngineImplementation.Sum) {
        return ArmChoice.Declined("${case.operation} has no vendor entry point in the bound CBLAS surface")
    }
    exactArmRejection(blas, operation)?.let { return ArmChoice.Declined(it) }
    val work = denseWork(case, blas) ?: return declined()
    return ArmChoice.Accepted(work.copy(comparisonKind = "direct", kernel = "${blas.vendor!!.vendorName}/$operation", provider = blas.id))
}

/** Admission and measured forks resolve the same exact or default-policy workload. */
internal fun benchmarkArm(case: BenchCase, engine: KoblasEngine, vendor: Boolean): ArmChoice? =
    if (vendor) vendorArm(case, engine) else engineArm(case, engine)

/** Both runners resolve an exact vendor separately from the platform's owned engines. */
internal fun benchmarkEngine(mode: String): Pair<KoblasEngine, String> =
    if (vendorFromMode(mode) != null) openVendorForMode(mode) else resolveEngine(mode)
