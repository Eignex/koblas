package com.eignex.koblas.vendor

/**
 * Stands in for the user's home directory inside a candidate path.
 *
 * At file level rather than in the companion because the enum entries are constructed before a companion of
 * their own class is initialized.
 */
private const val USER_HOME: String = "{home}"

/**
 * The closed set of BLAS libraries Koblas can bind.
 *
 * The set is fixed at compile time. There is no provider registration and no way to add an implementation at
 * runtime, so [select] is the whole of the selection policy and can be read in one sitting.
 *
 * [OpenBlas] is a benchmark reference only. It shares the CBLAS transport with the production vendors, which
 * makes it useful as a comparison arm and as a way to exercise the transport on a host that has no production
 * vendor installed, but [select] never returns it.
 */
public enum class Vendor(
    /** Name used in reports and route descriptions. */
    public val vendorName: String,
    /** Library file names tried in order. The first that opens and exports [keySymbol] wins. */
    internal val candidates: List<String>,
    /** How this vendor is held to one compute thread; see [ThreadControl]. */
    internal val threadControl: ThreadControl,
    /** Whether [select] may return this vendor. */
    internal val selectable: Boolean,
) {
    /**
     * Apple's system BLAS, supplied by macOS and installed with it.
     *
     * The one supported vendor whose single compute thread cannot be read back. It exports no thread-count
     * entry point, so the requirement is established through [ACCELERATE_THREAD_LIMIT] and reported as
     * [ThreadEvidence.Unconfirmed] rather than claimed as checked.
     *
     * It stays selectable on that basis. The requirement the plan sets is that a backend be held to one thread,
     * not that it be able to describe itself, and Accelerate can be held; treating an unreadable count as a
     * failure to enforce would leave macOS with no vendor at all, which is not what naming Accelerate the macOS
     * backend can mean. The cost is that its arm carries weaker evidence than the others, which its reports say.
     * Nothing here has been exercised on macOS hardware, so the lever's timing against Accelerate's own
     * initialization is the part still to confirm.
     */
    Accelerate(
        "Accelerate",
        listOf("/System/Library/Frameworks/Accelerate.framework/Accelerate"),
        threadControl = ThreadControl.Environment,
        selectable = true,
    ),

    /**
     * Intel oneAPI Math Kernel Library, reached through its single dispatching runtime.
     *
     * The default oneAPI installer puts the library under a versioned prefix that is not on the loader's
     * search path, and its `setvars` script is what normally puts it there. Depending on that script would
     * make whether Koblas finds a vendor depend on how the process was launched, so the standard prefixes are
     * candidates in their own right. `latest` is the symlink the installer maintains.
     */
    OneMkl(
        "oneMKL",
        listOf(
            "libmkl_rt.so.3",
            "libmkl_rt.so.2",
            "libmkl_rt.so",
            "$USER_HOME/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3",
            "/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3",
        ),
        threadControl = ThreadControl.Mkl,
        selectable = true,
    ),

    /**
     * AMD Optimizing CPU Libraries, whose BLAS is BLIS with the CBLAS interface compiled in.
     *
     * The serial build comes first. AOCL ships BLIS twice, once linked against a
     * threading runtime and once without, and both are held to one compute thread here; taking the serial
     * build means the single-thread requirement is met by the binary rather than by a call that configures a
     * pool down to one. The multithreaded build stays a candidate because a host may have only that one
     * installed, and it is still correct, just configured rather than built that way.
     *
     * Two sonames, because two projects package this library. AMD's AOCL 5 carries `libblis.so.5`; a
     * distribution's own BLIS package is a release behind at `libblis.so.4`.
     */
    Aocl(
        "AOCL",
        listOf("libblis.so.5", "libblis.so.4", "libblis.so", "libblis-mt.so.5", "libblis-mt.so.4", "libblis-mt.so"),
        threadControl = ThreadControl.Blis,
        selectable = true,
    ),

    /** Arm Performance Libraries, in its LP64 form. */
    ArmPl(
        "ArmPL",
        listOf("libarmpl_lp64.so", "libarmpl.so", "libarmpl_lp64_mp.so"),
        threadControl = ThreadControl.OpenMp,
        selectable = true,
    ),

    /** Benchmark reference only, never selected in production. */
    OpenBlas(
        "OpenBLAS",
        listOf("libopenblas.so.0", "libopenblas.so"),
        threadControl = ThreadControl.OpenBlas,
        selectable = false,
    ),
    ;

    /** Symbol every candidate must export to be accepted as this vendor. */
    internal val keySymbol: String get() = "cblas_dgemm"

    /** [candidates] with [USER_HOME] replaced by [home], or dropped when there is no home directory. */
    internal fun resolvedCandidates(home: String?): List<String> = candidates.mapNotNull { candidate ->
        when {
            USER_HOME !in candidate -> candidate
            home.isNullOrEmpty() -> null
            else -> candidate.replace(USER_HOME, home.trimEnd('/'))
        }
    }

    /** The preference order for a host, and the reasoning behind it. */
    public companion object {
        /**
         * The vendors to try on [host], most preferred first. An empty list means this host has no supported
         * accelerator, which is a clear failure for matrix operations and no failure at all for containers,
         * Level 1, and the generic primitives.
         *
         * Operating system and architecture decide before CPU vendor does. An Arm part from an unfamiliar
         * manufacturer still takes the ArmPL route rather than falling off the end of a CPU-vendor check.
         */
        public fun select(host: HostPlatform): List<Vendor> = preference(host).filter { it.selectable }

        /** The order alone, before [selectable] decides which of those a production caller may reach. */
        private fun preference(host: HostPlatform): List<Vendor> = when {
            host.operatingSystem == OperatingSystem.MacOs -> listOf(Accelerate)
            host.architecture == Architecture.Arm64 -> listOf(ArmPl)
            host.architecture == Architecture.X86_64 && host.cpuVendor == CpuVendor.Amd -> listOf(Aocl, OneMkl)
            host.architecture == Architecture.X86_64 -> listOf(OneMkl, Aocl)
            else -> emptyList()
        }
    }
}

/** Operating systems the binding layer distinguishes. */
public enum class OperatingSystem {
    /** Any Linux distribution. */
    Linux,

    /** macOS, where Accelerate is part of the system. */
    MacOs,

    /** Anything else, which has no supported vendor. */
    Other,
}

/** Instruction set architectures the binding layer distinguishes. */
public enum class Architecture {
    /** 64-bit x86. */
    X86_64,

    /** 64-bit Arm. */
    Arm64,

    /** Anything else, which has no supported vendor. */
    Other,
}

/** CPU manufacturers that change the preference order within one architecture. */
public enum class CpuVendor {
    /** An Intel part. */
    Intel,

    /** An AMD part. */
    Amd,

    /** A part whose manufacturer was not identified, which takes the architecture's default order. */
    Unknown,
}

/**
 * The host facts [Vendor.select] reads. Resolved once per process; none of these change while it runs.
 */
public class HostPlatform(
    /** The operating system. */
    public val operatingSystem: OperatingSystem,
    /** The instruction set architecture. */
    public val architecture: Architecture,
    /** The CPU manufacturer, which only refines the order within an architecture. */
    public val cpuVendor: CpuVendor,
) {
    override fun toString(): String = "$operatingSystem/$architecture/$cpuVendor"
}

/** The platform this process is running on. */
public expect fun hostPlatform(): HostPlatform
