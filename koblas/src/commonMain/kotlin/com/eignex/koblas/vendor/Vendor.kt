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
 */
public enum class Vendor(
    /** Name used in reports and route descriptions. */
    public val vendorName: String,
    /** Library file names tried in order. The first that opens and exports [keySymbol] wins. */
    internal val candidates: List<String>,
    /** How this vendor is held to one compute thread; see [ThreadControl]. */
    internal val threadControl: ThreadControl,
) {
    /**
     * Apple's system BLAS, supplied by macOS and installed with it.
     *
     * The one supported vendor whose single compute thread cannot be read back. It exports no thread-count
     * entry point, so the requirement is established through [ACCELERATE_THREAD_LIMIT] and reported as
     * [ThreadEvidence.Unconfirmed] rather than claimed as checked.
     *
     * It stays selectable on that basis: the requirement is that a backend be held to one thread, not that it
     * be able to describe itself, and Accelerate can be held. Treating an unreadable count as a failure to
     * enforce would leave macOS with no vendor at all. The cost is that its arm carries weaker evidence than
     * the others, which its reports say.
     */
    Accelerate(
        "Accelerate",
        listOf("/System/Library/Frameworks/Accelerate.framework/Accelerate"),
        threadControl = ThreadControl.Environment,
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
    ),

    /** Arm Performance Libraries, in its LP64 form. */
    ArmPl(
        "ArmPL",
        listOf("libarmpl_lp64.so", "libarmpl.so", "libarmpl_lp64_mp.so"),
        threadControl = ThreadControl.OpenMp,
    ),

    /**
     * The last resort on Linux, and the comparison arm every benchmark runs against.
     *
     * Every other vendor is tuned for one vendor's parts and installed deliberately. OpenBLAS is what a
     * distribution ships, so it is the library a host is most likely to already have and the only one that
     * makes the difference between Level 2 and 3 working and raising. It is chosen last because a
     * distribution build is compiled for whatever runs everywhere rather than for the part in front of it, so
     * where a tuned library is present that one is better; where none is, this is far better than nothing.
     *
     * Nothing is taken on trust: a build advertising 64-bit integers, which distributions do ship, is refused
     * by the ABI probe, and one that will not hold to a single compute thread never becomes a binding.
     *
     * Holding it reaches further than it does for the others. A tuned library is installed for this process,
     * whereas the distribution's OpenBLAS is the one everything on the host shares, and the thread count is
     * set in the loaded library rather than in Koblas, so anything else in this process that reached the same
     * file runs on one compute thread as well.
     */
    OpenBlas(
        "OpenBLAS",
        listOf("libopenblas.so.0", "libopenblas.so"),
        threadControl = ThreadControl.OpenBlas,
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
         *
         * Each Linux list ends in [OpenBlas], which is what makes the difference between a host with a
         * distribution BLAS computing and raising. The tuned library wins wherever it is installed; the
         * fallback only decides what happens when none is. macOS needs no fallback, because Accelerate is part
         * of the system and cannot be missing.
         *
         * Which one answered is never a guess: the route and the resolved file name the library that ran, so a
         * host that fell back says so rather than reporting the vendor it would have preferred.
         */
        public fun select(host: HostPlatform): List<Vendor> = when {
            host.operatingSystem == OperatingSystem.MacOs -> listOf(Accelerate)

            // Every candidate below is an ELF soname, so the operating system is part of the question and not
            // only the architecture: offering them anywhere else would name libraries that cannot open there.
            host.operatingSystem != OperatingSystem.Linux -> emptyList()

            host.architecture == Architecture.Arm64 -> listOf(ArmPl, OpenBlas)

            host.architecture == Architecture.X86_64 && host.cpuVendor == CpuVendor.Amd ->
                listOf(Aocl, OneMkl, OpenBlas)

            host.architecture == Architecture.X86_64 -> listOf(OneMkl, Aocl, OpenBlas)

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
