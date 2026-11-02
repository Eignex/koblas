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
 * The supported library bindings are fixed at compile time. Measured engine profiles choose among
 * available bindings; explicit selection opens one named vendor.
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
     * Distribution BLAS fallback on Linux and Windows. Default engines retain portable BLAS without any library.
     * A binding must pass the integer-ABI probe and single-thread controls before selection; those controls
     * also affect other callers of the same loaded library in this process.
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
    internal fun resolvedCandidates(
        home: String?,
        operatingSystem: OperatingSystem = hostPlatform().operatingSystem,
    ): List<String> = candidatesFor(operatingSystem).mapNotNull { candidate ->
        when {
            USER_HOME !in candidate -> candidate
            home.isNullOrEmpty() -> null
            else -> candidate.replace(USER_HOME, home.trimEnd('/'))
        }
    }

    /** Windows probes only absolute PATH entries so the working directory cannot supply a vendor DLL. */
    internal fun loadCandidates(home: String?, operatingSystem: OperatingSystem, searchPath: String?): List<String> {
        val names = resolvedCandidates(home, operatingSystem)
        if (operatingSystem != OperatingSystem.Windows) return names
        val directories = searchPath.orEmpty().split(';').map { it.trim().trim('"').replace('\\', '/') }
            .filter { it.isWindowsAbsolutePath() }.distinct()
        return names.flatMap { name -> directories.map { "${it.trimEnd('/')}/$name" } }
    }

    /** Windows uses DLL names; explicit Unix bindings retain the declared candidate paths. */
    private fun candidatesFor(operatingSystem: OperatingSystem): List<String> = when (operatingSystem) {
        OperatingSystem.Windows -> when (this) {
            OneMkl -> listOf("mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.dll")

            Aocl -> listOf(
                "libblis.dll",
                "AOCL-LibBlis-Win-dll.dll",
                "libblis-mt.dll",
                "AOCL-LibBlis-Win-MT-dll.dll",
                "aocl.dll",
            )

            ArmPl -> listOf("armpl_lp64.dll", "armpl_lp64_mp.dll")

            OpenBlas -> listOf("libopenblas.dll", "openblas.dll")

            else -> emptyList()
        }

        else -> candidates
    }
}

/** Operating systems the binding layer distinguishes. */
public enum class OperatingSystem {
    /** Any Linux distribution. */
    Linux,

    /** macOS, where Accelerate is part of the system. */
    MacOs,

    /** Windows, with optional installed BLAS DLLs. */
    Windows,

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

/**
 * The host facts measured profiles match. Resolved once per process; none of these change while it runs.
 */
public class HostPlatform(
    /** The operating system. */
    public val operatingSystem: OperatingSystem,
    /** The instruction set architecture. */
    public val architecture: Architecture,
    /** The reported CPU model, or null when it could not be identified. */
    public val cpuModel: String? = null,
) {
    override fun toString(): String = "$operatingSystem/$architecture/${cpuModel ?: "unknown"}"
}

/** The platform this process is running on. */
public expect fun hostPlatform(): HostPlatform

/** Drive rooted and UNC paths are independent of the current directory and the current drive. */
internal fun String.isWindowsAbsolutePath(): Boolean =
    (length >= 3 && this[0].isLetter() && this[1] == ':' && this[2] == '/') ||
        (startsWith("//") && split('/').filter { it.isNotEmpty() }.size >= 2)
