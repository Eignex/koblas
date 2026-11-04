package com.eignex.koblas.internal.host

import com.eignex.koblas.engine.*

/**
 * How one vendor is held to a single compute thread, and whether that can be read back.
 *
 * Declared per vendor rather than discovered by trying a list of symbols and accepting whatever answers. The
 * difference matters: a build of a known vendor that does not expose the control that vendor is supposed to
 * have is not a build this binding understands, and accepting it because no symbol matched would turn the
 * invariant into a hope. Such a library is refused instead.
 */
internal enum class ThreadControl(
    /** The entry point that fixes the count, or null where the vendor has none. */
    val setter: String?,
    /** The entry point that reads it back, or null where the vendor cannot be asked. */
    val getter: String?,
    /**
     * Whether the control being absent means the build simply has no threading.
     *
     * True only for OpenMP-based vendors. ArmPL ships a serial build alongside its `_mp` one, and the serial
     * build links no OpenMP runtime at all, so the absence of `omp_set_num_threads` is positive evidence that
     * there are no worker threads rather than a missing control. For every other vendor the control is part of
     * the library and its absence is a library this code does not recognize.
     * [permitsAbsentControl] also excludes candidates declared to be parallel builds.
     */
    val absenceMeansSerial: Boolean = false,
) {
    /** oneMKL, which additionally takes the sequential layer and has dynamic expansion switched off. */
    Mkl("MKL_Set_Num_Threads", "MKL_Get_Max_Threads"),

    /** OpenBLAS. */
    OpenBlas("openblas_set_num_threads", "openblas_get_num_threads"),

    /** AOCL, whose BLAS is BLIS. */
    Blis("bli_thread_set_num_threads", "bli_thread_get_num_threads"),

    /** ArmPL, which threads through OpenMP when it threads at all. */
    OpenMp("omp_set_num_threads", "omp_get_max_threads", absenceMeansSerial = true),

    /** Accelerate, which has no entry point either way and is held through the environment. */
    Environment(null, null),
    ;

    /** A parallel ArmPL build may hide its dependency's controls on Windows; absence cannot prove it serial. */
    fun permitsAbsentControl(candidate: String): Boolean {
        val fileName = candidate.replace('\\', '/').substringAfterLast('/').lowercase()
        return absenceMeansSerial && "_mp" !in fileName
    }
}

/**
 * The only lever Accelerate has for its thread count, and the value Koblas fixes it at.
 *
 * Accelerate exports no thread-count entry point, so unlike every other supported vendor it cannot be told how
 * many threads to use through a call, and cannot be asked afterwards. What it does read is this variable, which
 * is why the binding sets it in its own process before Accelerate performs any arithmetic rather than leaving
 * the requirement to whoever launched the process. Setting it is how the invariant is established; it is not a
 * configuration surface, and an inherited value is overwritten rather than honored.
 *
 * The consequence is recorded honestly: Accelerate reports [ThreadEvidence.Unconfirmed], because a variable
 * that can be set but not read back is not the same evidence as a count the library returns.
 */
internal const val ACCELERATE_THREAD_LIMIT: String = "VECLIB_MAXIMUM_THREADS"

/**
 * Whether a library's own configuration string says it was built with 64-bit BLAS integers.
 *
 * The ABI cannot be settled by calling the library. On this calling convention a 32-bit argument is passed in
 * the low half of a register whose upper half is zeroed, so an ILP64 routine reading 64 bits sees the same small
 * value an LP64 one does. A small dot probe can agree under both, while another ABI mismatch can crash the
 * process. What does distinguish them is what the build says about itself: OpenBLAS names its integer width
 * in the configuration string it hands back, so a library claiming a wide one under the unsuffixed CBLAS
 * names is rejected rather than trusted.
 *
 * A build that names no width here is not thereby accepted. BLIS answers the question directly instead, through
 * [BLIS_INTEGER_WIDTH], and a loader asks that too rather than reading silence as an LP64 answer. oneMKL's
 * interface is selected and verified separately before reading its version or making arithmetic probes.
 */
internal fun declaresWideIntegers(configuration: String): Boolean {
    val text = configuration.uppercase()
    return "USE64BITINT" in text || "INT64" in text || "ILP64" in text
}

/**
 * BLIS's own report of its BLAS integer width in bits, which is better evidence than its version string.
 *
 * AOCL is BLIS, and its version string is a bare number that says nothing about the ABI. The entry point is
 * absent from every other supported vendor, so a loader treats its absence as no answer rather than as an
 * LP64 one and falls back to [declaresWideIntegers].
 *
 * The interface width is the one asked for, not `bli_info_get_int_type_size`, which reports the internal
 * `gint_t` instead. A stock BLIS is built with 64-bit internal integers behind a 32-bit BLAS interface, so the
 * internal answer is 64 for an LP64 and an ILP64 build alike: asking it rejects every correct library and
 * distinguishes nothing.
 */
internal const val BLIS_INTEGER_WIDTH: String = "bli_info_get_blas_int_type_size"

/** The integer width in bits that the unsuffixed CBLAS symbols carry. */
internal const val LP64_INTEGER_BITS: Int = 32

/** oneMKL's `MKL_INTERFACE_LP64`, selecting the 32-bit integer interface of its dispatching runtime. */
internal const val MKL_LP64: Int = 0

/**
 * A known-answer check the library must pass before any caller reaches it.
 *
 * Small, exactly representable, and cheap: the products and sums here are integers a double holds without
 * rounding, so a correct implementation returns these values exactly and a mismatch is a real fault rather than
 * a tolerance question. It catches a partial or mismatched install that resolved every symbol but does not
 * compute, which symbol presence alone cannot.
 */
internal object AbiProbe {
    /** `x` for the dot product check. */
    val x: DoubleArray get() = doubleArrayOf(1.0, 2.0, 3.0, 4.0)

    /** `y` for the dot product check. */
    val y: DoubleArray get() = doubleArrayOf(1.0, 1.0, 1.0, 1.0)

    /** `xᵀ · y` for [x] and [y]. */
    const val DOT: Double = 10.0

    /** A 2x2 column-major identity. */
    val identity: DoubleArray get() = doubleArrayOf(1.0, 0.0, 0.0, 1.0)

    /** A 2x2 column-major operand; the identity product must return it unchanged. */
    val operand: DoubleArray get() = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
}

/** oneMKL's `MKL_THREADING_SEQUENTIAL`. */
internal const val MKL_SEQUENTIAL: Int = 1

/** The longest version string a binding reads. */
internal const val VERSION_BUFFER: Int = 256

/**
 * The raw calls a binding makes into a candidate library while admitting it, each by symbol name and each
 * answering null when the library does not export that symbol. [admit] owns the order they are made in.
 */
internal interface AdmissionCalls {
    /** Which vendor the candidate is. */
    val vendor: Vendor

    /** The file that opened. */
    val candidate: String

    /** The library's own version, read through [readVersion] on first use. */
    val version: String

    /** Calls `int name(int)`. */
    fun callIntToInt(name: String, argument: Int): Int?

    /** Calls `void name(int)`, doing nothing when it is absent. */
    fun callIntToUnit(name: String, argument: Int)

    /** Calls `int name(void)`. */
    fun callToInt(name: String): Int?

    /** Calls `void name(char *buffer, int size)` and reads the buffer back. */
    fun callIntoBuffer(name: String, size: Int): String?

    /** Calls `const char *name(void)` and reads the string, or null for a null pointer. */
    fun callToString(name: String): String?

    /** Sets a variable in this process's environment, replacing an inherited value. */
    fun setEnvironmentVariable(name: String, value: String)

    /** Applies the vendor's one-thread setter on the calling thread, even if it already did, and records it. */
    fun holdThisThread()

    /** Whether the library's own `ddot` and `dgemm` return the exact [AbiProbe] answers. */
    fun knownAnswersMatch(): Boolean
}

/**
 * Admits a candidate, returning how its single compute thread is known, or null to refuse it.
 *
 * oneMKL's integer interface is selected before any other oneMKL call, including the thread controls, the
 * version read and the probes. The thread configuration precedes the probes because a probe can resolve
 * oneMKL's threading layer, and the count is read back after them.
 */
internal fun AdmissionCalls.admit(): ThreadEvidence? {
    if (!establishIntegerInterface()) return null
    enforceSingleThread()
    if (!verifiedAbi()) return null
    return confirmSingleThread()
}

/**
 * Selects and verifies oneMKL's LP64 interface. The setter overrides an inherited `MKL_INTERFACE_LAYER` and
 * returns the selected interface, so a missing setter or a runtime already fixed to another interface fails.
 */
private fun AdmissionCalls.establishIntegerInterface(): Boolean =
    vendor != Vendor.OneMkl || callIntToInt("MKL_Set_Interface_Layer", MKL_LP64) == MKL_LP64

/**
 * Configures one compute thread before any arithmetic. oneMKL takes its sequential layer, which needs no OpenMP
 * runtime, and has dynamic expansion switched off. Accelerate has no entry point and is held through
 * [ACCELERATE_THREAD_LIMIT].
 */
private fun AdmissionCalls.enforceSingleThread() {
    if (vendor.threadControl == ThreadControl.Environment) setEnvironmentVariable(ACCELERATE_THREAD_LIMIT, "1")
    if (vendor.threadControl == ThreadControl.Mkl) {
        callIntToInt("MKL_Set_Threading_Layer", MKL_SEQUENTIAL)
        callIntToUnit("MKL_Set_Dynamic", 0)
    }
    // Unconditional: this thread may have held an earlier candidate of the same vendor that was then refused.
    holdThisThread()
}

/**
 * Whether the library matches the ABI Koblas binds and computes correctly through it. The integer width is
 * read from what the build says about itself, through [declaresWideIntegers] and [BLIS_INTEGER_WIDTH], since a
 * call cannot distinguish it; whether the library computes at all is settled by the known-answer probes.
 */
private fun AdmissionCalls.verifiedAbi(): Boolean {
    if (declaresWideIntegers(version)) return false
    val width = callToInt(BLIS_INTEGER_WIDTH)
    if (width != null && width != LP64_INTEGER_BITS) return false
    return knownAnswersMatch()
}

/** Reads back the thread count, refusing a library that still reports more than one. */
private fun AdmissionCalls.confirmSingleThread(): ThreadEvidence? {
    val control = vendor.threadControl
    // Accelerate, held through the environment and unable to answer. Declared, not discovered.
    val getter = control.getter ?: return ThreadEvidence.Unconfirmed
    val count = callToInt(getter)
    return when {
        // A build of a known vendor without its control is refused, except a declared serial build.
        count == null -> ThreadEvidence.Confirmed.takeIf { control.permitsAbsentControl(candidate) }

        count == 1 -> ThreadEvidence.Confirmed

        else -> null
    }
}

/** The library's own version string, or a note that it reports none. */
internal fun AdmissionCalls.readVersion(): String = callIntoBuffer("MKL_Get_Version_String", VERSION_BUFFER).nonBlank()
    ?: callToString("openblas_get_config").nonBlank()
    ?: callToString("bli_info_get_version_str").nonBlank()
    ?: if (vendor == Vendor.Accelerate) "system framework" else "unreported"

private fun String?.nonBlank(): String? = this?.trim()?.ifEmpty { null }
