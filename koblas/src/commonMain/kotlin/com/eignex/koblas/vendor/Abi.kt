package com.eignex.koblas.vendor

/**
 * Whether the single compute thread the binding requires was confirmed against the library itself.
 *
 * This is evidence, not a setting. Nothing reads it to decide anything and there is no way to change what it
 * reports; a benchmark records it because a run described as single-threaded should say whether that was
 * checked or only asked for.
 */
public enum class ThreadEvidence {
    /** The library reported one compute thread after being held to it. */
    Confirmed,

    /**
     * The library exposes no thread-count entry point, so the requirement could not be read back.
     *
     * Accelerate is the case this exists for: it has no CBLAS thread-count symbol at all, and its only lever is
     * an environment variable read before the process starts, which cannot be observed from inside it.
     */
    Unconfirmed,
    ;

    /** How the evidence reads in a report. */
    public val label: String get() = if (this == Confirmed) "1-thread-confirmed" else "1-thread-unconfirmed"
}

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
 * The required entry points [exports] does not have.
 *
 * Separated from the loaders so the rejection rule can be tested without a library that is partial in exactly
 * the right way. Both platforms ask this question the same way and reject on the same answer.
 */
internal fun missingRequiredSymbols(exports: (String) -> Boolean): List<String> =
    BlasOperation.entries.filter { it.required && !exports(it.entryPoint) }.map { it.entryPoint }

/**
 * Whether a library's own configuration string says it was built with 64-bit BLAS integers.
 *
 * The ABI cannot be settled by calling the library. On this calling convention a 32-bit argument is passed in
 * the low half of a register whose upper half is zeroed, so an ILP64 routine reading 64 bits sees the same small
 * value an LP64 one does, and every probe small enough to be safe agrees under both. What does distinguish them
 * is what the build says about itself: OpenBLAS names its integer width in the configuration string it hands
 * back, so a library claiming a wide one under the unsuffixed CBLAS names is rejected rather than trusted.
 *
 * A build that names no width here is not thereby accepted. BLIS answers the question directly instead, through
 * [BLIS_INTEGER_WIDTH], and a loader asks that too rather than reading silence as an LP64 answer.
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
