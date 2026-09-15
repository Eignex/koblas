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
    VendorOperation.entries.filter { it.required && !exports(it.entryPoint) }.map { it.entryPoint }

/**
 * Whether a library's own configuration string says it was built with 64-bit BLAS integers.
 *
 * The ABI cannot be settled by calling the library. On this calling convention a 32-bit argument is passed in
 * the low half of a register whose upper half is zeroed, so an ILP64 routine reading 64 bits sees the same small
 * value an LP64 one does, and every probe small enough to be safe agrees under both. What does distinguish them
 * is what the build says about itself: OpenBLAS and BLIS both advertise a 64-bit integer build in the string
 * they hand back, so a library claiming one under the unsuffixed CBLAS names is rejected rather than trusted.
 */
internal fun declaresWideIntegers(configuration: String): Boolean {
    val text = configuration.uppercase()
    return "USE64BITINT" in text || "INT64" in text || "ILP64" in text
}

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
