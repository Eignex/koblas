package com.eignex.koblas.engine

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
