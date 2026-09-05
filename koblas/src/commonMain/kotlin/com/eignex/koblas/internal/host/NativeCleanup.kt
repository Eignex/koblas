package com.eignex.koblas.internal.host

/** A registered native release, run when the caller asks for it rather than when the owner goes away. */
internal fun interface NativeCleanup {
    /** Releases now, through the same lifecycle the cleaner would have used. Idempotent. */
    fun clean()
}

/**
 * Registers [lifecycle]'s release to run once [owner] is unreachable, and hands back the handle that runs it
 * early.
 *
 * The JVM has a process-wide `Cleaner` that tracks [owner]; Kotlin/Native attaches its cleaner to the handle
 * itself, which the owner holds, so the two fire at the same point for the same reason.
 */
internal expect fun registerNativeCleanup(owner: Any, lifecycle: NativeResourceLifecycle): NativeCleanup
