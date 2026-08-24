package com.eignex.koblas.internal.host

import java.lang.ref.Cleaner

/**
 * The cleaner every koblas binding registers a native free with, one for the whole process.
 *
 * A cleaner owns a thread, so one per factorization type would be one thread per type for work that is a
 * handful of frees. Public because the bindings that register with it ship as their own artifacts.
 */
public val nativeCleaner: Cleaner = Cleaner.create()
