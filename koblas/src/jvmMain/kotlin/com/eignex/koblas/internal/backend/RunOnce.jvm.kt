package com.eignex.koblas.internal.backend

/** The monitor [withRunOnceLock] holds. Private to this file, so nothing else can be waiting on it. */
private val runOnceMonitor = Any()

/** A JVM monitor, which the thread holding it may re-enter and another thread waits for. */
internal actual fun <T> withRunOnceLock(block: () -> T): T = synchronized(runOnceMonitor) { block() }
