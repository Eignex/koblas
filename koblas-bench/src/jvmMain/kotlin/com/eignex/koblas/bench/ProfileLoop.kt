package com.eignex.koblas.bench

/**
 * Runs one benchmark case in this process for a fixed time, so a sampling profiler attached to the JVM sees the
 * kernel frames. JMH measures in a forked JVM a recording of the parent never reaches.
 *
 * Arguments: the case line, the seconds to run, and optionally the engine mode (default `jvm-simd`).
 */
public fun main(args: Array<String>) {
    require(args.size in 2..3) { "usage: ProfileLoop CASE SECONDS [MODE]" }
    val case = Cases.parse(args[0]).single()
    val seconds = args[1].toLong()
    val (engine, identity) = resolveEngine(args.getOrElse(2) { "jvm-simd" })
    val work = requireNotNull(denseWork(case, engine)) { "${case.id} has no dense work" }
    var sink = 0.0
    var calls = 0L
    val end = System.nanoTime() + seconds * 1_000_000_000
    while (System.nanoTime() < end) {
        repeat(1_000) { sink += work.run() }
        calls += 1_000
    }
    println("${case.id} on $identity: $calls calls, ${seconds * 1e9 / calls} ns per call, sink $sink")
}
