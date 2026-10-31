package com.eignex.koblas.bench

/**
 * Runs one benchmark case in this process for a fixed time, so a sampling profiler attached to the JVM sees the
 * kernel frames. JMH measures in a forked JVM a recording of the parent never reaches.
 *
 * Arguments: the case line, the seconds to run, and optionally the engine mode (default `jvm-simd`). The runtime
 * is printed first, because a missing Vector API module leaves no SIMD engine and the profile would show a
 * different backend from the one asked for. A launcher that left the module out is answered by [relaunch].
 */
public fun main(args: Array<String>) {
    require(args.size in 2..3) { "usage: ProfileLoop CASE SECONDS [MODE]" }
    val vector = runCatching { Class.forName("jdk.incubator.vector.DoubleVector") }
    println(
        "runtime ${Runtime.version()} at ${System.getProperty("java.home")}; vector module " +
            vector.fold({ "resolved" }, { "missing (${it.javaClass.simpleName})" }) + "; arguments " +
            java.lang.management.ManagementFactory.getRuntimeMXBean().inputArguments,
    )
    if (vector.isFailure) kotlin.system.exitProcess(relaunch(args))
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

/**
 * Runs this loop again under the same Java with the Vector API module added, for a launcher that left it out,
 * recording the child's flight recording into `$OUT` beside the launcher's, which then holds only this wait.
 */
private fun relaunch(args: Array<String>): Int {
    val output = System.getenv("OUT") ?: "."
    val command = listOf(
        "${System.getProperty("java.home")}/bin/java", "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
        "-XX:StartFlightRecording=filename=$output/kernel.jfr,settings=profile,dumponexit=true",
        "-cp", System.getProperty("java.class.path"), "com.eignex.koblas.bench.ProfileLoopKt",
    ) + args
    return ProcessBuilder(command).inheritIO().start().waitFor()
}
