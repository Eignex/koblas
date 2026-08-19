package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.koblas
import com.eignex.koblas.registerBackend
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors

/**
 * Soaks concurrent backend registration, looking for a round where the strongest offer does not end up
 * holding the half.
 *
 * A seam takes its offer with a compare-and-set, so two threads offering at once cannot leave the weaker one
 * active. Catching a regression in that needs the offers to collide, which is a matter of luck per round, so
 * it wants far more rounds than a unit test can afford. Hence a task here rather than a test in the suite.
 *
 * Each round offers a band of priorities strictly above everything offered before it, so the expected winner
 * is known without clearing the registry between rounds.
 */
private class Ranked(override val name: String, override val priority: Int) :
    F64Blas by F64ReferenceLinearAlgebra {
    override val isPortable: Boolean get() = false
    override val isAvailable: Boolean get() = true
}

/**
 * Enough rounds that one run is decisive. Measured against the pre-compare-and-set registration, a weaker
 * offer held the half about once every ten thousand rounds, at four threads and at eight, so the default
 * sits an order of magnitude above that and costs about eight seconds.
 */
private const val DEFAULT_ROUNDS = 200_000

private const val DEFAULT_THREADS = 4

/** Priorities of one round sit this far apart, so a later round cannot be won by an earlier offer. */
private const val BAND = 100

public fun main(args: Array<String>) {
    val rounds = args.intArg("--rounds") ?: DEFAULT_ROUNDS
    val threads = args.intArg("--threads") ?: DEFAULT_THREADS
    println("stressing registration: rounds=$rounds threads=$threads")

    val pool = Executors.newFixedThreadPool(threads)
    var collisions = 0
    var losses = 0
    val startedAt = System.nanoTime()
    try {
        for (round in 1..rounds) {
            val base = round * BAND
            val offered = (1..threads).map { base + it }
            val gate = CyclicBarrier(threads)
            val winners = offered.map { priority ->
                pool.submit {
                    gate.await()
                    registerBackend(Ranked("p$priority", priority))
                    koblas.blas.priority
                }
            }.map { it.get() }
            // A round where every thread already saw the strongest cannot have raced, so it proves nothing.
            if (winners.any { it != offered.max() }) collisions++
            val active = koblas.blas.priority
            if (active != offered.max()) {
                losses++
                println("round $round: active priority $active, strongest offered ${offered.max()}")
            }
        }
    } finally {
        pool.shutdownNow()
    }

    val seconds = (System.nanoTime() - startedAt) / 1e9
    println("rounds=$rounds contended=$collisions lost=$losses in ${"%.1f".format(seconds)}s")
    if (losses > 0) {
        println("FAILED: a weaker offer held the half in $losses of $rounds rounds")
        kotlin.system.exitProcess(1)
    }
    println("OK: the strongest offer held the half in every round")
}

private fun Array<String>.intArg(name: String): Int? =
    firstOrNull { it.startsWith("$name=") }?.substringAfter('=')?.toIntOrNull()
