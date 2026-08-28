package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.koblas
import com.eignex.koblas.registerBackend
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors

private class Ranked(override val name: String, override val priority: Int) :
    F64Blas by F64ReferenceLinearAlgebra {
    override val isPortable: Boolean get() = false
    override val isAvailable: Boolean get() = true
}

private const val DEFAULT_ROUNDS = 200_000

private const val DEFAULT_THREADS = 4

private const val BAND = 100

fun main(args: Array<String>) {
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
