@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.ReferenceLinearAlgebra
import kotlinx.cinterop.ExperimentalForeignApi
import platform.posix.getenv
import kotlin.random.Random
import kotlin.test.Test
import kotlin.time.Duration
import kotlin.time.TimeSource

/** Ad-hoc timing of the dlopen backend against the reference; opt in with KOBLAS_MICROBENCH=1. */
class MicrobenchTest {

    private fun bestOf(runs: Int, block: () -> Unit): Duration {
        var best = Duration.INFINITE
        repeat(runs) {
            val mark = TimeSource.Monotonic.markNow()
            block()
            val elapsed = mark.elapsedNow()
            if (elapsed < best) best = elapsed
        }
        return best
    }

    @Test
    fun microbench() {
        if (getenv("KOBLAS_MICROBENCH") == null) return
        for (n in intArrayOf(256, 1024)) {
            val rng = Random(20260728)
            val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            for (i in 0 until n) a[i, i] = a[i, i] + n
            val b = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val rhs = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val block = DenseMatrix.wrap(n, 8, DoubleArray(n * 8) { rng.nextDouble(-1.0, 1.0) })
            val backends: List<Pair<String, LinearAlgebra>> =
                listOf("cblas" to CblasLinearAlgebra(), "reference" to ReferenceLinearAlgebra)
            for ((label, la) in backends) {
                bestOf(1) { la.gemm(a, b) } // warmup
                val gemm = bestOf(3) { la.gemm(a, b) }
                val factor = bestOf(3) { la.factor(a) }
                val f = la.factor(a)
                val solve = bestOf(3) { la.solve(f, rhs) }
                val blockSolve = bestOf(3) { la.solve(f, block) }
                println("$label n=$n gemm=$gemm factor=$factor solve=$solve blockSolve=$blockSolve")
            }
        }
    }
}
