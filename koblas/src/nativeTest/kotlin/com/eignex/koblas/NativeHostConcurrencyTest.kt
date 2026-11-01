// Workers are the only multi-thread primitive Kotlin/Native exposes today, so the obsolete marker is
// accepted rather than worked around; nothing production-side depends on it.
@file:OptIn(kotlin.native.concurrent.ObsoleteWorkersApi::class)
// detekt exempts backtick test names and undocumented test functions on the standard test paths only, and
// this is not one of them, so each case below carries its own KDoc.
@file:Suppress("FunctionNaming")

package com.eignex.koblas

import com.eignex.koblas.dense.ReferenceBlas
import kotlin.native.concurrent.TransferMode
import kotlin.native.concurrent.Worker
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * Checks single-threaded host computation with concurrent callers. Workers share read-only
 * operands and own their destinations and workspaces.
 *
 * The shapes are checked against the host policy so portable fallback cannot imply host coverage.
 * Unit product multipliers satisfy the host policy; distinct destination multipliers identify workers.
 */
class NativeHostConcurrencyTest {
    private class Task(val a: DenseMatrix, val b: DenseMatrix, val beta: Double)

    /** Whether a product of these extents is one the default hands to a library, asserted not assumed. */
    private fun assertHostEligible(rows: Int, columns: Int, depth: Int, what: String) {
        if (koblas.vendor == null) {
            println(
                "SKIPPED: no measured host profile or binding; $what exercised portable scheduling",
            )
        }
        assertTrue(
            StandardEngineProfiles.nativeLinuxOneMkl.rules.any {
                EngineImplementation.Gemm in it.implementations && it.matches(rows, columns, depth, true, false)
            },
            "$what stays portable",
        )
    }

    /** Eight workers, one shared pair of operands, and a destination and a workspace each. */
    @Test
    fun `concurrent dense calls over shared inputs give each caller the answer it would get alone`() {
        val rng = Random(31)
        val a = randomMatrix(ORDER, DEPTH, rng)
        val b = randomMatrix(DEPTH, ORDER, rng)
        val untouched = a.values.copyOf() to b.values.copyOf()
        assertHostEligible(ORDER, ORDER, DEPTH, "the concurrent product")

        val futures = (0 until WORKERS).map { index ->
            val worker = Worker.start()
            worker to worker.execute(TransferMode.SAFE, { Task(a, b, betaFor(index)) }) { task ->
                // Every worker owns these two; nothing below them is shared but the read-only operands.
                val destination = DenseMatrix.wrap(
                    task.a.rows,
                    task.b.cols,
                    DoubleArray(task.a.rows * task.b.cols) { 1.0 },
                )
                val workspace = Workspace()
                repeat(REPEATS) {
                    destination.values.fill(1.0)
                    koblas.gemm(1.0, task.a, false, task.b, false, task.beta, destination, workspace)
                }
                task.beta to destination.values
            }
        }

        for ((worker, future) in futures) {
            val (beta, actual) = future.result
            val expected = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 1.0 })
            ReferenceBlas.gemm(1.0, a, false, b, false, beta, expected)
            assertClose(expected.values, actual, "concurrent gemm beta=$beta", TOLERANCE)
            worker.requestTermination().result
        }
        assertContentEquals(untouched.first, a.values, "a shared operand was written")
        assertContentEquals(untouched.second, b.values, "a shared operand was written")
    }

    /**
     * The staging path is the one that borrows, so a workspace shared by accident or a buffer retained
     * across calls shows up here as one worker reading another's staging.
     */
    @Test
    fun `concurrent calls that stage an alias keep their own scratch`() {
        val rng = Random(32)
        val square = wellConditioned(ORDER, rng)
        val expected = DenseMatrix.zero(ORDER, ORDER)
        ReferenceBlas.gemm(1.0, square, false, square, false, 0.0, expected)
        assertHostEligible(ORDER, ORDER, ORDER, "the concurrent staged product")

        val futures = (0 until WORKERS).map {
            val worker = Worker.start()
            worker to worker.execute(TransferMode.SAFE, { square }) { operand ->
                val workspace = Workspace()
                var last = DoubleArray(0)
                repeat(REPEATS) {
                    val shared = operand.copyOf()
                    koblas.gemm(1.0, shared, false, shared, false, 0.0, shared, workspace)
                    last = shared.values
                }
                last
            }
        }

        for ((worker, future) in futures) {
            assertClose(expected.values, future.result, "concurrent staged gemm", TOLERANCE)
            worker.requestTermination().result
        }
    }

    /** The library the cases above reached reports one compute thread, or exports no way to be asked. */
    @Test
    fun `the composed default runs on a library held to one compute thread`() {
        val vendor = koblas.vendor ?: return println(
            "SKIPPED: no measured host profile or binding; host thread configuration was not verified",
        )

        assertTrue(
            vendor.threadEvidence.label.startsWith("1-thread-"),
            "the default composed a library that was not held to one compute thread",
        )
    }

    /** A destination multiplier that is this worker's own, so no two workers expect the same answer. */
    private fun betaFor(index: Int): Double = 0.5 + index

    private companion object {
        const val WORKERS = 8
        const val REPEATS = 50

        /** With [DEPTH], past the policy's size, so the product below is one the default hands over. */
        const val ORDER = 24
        const val DEPTH = 16

        /** Loose enough for the library's own accumulation order. */
        const val TOLERANCE = 1e-9
    }
}
