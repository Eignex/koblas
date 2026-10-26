package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Workspace
import java.lang.management.ManagementFactory
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Finite strict-SPD workload adapted from Kumulant's Cholesky.kt at
 * 5014410198c53bc7bac3611cffa84353b595291e (Apache-2.0). Policy stays in the consumer;
 * this benchmark compares only its blocked factorization with copied or borrowed trailing updates.
 */
internal object CholeskyViewBenchmark {
    @JvmStatic
    fun main(args: Array<String>) {
        val orders = if (args.isEmpty()) listOf(65, 128, 256, 512) else args.map(String::toInt)
        val allocations = ManagementFactory.getThreadMXBean() as com.sun.management.ThreadMXBean
        check(allocations.isThreadAllocatedMemorySupported)
        allocations.isThreadAllocatedMemoryEnabled = true
        println("arm,order,path,median_ns_per_factorization,bytes_per_factorization")
        for ((arm, engine) in listOfNotNull("jvm-scalar" to BuiltinEngines.scalar, BuiltinEngines.simd?.let { "jvm-simd" to it })) {
            for (n in orders) {
                require(n > 0)
                val input = fixture(n)
                val expected = DenseMatrix.zero(n)
                val actual = DenseMatrix.zero(n)
                factor(input, expected, engine, Workspace(), false)
                factor(input, actual, engine, Workspace(), true)
                DenseReference.check(expected.values, actual.values, "Cholesky view order $n")
                verifyFactor(input, actual)
                val workspaces = arrayOf(Workspace(), Workspace())
                val outputs = arrayOf(DenseMatrix.zero(n), DenseMatrix.zero(n))
                // Alternate both paths so compilation and temperature do not belong to just one arm.
                repeat(100) { for (path in 0..1) factor(input, outputs[path], engine, workspaces[path], path == 1) }
                val times = Array(2) { LongArray(9) }
                val bytes = Array(2) { LongArray(9) }
                val thread = Thread.currentThread().threadId()
                for (sample in 0 until 9) for (turn in 0..1) {
                    val path = (sample + turn) % 2
                    val before = allocations.getThreadAllocatedBytes(thread)
                    val start = System.nanoTime()
                    repeat(10) { factor(input, outputs[path], engine, workspaces[path], path == 1) }
                    times[path][sample] = (System.nanoTime() - start) / 10
                    bytes[path][sample] = (allocations.getThreadAllocatedBytes(thread) - before) / 10
                }
                for (path in 0..1) {
                    println("$arm,$n,${if (path == 1) "view" else "copy"},${times[path].sorted()[4]},${bytes[path].sorted()[4]}")
                    verifyFactor(input, outputs[path])
                }
            }
        }
    }

    private fun fixture(n: Int): DenseMatrix {
        val a = DenseMatrix.zero(n)
        for (j in 0 until n) for (i in j until n) {
            val entry = if (i == j) n.toDouble() else ((i * 3 + j * 7) % 17 - 8) / 32.0
            a[i, j] = entry
            a[j, i] = entry
        }
        return a
    }

    private fun verifyFactor(input: DenseMatrix, factor: DenseMatrix) {
        for (j in 0 until input.cols) for (i in j until input.rows) {
            var value = 0.0
            for (p in 0..j) value += factor[i, p] * factor[j, p]
            check(abs(value - input[i, j]) <= 1e-10 * maxOf(1.0, abs(input[i, j])))
        }
    }

    private fun factor(input: DenseMatrix, out: DenseMatrix, engine: KoblasEngine, workspace: Workspace, views: Boolean) {
        val n = input.rows
        val values = out.values
        for (j in 0 until n) {
            values.fill(0.0, j * n, j * n + j)
            input.values.copyInto(values, j + j * n, j + j * n, (j + 1) * n)
        }
        val block = if (n <= 64) n else min(64, n / 4)
        var start = 0
        while (start < n) {
            val width = min(block, n - start)
            for (j in start until start + width) {
                val base = j + j * n
                for (p in start until j) {
                    val coefficient = values[j + p * n]
                    if (coefficient != 0.0) engine.vectorKernels.axpy(values, base, -coefficient, values, j + p * n, n - j)
                }
                check(values[base] > 0.0)
                values[base] = sqrt(values[base])
                for (i in base + 1 until base + n - j) values[i] /= values[base]
            }
            // The finite branch also scans each completed panel before its BLAS update in the consumer.
            for (j in start until start + width) for (i in j until n) check(values[i + j * n].isFinite())
            val first = start + width
            val height = n - first
            if (height > 0) {
                if (views) updateWithViews(out, first, start, height, width, engine, workspace)
                else updateWithCopies(out, first, start, height, width, engine, workspace)
            }
            start += width
        }
    }

    private fun updateWithViews(
        out: DenseMatrix, first: Int, start: Int, height: Int, width: Int, engine: KoblasEngine, workspace: Workspace,
    ) {
        val panel = out.view(first, start, height, width)
        val trailing = out.view(first, first, height, height)
        engine.syrk(-1.0, panel, false, 1.0, trailing, workspace = workspace)
    }

    private fun updateWithCopies(
        out: DenseMatrix, first: Int, start: Int, height: Int, width: Int, engine: KoblasEngine, workspace: Workspace,
    ) {
        val panel = block(out, first, start, height, width)
        val trailing = block(out, first, first, height, height)
        engine.syrk(-1.0, panel, false, 1.0, trailing, workspace = workspace)
        for (j in 0 until height) {
            trailing.values.copyInto(out.values, first + (first + j) * out.rows, j * height, (j + 1) * height)
        }
    }

    private fun block(a: DenseMatrix, row: Int, col: Int, rows: Int, cols: Int): DenseMatrix {
        val result = DenseMatrix.zero(rows, cols)
        for (j in 0 until cols) {
            val from = row + (col + j) * a.rows
            a.values.copyInto(result.values, j * rows, from, from + rows)
        }
        return result
    }
}
