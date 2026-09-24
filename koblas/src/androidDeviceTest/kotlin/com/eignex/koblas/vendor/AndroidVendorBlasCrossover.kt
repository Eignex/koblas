// detekt exempts test files by path, and its list predates AGP's androidDeviceTest name.
@file:Suppress("UndocumentedPublicClass", "UndocumentedPublicFunction", "FunctionNaming", "MagicNumber")

package com.eignex.koblas.vendor

import android.system.Os
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter
import kotlin.test.Test
import kotlin.test.assertNotNull

/**
 * Times the portable engine against the bundled OpenBLAS on the device, to place the crossover.
 *
 * Not a test: it asserts nothing about speed and runs only when asked, with
 * `-Pandroid.testInstrumentationRunnerArguments.koblas.crossover=true`, because a normal run must stay fast.
 * Results go to logcat under [TAG], one line per case and arm.
 *
 * An application thread cannot be pinned to one core type from here, and this phone mixes two, so each
 * figure is the fastest of many samples rather than their mean: the fastest is the one that ran on a big
 * core without being moved, which is the only one comparable across arms.
 */
class AndroidVendorBlasCrossover {
    private val enabled: Boolean
        get() = InstrumentationRegistry.getArguments().getString("koblas.crossover") == "true"

    @Test
    fun `times the portable engine against the bundled library`() {
        if (!enabled) return
        // OpenBLAS reads its core override once, when the library loads, so it has to be in the environment
        // before anything in this process touches the binding; running this class alone guarantees that.
        InstrumentationRegistry.getArguments().getString("koblas.coretype")?.let { core ->
            Os.setenv("OPENBLAS_CORETYPE", core, true)
        }
        val vendor = assertNotNull(openBlas())
        val portable = BuiltinEngines.scalar
        val composed = KoblasEngine(
            ScalarVectorKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
            vendor,
            hostDense = vendor,
        )
        Log.i(TAG, "library ${vendor.version} core=${AndroidCblas.coreName()}")

        // The first case measured also pays for compiling the harness itself, so it runs once and is dropped.
        val warm = values(64, 9)
        report("warmup (discard)", mapOf("portable" to { sink += ScalarVectorKernels.dot(warm, 0, warm, 0, 64) }))

        for (n in listOf(8, 16, 32, 64, 128, 256, 1024, 4096)) {
            val x = values(n, 1)
            val y = values(n, 2)
            val vx = DenseVector.wrap(x)
            val vy = DenseVector.wrap(y)
            report(
                "dot n=$n",
                mapOf(
                    "portable" to { sink += ScalarVectorKernels.dot(x, 0, y, 0, n) },
                    "openblas" to { sink += vendor.dot(vx, vy) },
                ),
            )
        }

        // The shapes the issue names, then larger ones so the crossover is bracketed from both sides.
        val shapes = listOf(12 to 24, 24 to 48, 48 to 48, 96 to 96, 192 to 192)
        for ((rows, cols) in shapes) {
            val j = DenseMatrix.wrap(rows, cols, values(rows * cols, 3))
            val x = values(cols, 4)
            val r = values(rows, 5)
            val jx = DoubleArray(rows)
            val jtr = DoubleArray(cols)
            val jtj = DenseMatrix(cols, cols)
            val workspace = Workspace()
            val vx = DenseVector.wrap(x)
            val vr = DenseVector.wrap(r)
            val vjx = DenseVector.wrap(jx)
            val vjtr = DenseVector.wrap(jtr)

            report(
                "J*x ${rows}x$cols",
                mapOf(
                    "portable" to { portable.gemv(1.0, j, x, 0.0, jx, false, workspace) },
                    "openblas" to { vendor.gemv(1.0, j, false, vx, 0.0, vjx) },
                    "composed" to { composed.gemv(1.0, j, x, 0.0, jx, false, workspace) },
                ),
            )
            report(
                "J^T*r ${rows}x$cols",
                mapOf(
                    "portable" to { portable.gemv(1.0, j, r, 0.0, jtr, true, workspace) },
                    "openblas" to { vendor.gemv(1.0, j, true, vr, 0.0, vjtr) },
                    "composed" to { composed.gemv(1.0, j, r, 0.0, jtr, true, workspace) },
                ),
            )
            report(
                "J^T*J syrk ${rows}x$cols",
                mapOf(
                    "portable" to { portable.syrk(1.0, j, true, 0.0, jtj, true, workspace) },
                    "openblas" to { vendor.syrk(1.0, j, true, 0.0, jtj, MatrixStructure.SymmetricLower) },
                    "composed" to { composed.syrk(1.0, j, true, 0.0, jtj, true, workspace) },
                ),
            )
            report(
                "J^T*J gemm ${rows}x$cols",
                mapOf(
                    "portable" to { portable.gemm(1.0, j, true, j, false, 0.0, jtj, workspace) },
                    "openblas" to { vendor.gemm(1.0, j, true, j, false, 0.0, jtj) },
                    "composed" to { composed.gemm(1.0, j, true, j, false, 0.0, jtj, workspace) },
                ),
            )
        }
        Log.i(TAG, "done sink=$sink")
    }

    /** Times every arm of one case, interleaved per sample so drift falls on all of them alike. */
    private fun report(case: String, arms: Map<String, () -> Unit>) {
        val iterations = arms.mapValues { (_, body) -> calibrate(body) }
        val best = arms.mapValues { Long.MAX_VALUE }.toMutableMap()
        repeat(SAMPLES) {
            for ((arm, body) in arms) {
                val count = iterations.getValue(arm)
                val start = System.nanoTime()
                for (i in 0 until count) body()
                val perCall = (System.nanoTime() - start) / count
                if (perCall < best.getValue(arm)) best[arm] = perCall
            }
        }
        val line = best.entries.joinToString("  ") { (arm, ns) -> "$arm=${ns}ns" }
        Log.i(TAG, "$case  $line")
    }

    /** Warms [body] until the JIT has had its chance, then sizes one sample to about [SAMPLE_NANOS]. */
    private fun calibrate(body: () -> Unit): Int {
        val warmUntil = System.nanoTime() + WARMUP_NANOS
        var calls = 0
        while (System.nanoTime() < warmUntil) {
            body()
            calls++
        }
        val perCall = WARMUP_NANOS / calls.coerceAtLeast(1)
        return (SAMPLE_NANOS / perCall.coerceAtLeast(1)).toInt().coerceIn(1, MAX_ITERATIONS)
    }

    private fun values(size: Int, seed: Int): DoubleArray {
        var state = seed
        return DoubleArray(size) {
            state = state * 1_103_515_245 + 12_345
            ((state ushr 8) % 1000) / 250.0 - 2.0
        }
    }

    private companion object {
        const val TAG = "KoblasCrossover"
        const val SAMPLES = 25
        const val WARMUP_NANOS = 300_000_000L
        const val SAMPLE_NANOS = 5_000_000L
        const val MAX_ITERATIONS = 1_000_000

        /** Keeps the results live so the JIT cannot drop a call whose answer nobody reads. */
        @Volatile var sink: Double = 0.0
    }
}
