package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import java.lang.invoke.MethodHandle
import java.lang.invoke.MethodHandles
import java.lang.invoke.MethodType
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/** JVM-only cost and outcome probe for the conservative packed-TRSM eligibility scan and bound. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class PackedTrsmEligibilityBenchmark {
    @Param("16", "32", "64")
    var order: Int = 16

    @Param("8", "32", "128")
    var panel: Int = 32

    @Param("eligible", "structural-zero", "overflow-bound")
    var scenario: String = "eligible"

    private lateinit var triangle: DenseMatrix
    private lateinit var rightHandSide: DenseMatrix
    private lateinit var supportsHandle: MethodHandle

    @Setup
    fun setup() {
        val rng = benchRng()
        supportsHandle = MethodHandles.lookup().findStatic(
            Class.forName("com.eignex.koblas.dense.PackedTriangularSolveKt"),
            "packedTrsmSupports",
            MethodType.methodType(
                Boolean::class.javaPrimitiveType,
                DenseMatrix::class.java,
                DenseMatrix::class.java,
                Boolean::class.javaPrimitiveType,
                Boolean::class.javaPrimitiveType,
            ),
        )
        triangle = dominantMatrix(order, rng)
        rightHandSide = randomMatrix(panel, order, rng)
        when (scenario) {
            "eligible" -> Unit
            "structural-zero" -> triangle[order - 1, order - 1] = 0.0
            "overflow-bound" -> rightHandSide[panel - 1, order - 1] = 1e308
            else -> error("unknown eligibility scenario $scenario")
        }
        val eligible = invokeSupports()
        check(eligible == (scenario == "eligible"))
        println(
            "resolved: packed-trsm-eligibility order=$order panel=$panel scenario=$scenario " +
                "eligible=$eligible fallback-rate=${if (eligible) 0 else 100}%",
        )
    }

    @Benchmark
    fun supports(): Boolean = invokeSupports()

    private fun invokeSupports(): Boolean = supportsHandle.invokeExact(
        triangle,
        rightHandSide,
        true,
        false,
    ) as Boolean
}
