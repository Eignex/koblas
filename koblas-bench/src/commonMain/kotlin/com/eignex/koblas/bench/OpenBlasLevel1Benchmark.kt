package com.eignex.koblas.bench

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.rotg
import com.eignex.koblas.rotmg
import kotlinx.benchmark.*

/** Direct external CBLAS level-1 counterparts. Non-BLAS sum and ssqd deliberately have no row here. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class ExternalDenseLevel1Benchmark {
    @Param("2", "4", "8", "16", "31", "32", "33", "63", "64", "65", "128", "256", "1024", "4096")
    var len: Int = 0

    @Param(OPENBLAS_BACKEND, ONEMKL_BACKEND)
    var comparator: String = OPENBLAS_BACKEND

    private lateinit var openblas: DenseComparator
    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray
    private lateinit var modifiedRotation: ModifiedGivens
    private var c: Double = 0.0
    private var s: Double = 0.0

    @Setup
    fun setup() {
        openblas = checkNotNull(if (comparator == OPENBLAS_BACKEND) openBlasComparator() else oneMklDenseComparator()) {
            "the benchmark-only $comparator comparator is unavailable"
        }
        check(openblas.identity.startsWith(comparator)) { "external arm resolved to ${openblas.identity}" }
        check(openblas.threading == "1 thread") { "$comparator is not explicitly single-threaded" }
        val rng = benchRng()
        x = randomVector(len, rng)
        y = randomVector(len, rng)
        modifiedRotation = rotmg(1.0, 1.0, 1.0, NEAR_UNIT_SCALE - 1.0)
        val rotation = rotg(3.0, 4.0)
        c = rotation.c
        s = rotation.s
        println("resolved: arm=$comparator level=1 dense=${openblas.identity} threading=${openblas.threading}")
        reportAllocatingWorkload("level1/$comparator/ffi", "benchmark foreign-function boundary wrappers")
        reportAllocatingWorkload("level1/$comparator/rotm", "BLAS parameter block materialization")
    }

    @Benchmark fun dot(): Double = openblas.dot(x, y)
    @Benchmark fun axpy() = openblas.axpy(NEAR_UNIT_SCALE, x, y)
    @Benchmark fun scale() = openblas.scale(NEAR_UNIT_SCALE, x)
    @Benchmark fun nrm2(): Double = openblas.nrm2(x)
    @Benchmark fun asum(): Double = openblas.asum(x)
    @Benchmark fun swap() = openblas.swap(x, y)
    @Benchmark fun rotm() = openblas.rotm(x, y, modifiedRotation)
    @Benchmark fun rot() = openblas.rot(x, y, c, s)
}

/** Four CBLAS ddot calls: a useful alternative comparison, not direct dot4 kernel parity. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class ExternalDenseLevel1CompositionBenchmark {
    @Param("32", "64", "65", "256", "4096")
    var len: Int = 0

    @Param(OPENBLAS_BACKEND, ONEMKL_BACKEND)
    var comparator: String = OPENBLAS_BACKEND

    private lateinit var openblas: DenseComparator
    private lateinit var shared: DoubleArray
    private lateinit var vectors: Array<DoubleArray>
    private val out = DoubleArray(4)

    @Setup
    fun setup() {
        openblas = checkNotNull(if (comparator == OPENBLAS_BACKEND) openBlasComparator() else oneMklDenseComparator()) {
            "the benchmark-only $comparator comparator is unavailable"
        }
        val rng = benchRng()
        shared = randomVector(len, rng)
        vectors = Array(4) { randomVector(len, rng) }
        println("resolved: arm=$comparator level=1 coverage=composition operation=dot4 calls=4")
    }

    @Benchmark
    fun dot4AsFourDots(): DoubleArray {
        for (i in 0 until 4) out[i] = openblas.dot(vectors[i], shared)
        return out
    }
}
