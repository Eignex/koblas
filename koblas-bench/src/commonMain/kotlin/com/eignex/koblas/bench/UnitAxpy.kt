package com.eignex.koblas.bench

import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine

/** Disjoint ordinary arrays matching the raw unit-multiplier update in Klause's cached-value restore. */
internal class UnitAxpy(size: Int) {
    val x = Fixtures.vector(size, 1)
    private val initial = Fixtures.vector(size, 2)
    val y = initial.copyOf()

    /** Check the complete destination before timing repeated accumulation without a destination reset. */
    fun verify(what: String, run: BenchmarkOperation) {
        val expected = initial.copyOf()
        repeat(2) {
            KoblasEngine.scalar.kernels.axpy(expected, 0, 1.0, x, 0, x.size)
            run()
        }
        DenseReference.check(expected, y, what)
        initial.copyInto(y)
    }
}
