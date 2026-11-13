package com.eignex.koblas.bench

/** A primitive result keeps timed callbacks from boxing when a compiler leaves them outlined. */
internal fun interface BenchmarkOperation {
    operator fun invoke(): Double
}

/** The timed callback representation, part of capture identity and regression controls. */
internal const val BENCHMARK_CALLBACK_ABI = "primitive-double-v1"
