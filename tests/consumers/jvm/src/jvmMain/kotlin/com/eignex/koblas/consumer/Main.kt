package com.eignex.koblas.consumer

fun main(args: Array<String>) {
    ConsumerCheck.run(expectSimd = args.contains("simd"))
}
