package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*

import java.nio.file.Files
import java.nio.file.Path

/** Development inputs, generated independently of benchmark results. */
internal object CalibrationCases {
    private val vectors = listOf("dot", "copy", "sum", "nrm2", "iamax", "asum", "axpy", "scal", "swap", "rot")
    fun supplement(): List<String> = buildList {
        for ((m, n) in listOf(16 to 512, 512 to 16, 63 to 257, 257 to 63)) {
            add("gemv+${m}x$n+uniform"); add("gemv+${m}x$n+uniform+transA=T"); add("ger+${m}x$n+uniform")
        }
        for (n in listOf(64, 128, 256, 512)) for (transpose in listOf("", "+transA=T"))
            add("gemm+${n}x${n}x$n+uniform+layout=borrowed$transpose")
        for ((m, n) in listOf(16 to 128, 128 to 16, 64 to 256, 256 to 64, 512 to 512)) {
            add("symm+${m}x$n+uniform+side=R+uplo=L")
            for (op in listOf("trsm", "trmm")) add("$op+${m}x$n+triangular+side=R+uplo=L+transA=N+diag=N")
        }
        for (shape in listOf("64x256x128", "128x256x128", "256x128x128", "256x512x256", "512x256x256"))
            for (layout in listOf("", "+layout=borrowed")) add("gemm+$shape+uniform$layout")
    }
    fun tail(): List<String> = buildList {
        for (op in vectors) for (n in listOf(2097152, 4194304))
            for (stride in listOf("", "+stride=2", "+stride=-1", "+stride=-2")) add("$op+$n+uniform$stride")
        for (op in vectors) { add("$op+8388608+uniform"); add("$op+4096+uniform+stride=256") }
        for (n in listOf(8192, 65536)) for (op in indexed) add("$op+${n * 4}+sparse-uniform+density=0.25")
    }
    fun nativeTail(): List<String> = vectors.map { "$it+6291456+uniform" } + buildList {
        for (op in vectors) for (n in listOf(6291456, 8388608)) for (stride in listOf(2, -1, -2))
            add("$op+$n+uniform+stride=$stride")
    }
    private val indexed = listOf("spdot", "spdot-sparse", "spaxpy", "spscatter", "spgather", "spgather-zero", "spnrm2-indexed")
    fun training(native: Boolean = false): List<String> = buildList {
        for (op in vectors) for (n in listOf(2, 4, 8, 16, 32, 64, 96, 128, 256, 1024, 65536, 1000000))
            for (stride in listOf("", "+stride=2", "+stride=-1", "+stride=-2")) add("$op+$n+uniform$stride")
        for (n in listOf(8, 16, 32, 64, 128, 256, 512)) {
            for (op in listOf("symv", "syr", "syr2")) add("$op+$n+uniform+uplo=L")
            for (op in listOf("trmv", "trsv")) for (transpose in listOf("N", "T"))
                add("$op+$n+triangular+uplo=L+transA=$transpose+diag=N")
            add("gemv+${n}x$n+uniform"); add("gemv+${n}x$n+uniform+transA=T"); add("ger+${n}x$n+uniform")
            add("gemm+${n}x${n}x$n+uniform"); add("symm+${n}x$n+uniform+side=L+uplo=L")
            add("gemmt+${n}x$n+uniform+uplo=L+transA=N+transB=N")
            for (op in listOf("syrk", "syr2k")) for (transpose in listOf("N", "T"))
                add("$op+${n}x$n+uniform+uplo=L+transA=$transpose")
            for (op in listOf("trsm", "trmm")) add("$op+${n}x$n+triangular+side=L+uplo=L+transA=N+diag=N")
        }
        for (shape in listOf("16x128x64", "128x16x64", "31x65x129") + listOf(256, 512, 1024, 2048, 8192).map { "16x16x$it" })
            for (layout in listOf("", "+layout=borrowed")) add("gemm+$shape+uniform$layout")
        for (n in listOf(2, 4, 8, 16, 32, 64, 256, 1024)) for (op in indexed)
            add("$op+${n * 4}+sparse-uniform+density=0.25")
        addAll(supplement()); addAll(tail()); if (native) addAll(nativeTail())
    }
    fun holdout(native: Boolean = false): List<String> = readCases(generateSequence(Path.of("").toAbsolutePath()) { it.parent }.first { Files.isDirectory(it.resolve("koblas-bench/lab")) }.resolve("koblas-bench/cases-addressed.txt")) + buildList {
        addAll(listOf("dot+3+uniform", "dot+3+uniform+stride=2", "axpy+3+uniform", "sum+3+uniform",
            "gemv+257x131+uniform+transA=T", "ger+131x257+uniform", "gemm+129x257x193+uniform+layout=borrowed",
            "gemm+257x129x193+uniform+layout=borrowed+transA=T", "gemm+257x257x257+uniform+layout=borrowed",
            "gemm+384x384x384+uniform+layout=borrowed", "gemm+512x257x129+uniform+layout=borrowed+transB=T",
            "symm+257x129+uniform+side=R+uplo=U", "trsm+257x129+triangular+side=R+uplo=U+transA=T+diag=N",
            "trmm+257x129+triangular+side=R+uplo=U+transA=T+diag=N", "spdot+524288+sparse-uniform+density=0.25",
            "gemmt+129x65+uniform+uplo=L+transA=N+transB=N", "symv+17+uniform+uplo=U"))
        for (op in listOf("dot", "axpy")) {
            add("$op+8388609+uniform"); add("$op+4194305+uniform+stride=${if (op == "dot") 2 else -2}")
            add("$op+4097+uniform+stride=257")
            if (native) for (stride in listOf(2, -1, -2)) add("$op+8388609+uniform+stride=$stride")
        }
        for (transpose in listOf("N", "T")) add("trmv+17+triangular+uplo=L+transA=$transpose+diag=N")
    }
    fun write(directory: Path) {
        Files.createDirectories(directory)
        val training = training()
        for ((name, cases) in mapOf("training" to training, "native-training" to training(true),
            "holdout" to holdout(), "native-holdout" to holdout(true), "supplement" to supplement(),
            "tail" to tail(), "native-tail" to nativeTail(), "native-timing" to listOf(
                "dot+64+uniform", "dot+1000000+uniform", "copy+1000000+uniform+stride=2", "axpy+64+uniform+stride=-2",
                "gemm+128x128x128+uniform", "gemm+16x16x2048+uniform", "trsm+128x128+triangular+side=L+uplo=L+transA=N+diag=N")))
            Files.writeString(directory.resolve("$name.txt"), cases.joinToString("\n", postfix = "\n"))
    }
}

internal fun readCases(path: Path): List<String> = Files.readAllLines(path).map(String::trim).filter { it.isNotEmpty() && !it.startsWith('#') }
