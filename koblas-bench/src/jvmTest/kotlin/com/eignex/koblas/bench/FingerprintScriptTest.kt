package com.eignex.koblas.bench

import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.util.HexFormat
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FingerprintScriptTest {
    @Test
    fun `a failed helper discards partial lines before complete fresh hashing`() = fixture { root ->
        val marker = root.resolve("called")
        val java = root.resolve("java")
        Files.writeString(java, "#!/usr/bin/env bash\nprintf partial\ntouch \"${'$'}HELPER_MARKER\"\nexit 1\n")
        check(java.toFile().setExecutable(true))
        val source = Files.createDirectories(root.resolve("source"))
        val input = Files.writeString(source.resolve("a file"), "original")
        Files.createDirectories(source.resolve("koblas-bench/build/lab/classes/com/eignex/koblas/bench"))
        Files.createFile(source.resolve(
            "koblas-bench/build/lab/classes/com/eignex/koblas/bench/PreparedFingerprints.class",
        ))

        val result = fingerprint(source, java, marker)

        assertEquals("${digest(input)} a file\n", result)
        assertTrue(Files.exists(marker))
    }

    @Test
    fun `malformed filename bytes and newlines fall back to the complete legacy output`() = fixture { root ->
        val source = Files.createDirectories(root.resolve("source"))
        Files.createDirectories(source.resolve("koblas-bench/build/lab/classes/com/eignex/koblas/bench"))
        Files.createFile(source.resolve(
            "koblas-bench/build/lab/classes/com/eignex/koblas/bench/PreparedFingerprints.class",
        ))
        val script = File("script-helpers.sh").absolutePath
        val classpath = listOf(PreparedFingerprints::class.java, Unit::class.java).map { type ->
            File(type.protectionDomain.codeSource.location.toURI()).path
        }.distinct().joinToString(File.pathSeparator)
        val body = """
            set -euo pipefail
            source "${'$'}1"
            cd "${'$'}2"
            printf original > ${'$'}'bad\xff'
            printf newline > ${'$'}'split\nname'
            unset LAB_WORKSPACE_CACHE
            fingerprint "${'$'}2" sources archive > "${'$'}3/fresh"
            export LAB_WORKSPACE_CACHE="${'$'}3/cache"
            fingerprint "${'$'}2" sources archive > "${'$'}3/cached"
        """.trimIndent()
        val process = ProcessBuilder(captureShell(), "-c", body, "fixture", script, source.toString(), root.toString())
            .redirectErrorStream(true).apply {
                environment()["KOBLAS_BENCH_JAVA"] = Path.of(System.getProperty("java.home"), "bin", "java").toString()
                environment()["KOBLAS_BENCH_CLASSPATH"] = classpath
            }.start()
        val log = process.inputStream.bufferedReader().readText()

        assertEquals(0, process.waitFor(), log)
        assertTrue(Files.size(root.resolve("fresh")) > 64)
        assertContentEquals(Files.readAllBytes(root.resolve("fresh")), Files.readAllBytes(root.resolve("cached")))
    }

    @Test
    fun `a missing prepared helper retains complete fresh fingerprints`() = fixture { root ->
        val source = Files.createDirectories(root.resolve("source"))
        val input = Files.writeString(source.resolve("binary"), "original")
        check(input.toFile().setExecutable(true))

        val result = fingerprint(source, root.resolve("missing-java"), root.resolve("unused"))

        assertEquals("${digest(input)} binary\n", result)
    }

    private fun fingerprint(source: Path, java: Path, marker: Path): String {
        val script = File("script-helpers.sh").absolutePath
        val process = ProcessBuilder(
            captureShell(), "-c", "source \"${'$'}1\"; fingerprint \"${'$'}2\" sources archive",
            "fixture", script, source.toString(),
        )
            .redirectErrorStream(true).apply {
                environment()["LAB_WORKSPACE_CACHE"] = source.parent.resolve("cache").toString()
                environment()["KOBLAS_BENCH_JAVA"] = java.toString()
                environment()["KOBLAS_BENCH_CLASSPATH"] = "fixture"
                environment()["HELPER_MARKER"] = marker.toString()
            }.start()
        val result = process.inputStream.bufferedReader().readText()
        assertEquals(0, process.waitFor(), result)
        return result
    }

    private fun digest(file: Path): String =
        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)))

    private fun fixture(block: (Path) -> Unit) {
        if (System.getProperty("os.name").startsWith("Windows")) return
        val root = Files.createTempDirectory("koblas-script-fingerprint")
        try { block(root) } finally { root.toFile().deleteRecursively() }
    }
}
