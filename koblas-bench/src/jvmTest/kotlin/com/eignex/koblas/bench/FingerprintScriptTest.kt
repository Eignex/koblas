package com.eignex.koblas.bench

import com.eignex.lab.bench.WorkspaceChecksums
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
        writeExecutableScript(java, "printf partial\ntouch \"${'$'}HELPER_MARKER\"\nexit 1\n")
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
    fun `supported malformed filename bytes and newlines preserve the complete legacy output`() = fixture { root ->
        val source = Files.createDirectories(root.resolve("source"))
        Files.createDirectories(source.resolve("koblas-bench/build/lab/classes/com/eignex/koblas/bench"))
        Files.createFile(source.resolve(
            "koblas-bench/build/lab/classes/com/eignex/koblas/bench/PreparedFingerprints.class",
        ))
        val script = File("script-helpers.sh").absolutePath
        val classpath = helperClasspath()
        val body = """
            set -euo pipefail
            source "${'$'}1"
            cd "${'$'}2"
            fixture_error="${'$'}3/filename-error"
            trap 'printf "filename fixture failed at line %s\n" "${'$'}LINENO" >&2; cat "${'$'}fixture_error" >&2' ERR
            if ! (printf original > ${'$'}'bad\xff') 2>"${'$'}3/filename-error"; then
                # APFS refuses malformed UTF-8 names before either fingerprint path can inspect them.
                test "${'$'}(uname -s)" = Darwin
                LC_ALL=C grep -q 'Illegal byte sequence' "${'$'}3/filename-error"
            fi
            printf newline > ${'$'}'split\nname'
            printf ordinary > ordinary
            unset LAB_WORKSPACE_CACHE
            fingerprint "${'$'}2" sources archive > "${'$'}3/fresh"
            export LAB_WORKSPACE_CACHE="${'$'}3/cache"
            fingerprint "${'$'}2" sources archive > "${'$'}3/cached"
        """.trimIndent()
        runProcess(listOf(captureShell(), "-c", body, "fixture", script, source.toString(), root.toString()), mapOf(
            "KOBLAS_BENCH_JAVA" to Path.of(System.getProperty("java.home"), "bin", "java").toString(),
            "KOBLAS_BENCH_CLASSPATH" to classpath,
        ))
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
        return runProcess(listOf(
            captureShell(), "-c", "source \"${'$'}1\"; fingerprint \"${'$'}2\" sources archive",
            "fixture", script, source.toString(),
        ), mapOf(
            "LAB_WORKSPACE_CACHE" to source.parent.resolve("cache").toString(),
            "KOBLAS_BENCH_JAVA" to java.toString(),
            "KOBLAS_BENCH_CLASSPATH" to "fixture",
            "HELPER_MARKER" to marker.toString(),
        ))
    }

    private fun helperClasspath(): String =
        listOf(PreparedFingerprints::class.java, WorkspaceChecksums::class.java, Unit::class.java).map { type ->
            File(type.protectionDomain.codeSource.location.toURI()).path
        }.distinct().joinToString(File.pathSeparator)

    private fun digest(file: Path): String =
        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)))

    private fun fixture(block: (Path) -> Unit) {
        if (System.getProperty("os.name").startsWith("Windows")) return
        withTemporaryDirectory("koblas-script-fingerprint", block)
    }
}
