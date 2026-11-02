package com.eignex.koblas.bench

import java.io.ByteArrayInputStream
import java.io.StringWriter
import java.nio.charset.CharacterCodingException
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.util.HexFormat
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class PreparedFingerprintsTest {
    @Test
    fun `empty input has no fingerprint lines or receipt`() = fixture { root ->
        val cache = root.resolve("cache")

        assertEquals("", fingerprint(root, emptyList(), cache))
        assertEquals(emptyList(), PreparedFingerprints.names(ByteArrayInputStream(byteArrayOf())).toList())
        assertFalse(Files.exists(cache))
    }

    @Test
    fun `platform escaped names require complete legacy hashing`() = fixture { root ->
        for (name in listOf("back\\slash", "carriage\rreturn", "-option")) {
            assertFailsWith<IllegalArgumentException> {
                fingerprint(root, listOf(name), root.resolve("cache"))
            }
        }
    }

    @Test
    fun `cached fingerprints preserve exact lines and detect membership and linked target changes`() = fixture { root ->
        val input = Files.createDirectories(root.resolve("inputs"))
        val first = Files.writeString(input.resolve("a file"), "original")
        val second = Files.writeString(input.resolve("z"), "second")
        Files.createSymbolicLink(input.resolve("linked"), first.fileName)
        val cache = root.resolve("cache")
        val names = listOf("inputs/a file", "inputs/linked", "inputs/z")
        val before = names.joinToString("") { name -> "${digest(root.resolve(name))} $name\n" }

        assertEquals(before, fingerprint(root, names, null))
        assertEquals(before, fingerprint(root, names, cache))
        assertEquals(before, fingerprint(root, names, cache))
        Files.delete(first)
        Files.move(second, first)
        val changed = listOf("inputs/a file", "inputs/linked", "inputs/missing")
        val expected = "${digest(first)} inputs/a file\n${digest(first)} inputs/linked\n"
        assertEquals(expected, fingerprint(root, changed, cache))
        assertEquals(expected, fingerprint(root, changed, null))
    }

    @Test
    fun `line inventory preserves carriage returns and unicode names`() {
        val names = listOf("a\rb", "space name", "\u03bb")
        val bytes = (names.joinToString("\n") + "\n").toByteArray()

        assertEquals(names, PreparedFingerprints.names(ByteArrayInputStream(bytes)).toList())
    }

    @Test
    fun `invalid inventory encoding fails rather than silently dropping files`() {
        assertFailsWith<CharacterCodingException> {
            PreparedFingerprints.names(ByteArrayInputStream(byteArrayOf(0xC3.toByte(), 0x28, 10))).toList()
        }
    }

    @Test
    fun `executable permission changes retain the same fingerprint bytes`() = fixture { root ->
        val file = Files.writeString(root.resolve("binary"), "executable bytes")
        val cache = root.resolve("cache")
        val before = fingerprint(root, listOf("binary"), cache)

        assertFalse(Files.isExecutable(file))
        check(file.toFile().setExecutable(true))

        assertEquals(before, fingerprint(root, listOf("binary"), cache))
        assertEquals(fingerprint(root, listOf("binary"), null), before)
    }

    private fun fingerprint(root: Path, paths: List<String>, cache: Path?): String = StringWriter().also {
        PreparedFingerprints.write(root, paths.asSequence(), it, cache)
    }.toString()

    private fun digest(file: Path): String =
        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)))

    private fun fixture(block: (Path) -> Unit) {
        val root = Files.createTempDirectory("koblas-fingerprints")
        try {
            if (Files.getFileStore(root).supportsFileAttributeView("unix")) block(root)
        } finally {
            root.toFile().deleteRecursively()
        }
    }
}
