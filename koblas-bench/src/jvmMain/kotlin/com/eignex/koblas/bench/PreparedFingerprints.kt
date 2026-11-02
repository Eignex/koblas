package com.eignex.koblas.bench

import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.io.Writer
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Files
import java.nio.file.Path

internal object PreparedFingerprints {
    @JvmStatic
    fun main(args: Array<String>) {
        require(args.size == 1) { "Expected the captured tree" }
        System.out.writer(UTF_8).buffered().use { output ->
            write(Path.of(args.single()), names(System.`in`), output)
        }
    }

    fun write(root: Path, paths: Sequence<String>, output: Writer, cache: Path? = WorkspaceChecksums.directory()) {
        WorkspaceChecksums(root, cache).use { checksums ->
            for (name in paths) {
                // Legacy checksum tools escape these names differently on different hosts.
                require('\\' !in name && '\r' !in name && !name.startsWith('-')) {
                    "Filename requires the platform checksum tool"
                }
                val file = root.resolve(name)
                if (Files.isRegularFile(file)) {
                    output.append(checksums.checksum(file)).append(' ').append(name).append('\n')
                }
            }
        }
    }

    internal fun names(input: InputStream): Sequence<String> = sequence {
        val decoder = UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
        val line = ByteArrayOutputStream()
        val buffered = input.buffered()
        while (true) {
            val next = buffered.read()
            if (next == -1) {
                if (line.size() > 0) yield(decoder.decode(ByteBuffer.wrap(line.toByteArray())).toString())
                break
            }
            if (next == '\n'.code) {
                yield(decoder.decode(ByteBuffer.wrap(line.toByteArray())).toString())
                line.reset()
            } else {
                line.write(next)
            }
        }
    }
}
