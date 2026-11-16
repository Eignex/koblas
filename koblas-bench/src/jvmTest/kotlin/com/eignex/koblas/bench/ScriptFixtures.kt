package com.eignex.koblas.bench

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.assertEquals

internal fun withTemporaryDirectory(prefix: String, block: (Path) -> Unit) {
    val root = Files.createTempDirectory(prefix)
    try {
        block(root)
    } finally {
        root.toFile().deleteRecursively()
    }
}

internal fun writeExecutableScript(path: Path, body: String) {
    Files.writeString(path, "#!/usr/bin/env bash\nset -eu\n${body.trimIndent()}\n")
    check(path.toFile().setExecutable(true))
}

internal fun runProcess(
    arguments: List<String>,
    environment: Map<String, String> = emptyMap(),
    expectedExit: Int = 0,
): String {
    val process = ProcessBuilder(arguments).redirectErrorStream(true).apply {
        environment().putAll(environment)
    }.start()
    val output = process.inputStream.bufferedReader().use { it.readText() }
    assertEquals(expectedExit, process.waitFor(), output)
    return output
}
