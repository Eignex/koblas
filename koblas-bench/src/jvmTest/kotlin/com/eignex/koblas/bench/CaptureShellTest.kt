package com.eignex.koblas.bench

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class CaptureShellTest {
    @Test
    fun `windows shell selection bypasses the wsl launcher and finds git cmd installations`() {
        val root = Files.createTempDirectory("koblas-shell")
        try {
            val system = Files.createDirectories(root.resolve("Windows/System32"))
            Files.createFile(system.resolve("bash.exe"))
            val git = Files.createDirectories(root.resolve("Git/cmd"))
            Files.createFile(git.resolve("git.exe"))
            val bash = Files.createDirectories(root.resolve("Git/bin")).resolve("bash.exe")
            Files.createFile(bash)
            val env = mapOf("SystemRoot" to root.resolve("Windows").toString(), "Path" to "$system;$git")

            assertEquals(bash.toString(), captureShell(windows = true, environment = env))
            assertFailsWith<IllegalStateException> { captureShell(windows = true, environment = env + ("Path" to system.toString())) }
            assertEquals(system.resolve("bash.exe").toString(), captureShell(windows = true,
                environment = env + ("KOBLAS_BASH" to system.resolve("bash.exe").toString())))
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    @Test
    fun `explicit bash settings must name an existing absolute executable`() {
        assertFailsWith<IllegalArgumentException> { captureShell(environment = mapOf("KOBLAS_BASH" to "bash")) }
    }
}
