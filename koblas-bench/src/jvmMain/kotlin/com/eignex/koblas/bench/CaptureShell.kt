package com.eignex.koblas.bench

import java.nio.file.Files
import java.nio.file.Path

/** Select an absolute Git Bash executable before Windows can substitute the WSL launcher. */
internal fun captureShell(
    windows: Boolean = System.getProperty("os.name").startsWith("Windows", ignoreCase = true),
    environment: Map<String, String> = System.getenv(),
): String {
    fun setting(name: String): String? = environment.entries.firstOrNull { it.key.equals(name, ignoreCase = windows) }?.value
    setting("KOBLAS_BASH")?.let { value ->
        val path = Path.of(value)
        require(path.isAbsolute && Files.isRegularFile(path)) { "KOBLAS_BASH must name an existing absolute Bash executable: $value" }
        return path.toString()
    }
    if (!windows) return "bash"
    val systemRoot = setting("SystemRoot")?.let { Path.of(it).toAbsolutePath().normalize() }
    val directories = setting("PATH").orEmpty().split(';').map { Path.of(it.trim().trim('"')) }
        .filter { it.isAbsolute }.map { it.normalize() }
        .filter { path -> systemRoot == null || !(path.toString().equals(systemRoot.toString(), ignoreCase = true) ||
            path.toString().lowercase().startsWith(systemRoot.toString().lowercase() + path.fileSystem.separator)) }
    for (directory in directories) {
        val bash = directory.resolve("bash.exe")
        if (Files.isRegularFile(bash)) return bash.toString()
        // Git's installer can add only Git/cmd to PATH instead of Git/bin.
        if (Files.isRegularFile(directory.resolve("git.exe"))) {
            for (relative in listOf("../bin/bash.exe", "../../bin/bash.exe")) {
                val installed = directory.resolve(relative).normalize()
                if (Files.isRegularFile(installed)) return installed.toString()
            }
        }
    }
    error("Git Bash is required on Windows; put its bin directory on PATH or set KOBLAS_BASH to its absolute bash.exe path")
}
