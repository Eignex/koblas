package com.eignex.koblas.bench

import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CaptureReportTest {
    @Test
    fun `windows captures convert file paths and preserve the native classpath`() {
        windowsCapture("amd64", native = true)
    }

    @Test
    fun `windows arm java overrides an emulated x64 shell`() {
        windowsCapture("aarch64", native = false)
    }

    @Test
    fun `windows arm java rejects a supplied x64 native executable`() {
        windowsCapture("aarch64", native = true, expectedExit = 2)
    }

    private fun windowsCapture(javaArchitecture: String, native: Boolean, expectedExit: Int = 0) {
        // These fixtures emulate MSYS paths on Unix; real Windows captures use the installed cygpath.
        if (System.getProperty("os.name").startsWith("Windows")) return
        val root = Files.createTempDirectory("koblas-windows capture")
        try {
            val bin = Files.createDirectory(root.resolve("bin"))
            val repository = root.resolve("repository")
            val bench = Files.createDirectories(repository.resolve("koblas-bench"))
            for (name in listOf("cases.txt", "select-cases.awk")) {
                Files.copy(File(name).toPath(), bench.resolve(name))
            }
            val shellEnvironment = bin.resolve("environment")
            Files.writeString(shellEnvironment, """
                uname() {
                  case "${'$'}1" in
                    -s) echo MINGW64_NT-10.0 ;;
                    -m) echo x86_64 ;;
                    *) echo 'MINGW64_NT-10.0 fixture x86_64' ;;
                  esac
                }
                powershell.exe() { echo cpu_model=fixture; }
                cygpath() {
                  case "${'$'}1" in
                    -u) printf '%s\n' "${'$'}{2#C:/fixture}" ;;
                    -m) printf 'C:/fixture%s\n' "${'$'}2" ;;
                    *) return 2 ;;
                  esac
                }
                git() {
                  case "${'$'}{!#}" in
                    --show-toplevel) echo "C:/fixture${'$'}KOBLAS_CAPTURE_ROOT" ;;
                    HEAD) echo fixture ;;
                    --porcelain) ;;
                    -z) printf 'koblas-bench/cases.txt\0koblas-bench/select-cases.awk\0' ;;
                    *) return 2 ;;
                  esac
                }
            """.trimIndent())
            val runner = bin.resolve("runner")
            executable(runner, """
                if [[ ${'$'}1 == -XshowSettings:properties ]]; then
                  printf '    os.arch = %s\r\n' "${'$'}{FIXTURE_JAVA_ARCH:-amd64}" >&2
                  exit 0
                fi
                for argument in "${'$'}@"; do
                  case "${'$'}argument" in --output=*) output=${'$'}{argument#--output=C:/fixture} ;; esac
                done
                printf 'case,ns\nfixture,12\n' > "${'$'}output"
                printf '%s\n' "${'$'}@" > "${'$'}{output%.csv}.args"
                echo 'resolved implementation=fixture runtime=fixture harness=fixture'
            """)
            val output = root.resolve("report")
            val classpath = "C:\\first path;D:\\second path"
            val targets = if (native) "jvm-scalar,native" else "jvm-scalar"
            val arguments = mutableListOf(captureShell(), File("capture-report.sh").absolutePath,
                "--targets", targets, "--libraries", "all", "--prepared", "--smoke", "--output", "C:/fixture$output")
            if (native) arguments += listOf("--native-executable", "C:/fixture$runner")
            val process = ProcessBuilder(arguments).redirectErrorStream(true).apply {
                environment()["BASH_ENV"] = shellEnvironment.toString()
                environment()["FIXTURE_JAVA_ARCH"] = javaArchitecture
                environment()["PROCESSOR_ARCHITECTURE"] = "AMD64"
                environment()["PROCESSOR_ARCHITEW6432"] = "AMD64"
                environment()["KOBLAS_CAPTURE_ROOT"] = repository.toString()
                environment()["TMPDIR"] = "C:/fixture$root"
                environment()["KOBLAS_BENCH_JAVA"] = "C:/fixture$runner"
                environment()["KOBLAS_BENCH_CLASSPATH"] = classpath
            }.start()
            val log = process.inputStream.bufferedReader().readText()

            assertEquals(expectedExit, process.waitFor(), log)
            if (expectedExit != 0) {
                assertTrue("Windows Native captures require an x64 JVM" in log, log)
                return
            }
            val metadata = Files.readString(output.resolve("metadata.txt"))
            assertTrue(metadata.startsWith("status=complete\n"), metadata)
            assertTrue("source_kind=git" in metadata, metadata)
            assertTrue("native_capable=$native" in metadata, metadata)
            assertTrue("libraries=openblas,${if (native) "onemkl" else "armpl"}" in metadata, metadata)
            assertTrue("architecture=${if (native) "x86_64" else "arm64"}" in metadata, metadata)
            for (target in targets.split(',')) {
                val arguments = Files.readAllLines(output.resolve("$target.args"))
                assertTrue(arguments.any { it.startsWith("--cases=C:/fixture") }, arguments.toString())
                assertTrue(arguments.any { it.startsWith("--output=C:/fixture") }, arguments.toString())
                assertEquals(target == "jvm-scalar", classpath in arguments)
            }
        } finally {
            root.toFile().deleteRecursively()
        }
    }

    private fun executable(path: Path, body: String) {
        Files.writeString(path, "#!/usr/bin/env bash\nset -eu\n${body.trimIndent()}\n")
        check(path.toFile().setExecutable(true))
    }

    @Test fun `failed prepared captures retain partial CSV and logs`() {
        val root = Files.createTempDirectory("koblas-partial")
        try {
            val java = root.resolve("java")
            Files.writeString(java, "#!/usr/bin/env bash\nfor arg in \"\$@\"; do case \"\$arg\" in --output=*) printf 'case,ns\\npartial,12\\n' > \"\${arg#--output=}\";; esac; done\necho failure-log\nexit 7\n")
            check(java.toFile().setExecutable(true))
            val output = root.resolve("report")
            val script = File("capture-report.sh").absoluteFile
            val process = ProcessBuilder(captureShell(), script.path, "--targets", "jvm-scalar", "--libraries", "openblas", "--prepared", "--retain-incomplete", "--smoke", "--output", output.toString())
                .redirectErrorStream(true).apply { environment()["KOBLAS_BENCH_JAVA"] = java.toString(); environment()["KOBLAS_BENCH_CLASSPATH"] = "unused" }.start()
            val log = process.inputStream.bufferedReader().readText()
            assertEquals(7, process.waitFor(), log)
            assertTrue(Files.readString(output.resolve("jvm-scalar.csv")).contains("partial,12"))
            assertEquals("status=incomplete", Files.readAllLines(output.resolve("metadata.txt")).first())
            assertTrue(Files.readString(output.resolve("jvm-scalar.log")).contains("failure-log"))
        } finally { root.toFile().deleteRecursively() }
    }

    @Test fun `prepared rounds preserve target ordering and JVM settings`() {
        val root = Files.createTempDirectory("koblas-prepared-rounds")
        try {
            val repository = root.resolve("repository")
            val bench = Files.createDirectories(repository.resolve("koblas-bench"))
            listOf("cases.txt", "select-cases.awk").forEach { name ->
                Files.copy(File(name).toPath(), bench.resolve(name))
            }
            val java = root.resolve("java")
            Files.writeString(java, """
                #!/usr/bin/env bash
                set -eu
                for arg in "${'$'}@"; do
                  case "${'$'}arg" in --output=*) output=${'$'}{arg#--output=};; esac
                done
                printf 'case,ns\nfixture,12\n' > "${'$'}output"
                printf '%s\n' "${'$'}@" > "${'$'}{output%.csv}.args"
                echo 'resolved implementation=fixture runtime=fixture harness=fixture'
                echo 'profile_cpu_model=runtime fixture'
                echo 'profile_cpu_class=Avx2'
                echo 'profile_sve_vector_bits=*'
                echo 'profile_runtime=jvm'
                echo 'profile_os=Linux'
                echo 'profile_arch=X86_64'
            """.trimIndent() + "\n")
            check(java.toFile().setExecutable(true))
            val output = root.resolve("report")
            val process = ProcessBuilder(captureShell(), File("capture-report.sh").absolutePath,
                "--targets", "jvm-simd,jvm-scalar", "--libraries", "openblas", "--prepared",
                "--retain-incomplete", "--smoke", "--rounds", "2", "--jvm-gc", "serial",
                "--jvm-heap", "256m", "--output", output.toString()).redirectErrorStream(true).apply {
                environment()["KOBLAS_CAPTURE_ROOT"] = repository.toString()
                environment()["KOBLAS_BENCH_JAVA"] = java.toString()
                environment()["KOBLAS_BENCH_CLASSPATH"] = "fixture-classpath"
            }.start()
            val log = process.inputStream.bufferedReader().readText()
            assertEquals(0, process.waitFor(), log)
            val metadata = Files.readString(output.resolve("metadata.txt"))
            assertTrue(metadata.startsWith("status=complete\n"), metadata)
            assertTrue(metadata.contains("round_1_targets=jvm-simd,jvm-scalar"), metadata)
            assertTrue(metadata.contains("round_2_targets=jvm-scalar,jvm-simd"), metadata)
            for (round in 1..2) for (target in listOf("jvm-scalar", "jvm-simd")) {
                assertTrue(Files.readString(output.resolve("pass-$round/metadata.txt")).contains("profile_cpu_model=runtime fixture"))
                assertTrue(Files.readString(output.resolve("pass-$round/metadata.txt")).contains("profile_cpu_class=Avx2"))
                assertTrue(Files.readString(output.resolve("pass-$round/metadata.txt")).contains("profile_sve_vector_bits=*"))
                assertTrue(Files.readString(output.resolve("pass-$round/metadata.txt")).contains("profile_runtime=jvm\nprofile_os=Linux\nprofile_arch=X86_64"))
                val arguments = Files.readAllLines(output.resolve("pass-$round/$target.args"))
                assertTrue("-XX:+UseSerialGC" in arguments)
                assertTrue("-Xms256m" in arguments && "-Xmx256m" in arguments)
                assertTrue("fixture-classpath" in arguments)
                assertEquals(target == "jvm-simd", "--add-modules=jdk.incubator.vector" in arguments)
                assertTrue(Files.readString(output.resolve("pass-$round/$target.csv")).contains("fixture,12"))
            }
        } finally { root.toFile().deleteRecursively() }
    }
}
