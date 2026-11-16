package com.eignex.koblas.bench

import java.io.File
import java.nio.file.Files
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
        withTemporaryDirectory("koblas-windows capture") { root ->
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
            writeExecutableScript(runner, """
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
            val log = runProcess(arguments, mapOf(
                "BASH_ENV" to shellEnvironment.toString(),
                "FIXTURE_JAVA_ARCH" to javaArchitecture,
                "PROCESSOR_ARCHITECTURE" to "AMD64",
                "PROCESSOR_ARCHITEW6432" to "AMD64",
                "KOBLAS_CAPTURE_ROOT" to repository.toString(),
                "TMPDIR" to "C:/fixture$root",
                "KOBLAS_BENCH_JAVA" to "C:/fixture$runner",
                "KOBLAS_BENCH_CLASSPATH" to classpath,
            ), expectedExit)
            if (expectedExit != 0) {
                assertTrue("Windows Native captures require an x64 JVM" in log, log)
                return@withTemporaryDirectory
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
        }
    }

    @Test fun `failed prepared captures retain partial CSV and logs`() {
        withTemporaryDirectory("koblas-partial") { root ->
            val java = root.resolve("java")
            writeExecutableScript(java,
                "for arg in \"\$@\"; do case \"\$arg\" in --output=*) " +
                    "printf 'case,ns\\npartial,12\\n' > \"\${arg#--output=}\";; esac; done\n" +
                    "echo failure-log\nexit 7\n")
            val output = root.resolve("report")
            val script = File("capture-report.sh").absoluteFile
            runProcess(listOf(captureShell(), script.path, "--targets", "jvm-scalar", "--libraries", "openblas",
                "--prepared", "--retain-incomplete", "--smoke", "--output", output.toString()), mapOf(
                "KOBLAS_BENCH_JAVA" to java.toString(),
                "KOBLAS_BENCH_CLASSPATH" to "unused",
            ), expectedExit = 7)
            assertTrue(Files.readString(output.resolve("jvm-scalar.csv")).contains("partial,12"))
            assertEquals("status=incomplete", Files.readAllLines(output.resolve("metadata.txt")).first())
            assertTrue(Files.readString(output.resolve("jvm-scalar.log")).contains("failure-log"))
        }
    }

    @Test fun `prepared rounds preserve target ordering and JVM settings`() {
        withTemporaryDirectory("koblas-prepared-rounds") { root ->
            val repository = root.resolve("repository")
            val bench = Files.createDirectories(repository.resolve("koblas-bench"))
            listOf("cases.txt", "select-cases.awk").forEach { name ->
                Files.copy(File(name).toPath(), bench.resolve(name))
            }
            val java = root.resolve("java")
            writeExecutableScript(java, """
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
            """)
            val output = root.resolve("report")
            runProcess(listOf(captureShell(), File("capture-report.sh").absolutePath,
                "--targets", "jvm-simd,jvm-scalar,jvm-default-no-simd,jvm-auto-openblas-no-simd",
                "--libraries", "openblas", "--prepared",
                "--retain-incomplete", "--smoke", "--rounds", "2", "--jvm-gc", "serial",
                "--jvm-heap", "256m", "--output", output.toString()), mapOf(
                "KOBLAS_CAPTURE_ROOT" to repository.toString(),
                "KOBLAS_BENCH_JAVA" to java.toString(),
                "KOBLAS_BENCH_CLASSPATH" to "fixture-classpath",
            ))
            val metadata = Files.readString(output.resolve("metadata.txt"))
            assertTrue(metadata.startsWith("status=complete\n"), metadata)
            assertTrue(metadata.contains(
                "round_1_targets=jvm-simd,jvm-scalar,jvm-default-no-simd,jvm-auto-openblas-no-simd"), metadata)
            assertTrue(metadata.contains(
                "round_2_targets=jvm-scalar,jvm-default-no-simd,jvm-auto-openblas-no-simd,jvm-simd"), metadata)
            for (round in 1..2) {
                for (target in listOf("jvm-scalar", "jvm-simd", "jvm-default-no-simd", "jvm-auto-openblas-no-simd")) {
                    assertTrue(Files.readString(output.resolve("pass-$round/metadata.txt")).contains(
                        "profile_cpu_model=runtime fixture"))
                    assertTrue(Files.readString(output.resolve("pass-$round/metadata.txt")).contains(
                        "profile_cpu_class=Avx2"))
                    assertTrue(Files.readString(output.resolve("pass-$round/metadata.txt")).contains(
                        "profile_sve_vector_bits=*"))
                    assertTrue(Files.readString(output.resolve("pass-$round/metadata.txt")).contains(
                        "profile_runtime=jvm\nprofile_os=Linux\nprofile_arch=X86_64"))
                    val arguments = Files.readAllLines(output.resolve("pass-$round/$target.args"))
                    assertTrue("-XX:+UseSerialGC" in arguments)
                    assertTrue("-Xms256m" in arguments && "-Xmx256m" in arguments)
                    assertTrue("fixture-classpath" in arguments)
                    assertEquals(target == "jvm-simd", "--add-modules=jdk.incubator.vector" in arguments)
                    assertTrue(Files.readString(output.resolve("pass-$round/$target.csv")).contains("fixture,12"))
                }
            }
        }
    }
}
