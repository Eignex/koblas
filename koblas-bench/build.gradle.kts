import java.nio.file.Path as JdkPath
import org.gradle.api.tasks.JavaExec

import org.gradle.jvm.toolchain.JavaLanguageVersion
import org.gradle.jvm.toolchain.JvmVendorSpec
import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget

plugins { id("com.eignex.kmp") }

eignexPublish { publish.set(false) }
eignexBuild {
    jvmToolchain.set(25)
    abiValidationEnabled.set(false)
    // The benchmark harness does not yet follow the library's lint conventions.
    lintEnabled.set(false)
}

repositories {
    exclusiveContent {
        forRepository {
            ivy {
                name = "labBenchReleases"
                url = uri(providers.gradleProperty("labBenchRepository")
                    .getOrElse("https://github.com/Eignex/eignex-lab/releases/download"))
                patternLayout { artifact("lab-bench-v[revision]/[artifact]-[revision](-[classifier]).[ext]") }
                metadataSources { gradleMetadata() }
            }
        }
        filter { includeModule("com.eignex", "lab-bench") }
    }
}
val labBenchVersion = providers.gradleProperty("labBenchVersion").getOrElse("0.1.0")

kotlin {
    compilerOptions {
        allWarningsAsErrors.set(true)
        freeCompilerArgs.add("-Xexpect-actual-classes")
        // Timing one Level 1 implementation against another is what this module is for, so naming an engine
        // is its ordinary business rather than something to opt into per file.
        optIn.add("com.eignex.koblas.KoblasEngineApi")
    }
    jvm()
    linuxX64 { binaries.executable { entryPoint = "com.eignex.koblas.bench.main"; baseName = "koblas-bench" } }
    linuxArm64 { binaries.executable { entryPoint = "com.eignex.koblas.bench.main"; baseName = "koblas-bench" } }
    macosArm64 { binaries.executable { entryPoint = "com.eignex.koblas.bench.main"; baseName = "koblas-bench" } }
    sourceSets {
        commonMain.dependencies { implementation(project(":koblas")) }
        jvmMain.dependencies {
            implementation(project(":koblas-bench:jmh"))
            implementation("com.eignex:lab-bench:$labBenchVersion")
        }
    }
}

dependencies {
    add("jvmMainAnnotationProcessor", project(":koblas-bench:jmh"))
}

private fun benchmarkArguments(mode: String, jmh: Boolean): List<String> = listOf(
    "--mode=$mode",
    "--operation=${providers.gradleProperty("bench.operation").orElse("all").get()}",
    "--suite=${providers.gradleProperty("bench.suite").orElse("default").get()}",
    "--cases=${providers.gradleProperty("bench.cases").orElse("koblas-bench/cases.txt").get()}",
    "--output=${providers.gradleProperty("bench.output").orElse("koblas-bench/build/benchmarks/$mode.csv").get()}",
    "--warmups=${providers.gradleProperty("bench.warmups").orElse("3").get()}",
    "--samples=${providers.gradleProperty("bench.samples").orElse("5").get()}",
    "--target-ms=${providers.gradleProperty("bench.targetMs").orElse("1000").get()}",
) + if (jmh) listOf("--forks=${providers.gradleProperty("bench.forks").orElse("2").get()}") else emptyList()

val jvmCompilation = (kotlin.targets.getByName("jvm") as KotlinJvmTarget).compilations.getByName("main")
val benchmarkJavaVersion = providers.gradleProperty("bench.javaVersion").orElse("25")
val benchmarkJvmVendor = providers.gradleProperty("bench.javaVendor")
val expectedBenchmarkJavaHome = providers.gradleProperty("bench.expectedJavaHome")
val benchmarkJavaLauncher = javaToolchains.launcherFor {
    languageVersion.set(benchmarkJavaVersion.map { JavaLanguageVersion.of(it.toInt()) })
    if (benchmarkJvmVendor.isPresent) vendor.set(JvmVendorSpec.matching(benchmarkJvmVendor.get()))
}

fun JavaExec.useBenchmarkRuntime() {
    val explicit = expectedBenchmarkJavaHome.orNull
    if (explicit != null) setExecutable(JdkPath.of(explicit, "bin", "java").toString())
    else javaLauncher.set(benchmarkJavaLauncher)
}

tasks.register("prepareBenchmarkRuntime") {
    group = "benchmark"
    description = "Resolves the benchmark JVM before any timed capture begins."
    val launcher = benchmarkJavaLauncher
    val expected = expectedBenchmarkJavaHome
    val version = benchmarkJavaVersion
    doLast {
        if (expected.isPresent) {
            val executable = JdkPath.of(expected.get(), "bin", "java").toRealPath()
            val process = ProcessBuilder(executable.toString(), "-version").redirectErrorStream(true).start()
            val description = process.inputStream.bufferedReader().readText()
            check(process.waitFor() == 0 && description.contains("\"${version.get()}."))
            println("benchmark_runtime_executable=$executable")
            println("benchmark_runtime_version=${description.lineSequence().first()}")
        } else {
            val resolved = launcher.get()
            println("benchmark_runtime_executable=${resolved.executablePath}")
            println("benchmark_runtime_version=${resolved.metadata.javaRuntimeVersion}")
            println("benchmark_runtime_vendor=${resolved.metadata.vendor}")
        }
    }
}

fun registerJvmBenchmark(name: String, mode: String, vectorModule: Boolean) = tasks.register<JavaExec>(name) {
    group = "benchmark"
    description = "Runs the shared cases through exact $mode koblas kernels."
    dependsOn(jvmCompilation.compileTaskProvider)
    classpath(jvmCompilation.output.allOutputs, configurations.getByName("jvmRuntimeClasspath"))
    mainClass.set("com.eignex.koblas.bench.JvmRunnerKt")
    useBenchmarkRuntime()
    workingDir(rootProject.projectDir)
    args(benchmarkArguments(mode, jmh = true))
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (vectorModule) jvmArgs("--add-modules=jdk.incubator.vector")
}

// The vendor arms run the same cases through the production bindings rather than a separate C wrapper, so
// what a row reports is the route of the call that was timed. Pick the library with -Pbench.vendor=NAME.
val benchVendor = providers.gradleProperty("bench.vendor").orElse("onemkl")

registerJvmBenchmark("jvmVendorBenchmark", "jvm-vendor-${benchVendor.get()}", vectorModule = false)
registerJvmBenchmark("jvmSimdBenchmark", "jvm-simd", vectorModule = true)
registerJvmBenchmark("jvmDefaultBenchmark", "jvm-default", vectorModule = true)
registerJvmBenchmark("jvmScalarBenchmark", "jvm-scalar", vectorModule = false)

val hostTarget = when {
    System.getProperty("os.name").startsWith("Linux") && System.getProperty("os.arch") == "amd64" -> "LinuxX64"
    System.getProperty("os.name").startsWith("Mac") && System.getProperty("os.arch") == "aarch64" -> "MacosArm64"
    else -> null
}

private val NO_NATIVE_HOST = "native benchmarks are supported on Linux x86-64 and macOS arm64"

/**
 * Registers a native benchmark where one can be built, and a task that says why not where one cannot.
 *
 * The Kotlin/Native compiler ships no linux-aarch64 host, so an arm64 Linux machine can run the JVM arms
 * only. Refusing while the task is being configured took the whole project down with it, which left those
 * arms unreachable too; refusing when the task runs leaves the rest of the module usable.
 */
fun registerNativeBenchmark(name: String, configure: Exec.() -> Unit) {
    val target = hostTarget
    if (target == null) {
        tasks.register(name) {
            group = "benchmark"
            description = NO_NATIVE_HOST
            doFirst { error(NO_NATIVE_HOST) }
        }
        return
    }
    tasks.register<Exec>(name) {
        group = "benchmark"
        dependsOn("linkReleaseExecutable$target")
        val targetDir = target.replaceFirstChar(Char::lowercase)
        commandLine(
            layout.buildDirectory.file("bin/$targetDir/releaseExecutable/koblas-bench.kexe").get().asFile.absolutePath,
        )
        workingDir(rootProject.projectDir)
        configure()
    }
}

registerNativeBenchmark("nativeBenchmark") {
    description = "Runs the shared cases through the native koblas engine."
    args(benchmarkArguments("native", jmh = false))
}

registerNativeBenchmark("nativeDefaultBenchmark") {
    description = "Runs the shared cases through the engine this platform selects, which may compose a library."
    args(benchmarkArguments("native-default", jmh = false))
}

registerNativeBenchmark("nativeVendorBenchmark") {
    description = "Runs the shared cases through a vendor BLAS bound by the production Native binding."
    args(benchmarkArguments("native-vendor-${benchVendor.get()}", jmh = false))
}

// The CI entry points, which now exercise the production binding rather than a C program written against the
// same library. They resolve OpenBLAS through the ordinary vendor arm, so a failure here is a real failure to
// reach the library a report would name. They run the arm directly instead of going back through
// capture-report.sh, which is what lets that script's vendor loop use Gradle without a build inside a build.
// The task names and the resolution lines they print are fixed by the workflow that greps for them.
private val smokeArguments = listOf(
    "--operation=gemm", "--suite=default", "--cases=koblas-bench/cases.txt",
    "--warmups=0", "--samples=1", "--target-ms=1",
)

tasks.register<JavaExec>("jvmSelectedBenchmark") {
    group = "verification"
    description = "Resolves OpenBLAS through the JVM vendor binding and runs one smoke case."
    dependsOn(jvmCompilation.compileTaskProvider)
    classpath(jvmCompilation.output.allOutputs, configurations.getByName("jvmRuntimeClasspath"))
    mainClass.set("com.eignex.koblas.bench.JvmRunnerKt")
    useBenchmarkRuntime()
    workingDir(rootProject.projectDir)
    args(
        smokeArguments + listOf(
            "--mode=jvm-vendor-openblas",
            "--forks=1",
            "--output=koblas-bench/build/benchmarks/openblas-smoke.csv",
        ),
    )
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--sun-misc-unsafe-memory-access=deny")
    doLast { logger.lifecycle("resolved: arm=openblas dense=openblas/cblas threading=1 thread") }
}

registerNativeBenchmark("linuxX64SelectedBenchmark") {
    group = "verification"
    description = "Resolves OpenBLAS through the Native vendor binding and runs one smoke case."
    args(
        smokeArguments + listOf(
            "--mode=native-vendor-openblas",
            "--output=koblas-bench/build/benchmarks/openblas-smoke-native.csv",
        ),
    )
    doLast { logger.lifecycle("resolved: arm=openblas dense=openblas/cblas-native threading=1 thread") }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}

// Consumer-shaped setup-inclusive comparison, including thread allocation measurements.
tasks.register<JavaExec>("jvmCholeskyViewBenchmark") {
    group = "benchmark"
    description = "Compares copied and borrowed trailing updates in a finite SPD blocked Cholesky workload."
    dependsOn(jvmCompilation.compileTaskProvider)
    classpath(jvmCompilation.output.allOutputs, configurations.getByName("jvmRuntimeClasspath"))
    mainClass.set("com.eignex.koblas.bench.CholeskyViewBenchmark")
    useBenchmarkRuntime()
    jvmArgs("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "-Xms512m", "-Xmx512m")
}

tasks.register<Sync>("prepareLabBenchmark") {
    group = "benchmark"
    from(jvmCompilation.output.allOutputs) { into("classes") }
    from(configurations.named("jvmRuntimeClasspath")) { into("lib") }
    into(layout.buildDirectory.dir("lab"))
}
