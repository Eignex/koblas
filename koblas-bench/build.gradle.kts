import org.gradle.api.tasks.JavaExec
import org.gradle.jvm.toolchain.JavaLanguageVersion
import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget

plugins { id("com.eignex.kmp") version "1.3.3" }

eignexPublish { publish.set(false) }
eignexBuild {
    abiValidationEnabled.set(false)
    // The benchmark harness does not yet follow the library's lint conventions.
    lintEnabled.set(false)
}

kotlin {
    applyDefaultHierarchyTemplate()
    compilerOptions { freeCompilerArgs.add("-Xexpect-actual-classes") }
    jvm()
    linuxX64 { binaries.executable { entryPoint = "com.eignex.koblas.bench.main"; baseName = "koblas-bench" } }
    macosArm64 { binaries.executable { entryPoint = "com.eignex.koblas.bench.main"; baseName = "koblas-bench" } }
    sourceSets {
        commonMain.dependencies { implementation(project(":koblas")) }
        jvmMain.dependencies { implementation("org.openjdk.jmh:jmh-core:1.37") }
    }
}

dependencies {
    add("jvmMainAnnotationProcessor", "org.openjdk.jmh:jmh-generator-annprocess:1.37")
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
val benchmarkJavaLauncher = javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) }

fun registerJvmBenchmark(name: String, mode: String, vectorModule: Boolean) = tasks.register<JavaExec>(name) {
    group = "benchmark"
    description = "Runs the shared cases through exact $mode koblas kernels."
    dependsOn(jvmCompilation.compileTaskProvider)
    classpath(jvmCompilation.output.allOutputs, configurations.getByName("jvmRuntimeClasspath"))
    mainClass.set("com.eignex.koblas.bench.JvmRunnerKt")
    javaLauncher.set(benchmarkJavaLauncher)
    workingDir(rootProject.projectDir)
    args(benchmarkArguments(mode, jmh = true))
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (vectorModule) jvmArgs("--add-modules=jdk.incubator.vector")
}

// The vendor arms run the same cases through the production bindings rather than a separate C wrapper, so
// what a row reports is the route of the call that was timed. Pick the library with -Pbench.vendor=NAME.
val benchVendor = providers.gradleProperty("bench.vendor").orElse("onemkl")

registerJvmBenchmark("jvmVendorBenchmark", "jvm-vendor-${benchVendor.get()}", vectorModule = false)
registerJvmBenchmark("jvmCBenchmark", "jvm-c", vectorModule = false)
registerJvmBenchmark("jvmCRawBenchmark", "jvm-c-raw-" + providers.gradleProperty("bench.variant").orElse("scalar").get(), vectorModule = false)
registerJvmBenchmark("jvmSimdBenchmark", "jvm-simd", vectorModule = true)
registerJvmBenchmark("jvmScalarBenchmark", "jvm-scalar", vectorModule = false)

val hostTarget = when {
    System.getProperty("os.name").startsWith("Linux") && System.getProperty("os.arch") == "amd64" -> "LinuxX64"
    System.getProperty("os.name").startsWith("Mac") && System.getProperty("os.arch") == "aarch64" -> "MacosArm64"
    else -> null
}

tasks.register<Exec>("nativeBenchmark") {
    group = "benchmark"
    description = "Runs the shared cases through the exact native koblas C engine."
    require(hostTarget != null) { "native benchmarks are supported on Linux x86-64 and macOS arm64" }
    dependsOn("linkReleaseExecutable$hostTarget")
    val targetDir = hostTarget!!.replaceFirstChar(Char::lowercase)
    commandLine(layout.buildDirectory.file("bin/$targetDir/releaseExecutable/koblas-bench.kexe").get().asFile.absolutePath)
    workingDir(rootProject.projectDir)
    args(benchmarkArguments(providers.gradleProperty("bench.variant").map { "native-raw-$it" }.orElse("native").get(), jmh = false))
}

tasks.register<Exec>("nativeVendorBenchmark") {
    group = "benchmark"
    description = "Runs the shared cases through a vendor BLAS bound by the production Native binding."
    require(hostTarget != null) { "native benchmarks are supported on Linux x86-64 and macOS arm64" }
    dependsOn("linkReleaseExecutable$hostTarget")
    val targetDir = hostTarget!!.replaceFirstChar(Char::lowercase)
    commandLine(layout.buildDirectory.file("bin/$targetDir/releaseExecutable/koblas-bench.kexe").get().asFile.absolutePath)
    workingDir(rootProject.projectDir)
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
    javaLauncher.set(benchmarkJavaLauncher)
    workingDir(rootProject.projectDir)
    args(
        smokeArguments + listOf(
            "--mode=jvm-vendor-openblas",
            "--forks=1",
            "--output=koblas-bench/build/benchmarks/openblas-smoke.csv",
        ),
    )
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    doLast { logger.lifecycle("resolved: arm=openblas dense=openblas/cblas threading=1 thread") }
}

tasks.register<Exec>("linuxX64SelectedBenchmark") {
    group = "verification"
    description = "Resolves OpenBLAS through the Native vendor binding and runs one smoke case."
    require(hostTarget != null) { "native benchmarks are supported on Linux x86-64 and macOS arm64" }
    dependsOn("linkReleaseExecutable$hostTarget")
    val targetDir = hostTarget!!.replaceFirstChar(Char::lowercase)
    commandLine(layout.buildDirectory.file("bin/$targetDir/releaseExecutable/koblas-bench.kexe").get().asFile.absolutePath)
    workingDir(rootProject.projectDir)
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

