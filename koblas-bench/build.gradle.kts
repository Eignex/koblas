import org.gradle.api.tasks.JavaExec
import org.gradle.jvm.toolchain.JavaLanguageVersion
import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget

plugins { kotlin("multiplatform") version "2.4.10" }

repositories { mavenCentral() }

kotlin {
    applyDefaultHierarchyTemplate()
    jvmToolchain(25)
    compilerOptions { freeCompilerArgs.add("-Xexpect-actual-classes") }
    jvm()
    linuxX64 { binaries.executable { entryPoint = "com.eignex.koblas.bench.main"; baseName = "koblas-bench" } }
    macosArm64 { binaries.executable { entryPoint = "com.eignex.koblas.bench.main"; baseName = "koblas-bench" } }
    sourceSets {
        commonMain.dependencies { implementation(project(":koblas")) }
        commonTest.dependencies { implementation(kotlin("test")) }
        jvmMain.dependencies { implementation("org.openjdk.jmh:jmh-core:1.37") }
    }
}

dependencies {
    add("jvmMainAnnotationProcessor", "org.openjdk.jmh:jmh-generator-annprocess:1.37")
}

private val sourceCommit = providers.exec {
    commandLine("git", "rev-parse", "HEAD")
    workingDir(rootProject.projectDir)
}.standardOutput.asText.map { it.trim() }
private val sourceDirty = providers.exec {
    commandLine("git", "status", "--porcelain", "--untracked-files=normal")
    workingDir(rootProject.projectDir)
}.standardOutput.asText.map { if (it.isBlank()) "false" else "true" }

private fun benchmarkArguments(mode: String, jmh: Boolean): List<String> = listOf(
    "--mode=$mode",
    "--operation=${providers.gradleProperty("bench.operation").orElse("all").get()}",
    "--cases=${providers.gradleProperty("bench.cases").orElse("koblas-bench/cases.txt").get()}",
    "--output=${providers.gradleProperty("bench.output").orElse("koblas-bench/build/benchmarks/$mode.csv").get()}",
    "--warmups=${providers.gradleProperty("bench.warmups").orElse("3").get()}",
    "--samples=${providers.gradleProperty("bench.samples").orElse("5").get()}",
    "--target-ms=${providers.gradleProperty("bench.targetMs").orElse("1000").get()}",
    "--pass=${providers.gradleProperty("bench.pass").orElse("1").get()}",
    "--source-commit=${sourceCommit.get()}",
    "--dirty=${sourceDirty.get()}",
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

registerJvmBenchmark("jvmCBenchmark", "jvm-c", vectorModule = false)
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
    args(benchmarkArguments("native", jmh = false))
}

fun registerOpenBlasCompatibilityCheck(name: String, resolution: String) = tasks.register<Exec>(name) {
    group = "verification"
    description = "Runs the standalone OpenBLAS verifier for the existing CI entry point."
    commandLine("bash", rootProject.file("koblas-bench/reference/test.sh").absolutePath)
    workingDir(rootProject.projectDir)
    doLast { logger.lifecycle(resolution) }
}

registerOpenBlasCompatibilityCheck(
    "jvmSelectedBenchmark",
    "resolved: arm=openblas dense=openblas/cblas threading=1 thread",
)
registerOpenBlasCompatibilityCheck(
    "linuxX64SelectedBenchmark",
    "resolved: arm=openblas dense=openblas/cblas-native threading=1 thread",
)

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}
