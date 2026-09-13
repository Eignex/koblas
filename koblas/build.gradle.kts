import org.jetbrains.kotlin.gradle.tasks.CInteropProcess
import org.jetbrains.kotlin.konan.target.PlatformManager
import org.jetbrains.kotlin.konan.target.KonanTarget
import org.gradle.api.provider.ValueSource
import org.gradle.api.provider.ValueSourceParameters
import org.gradle.api.tasks.JavaExec
import org.gradle.jvm.toolchain.JavaLanguageVersion
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
    id("com.eignex.kmp") version "1.3.3"
    kotlin("plugin.serialization") version "2.4.10"
}

// Apple toolchain discovery invokes xcrun; ValueSource makes its result a configuration-cache input.
abstract class NativeKernelToolchain : ValueSource<List<String>, NativeKernelToolchain.Parameters> {
    interface Parameters : ValueSourceParameters {
        val nativeHome: Property<String>
        val dataDirectory: Property<String>
        val targetName: Property<String>
    }

    override fun obtain(): List<String> {
        val manager = PlatformManager(parameters.nativeHome.get(), konanDataDir = parameters.dataDirectory.orNull?.takeIf(String::isNotEmpty))
        val clang = manager.platform(KonanTarget.predefinedTargets.getValue(parameters.targetName.get())).clang
        return listOf(clang.clangC().first(), clang.llvmAr().first()) + clang.clangArgs
    }
}

abstract class PrepareNativeKernelToolchain : DefaultTask() {
    @get:Input abstract val nativeHome: Property<String>
    @get:Input @get:Optional abstract val dataDirectory: Property<String>
    @get:Input abstract val targetName: Property<String>

    @TaskAction
    fun prepare() {
        PlatformManager(nativeHome.get(), konanDataDir = dataDirectory.orNull?.takeIf(String::isNotEmpty))
            .loader(KonanTarget.predefinedTargets.getValue(targetName.get())).downloadDependencies()
    }
}

eignexPublish {
    description.set("Dense and sparse BLAS for Kotlin Multiplatform with built-in C and SIMD kernels.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
    val nativeHostIsMacos = System.getProperty("os.name").startsWith("Mac", ignoreCase = true)
    explicitApi()
    applyDefaultHierarchyTemplate()
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
        optIn.add("com.eignex.koblas.UnsafeKoblasApi")
    }
    // The JVM C kernels use java.lang.foreign, finalized in 22 and used here with
    // Linker.Option.critical. 25 is the current LTS-track release; this is the floor for JVM consumers.
    jvmToolchain(25)
    jvm {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_25)
        }
    }
    linuxX64(); linuxArm64()
    macosArm64()

    targets.withType<KotlinNativeTarget>().configureEach {
        if (name != "macosArm64" || nativeHostIsMacos) {
            val interop = compilations.getByName("main").cinterops.create("koblasKernels") {
                definitionFile.set(project.file("src/nativeInterop/cinterop/koblas_kernels.def"))
                includeDirs(project.file("src/nativeInterop/kernels"))
            }
            val interopTask = tasks.named<CInteropProcess>(interop.interopProcessingTaskName)
            val target = konanTarget
            val archiveDirectory = layout.buildDirectory.dir("kernels/$name")
            // KGP 2.4 exposes the selected distribution only through these accessors.
            @Suppress("DEPRECATION_ERROR")
            val nativeHome = interopTask.get().konanHome
            @Suppress("DEPRECATION_ERROR")
            val dataDirectory = interopTask.get().konanDataDir
            val toolchain = providers.of(NativeKernelToolchain::class) {
                parameters.nativeHome.set(nativeHome)
                parameters.dataDirectory.set(dataDirectory.map { it.orEmpty() })
                parameters.targetName.set(target.name)
            }
            val nativePlatform = when (name) {
                "linuxX64" -> "linux-x86_64"
                "linuxArm64" -> "linux-arm64"
                else -> "macosx-arm64"
            }
            val selectedNativeHome = nativeHome
            val selectedDataDirectory = dataDirectory
            val prepareToolchain = tasks.register<PrepareNativeKernelToolchain>(
                "prepare${name.replaceFirstChar(Char::uppercase)}KernelToolchain",
            ) {
                dependsOn(tasks.matching { it.name == "downloadKotlinNativeDistribution" })
                this.nativeHome.set(selectedNativeHome)
                this.dataDirectory.set(selectedDataDirectory.map { it.orEmpty() })
                targetName.set(target.name)
            }
            val archive = tasks.register<Exec>("build${name.replaceFirstChar(Char::uppercase)}Kernels") {
                dependsOn(prepareToolchain)
                inputs.file(toolchain.map { it[0] })
                inputs.file(toolchain.map { it[1] })
                inputs.files(fileTree("src/nativeInterop/kernels"), "../scripts/build-koblas-kernels.sh")
                inputs.file(nativeHome.map { "$it/konan/konan.properties" })
                inputs.property("target", target.name)
                inputs.property("compiler", toolchain.map { it[0] })
                inputs.property("flags", toolchain.map { it.drop(2) })
                inputs.property("nativeHome", nativeHome)
                outputs.dir(archiveDirectory)
                val script = rootProject.file("scripts/build-koblas-kernels.sh").absolutePath
                val destination = archiveDirectory.get().asFile.absolutePath
                doFirst {
                    val selected = toolchain.get()
                    commandLine(
                        listOf("bash", script, "--platform", nativePlatform, "--kind", "static",
                            "--output", destination, "--compiler", selected[0],
                            "--archiver", selected[1]) +
                            selected.drop(2).flatMap { listOf("--cflag", it) },
                    )
                }
            }
            interop.extraOpts("-libraryPath", archiveDirectory.get().asFile.absolutePath)
            interopTask.configure {
                dependsOn(archive)
                inputs.file(archiveDirectory.map { it.file("libkoblas_kernels.a") })
            }
        }
    }

    sourceSets {
        commonMain { kotlin.srcDir("src/tuning/kotlin") }
        val cMain = create("cMain") {
            dependsOn(commonMain.get())
        }
        val crossScalarMain = create("crossScalarMain") {
            dependsOn(commonMain.get())
        }
        linuxMain.get().dependsOn(cMain)
        macosMain.get().dependsOn(if (nativeHostIsMacos) cMain else crossScalarMain)

        commonMain.dependencies {
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
        }
        commonTest.dependencies {
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
        }
    }
}

val jvmKernelsPlatform = providers.gradleProperty("koblas.kernels.platform").orElse(
    providers.systemProperty("os.name").zip(providers.systemProperty("os.arch")) { osName, architecture ->
        when {
            osName.startsWith("Linux", ignoreCase = true) && architecture in setOf("amd64", "x86_64") ->
                "linux-x86_64"
            osName.startsWith("Linux", ignoreCase = true) && architecture in setOf("aarch64", "arm64") ->
                "linux-arm64"
            osName.startsWith("Mac", ignoreCase = true) && architecture in setOf("aarch64", "arm64") ->
                "macosx-arm64"
            else -> "unsupported"
        }
    },
)
val jvmKernelsResources = layout.buildDirectory.dir("kernels/resources")
val prebuiltJvmKernels = providers.gradleProperty("koblas.kernels.prebuilt").map(String::toBoolean).orElse(false)
val buildJvmKernels = tasks.register<Exec>("buildJvmKernels") {
    inputs.files(fileTree("src/nativeInterop/kernels"), "../scripts/build-koblas-kernels.sh")
    inputs.property("platform", jvmKernelsPlatform)
    val compiler = providers.environmentVariable("CC").orElse("cc")
    inputs.property("CC", compiler)
    inputs.property("compilerIdentity", providers.exec { commandLine(compiler.get(), "--version") }.standardOutput.asText)
    inputs.property("compilerTarget", providers.exec { commandLine(compiler.get(), "-dumpmachine") }.standardOutput.asText)
    outputs.dir(jvmKernelsResources)
    outputs.dir(layout.buildDirectory.dir("kernels/resources.build/${jvmKernelsPlatform.get()}"))
    commandLine(
        "bash",
        rootProject.file("scripts/build-koblas-kernels.sh").absolutePath,
        "--platform",
        jvmKernelsPlatform.get(),
        "--output",
        jvmKernelsResources.get().asFile.absolutePath,
    )
    environment("CC", providers.environmentVariable("CC").orElse("cc").get())
    enabled = !prebuiltJvmKernels.get() && jvmKernelsPlatform.get() != "unsupported"
}
val checkNativeKernels = tasks.register<Exec>("checkNativeKernels") {
    group = "verification"
    description = "Checks native probe ABI, exact rejection and exported symbols on the host."
    dependsOn(buildJvmKernels)
    inputs.files(fileTree("src/nativeInterop/tests"), "../scripts/check-koblas-kernels.sh")
    inputs.dir(jvmKernelsResources)
    commandLine("bash", rootProject.file("scripts/check-koblas-kernels.sh").absolutePath,
        jvmKernelsResources.get().asFile.absolutePath, jvmKernelsPlatform.get())
    enabled = !prebuiltJvmKernels.get() && jvmKernelsPlatform.get() != "unsupported"
}
tasks.named("check") { dependsOn(checkNativeKernels) }
tasks.named<Test>("jvmTest") {
    dependsOn(checkNativeKernels)
    systemProperty("koblas.test.nativeFixtures",
        layout.buildDirectory.dir("kernels/resources.checks/${jvmKernelsPlatform.get()}/fixtures").get().asFile.absolutePath)
}
val verifyJvmKernelResources = tasks.register<Exec>("verifyJvmKernelResources") {
    val root = jvmKernelsResources.get().asFile.absolutePath
    val required = listOf(
        "com/eignex/koblas/internal/kernels/linux-x86_64/libkoblas_kernels.so",
        "com/eignex/koblas/internal/kernels/linux-arm64/libkoblas_kernels.so",
        "com/eignex/koblas/internal/kernels/macosx-arm64/libkoblas_kernels.dylib",
    )
    inputs.files(required.map { "$root/$it" })
    commandLine(
        "bash",
        "-c",
        "set -euo pipefail; root=\"\$1\"; shift; for resource in \"\$@\"; do " +
            "test -f \"\$root/\$resource\" || { echo \"missing prebuilt JVM kernel resource: \$resource\" >&2; exit 1; }; done",
        "verify-jvm-kernels",
        root,
        *required.toTypedArray(),
    )
    enabled = prebuiltJvmKernels.get()
}
kotlin.sourceSets.named("jvmMain") { resources.srcDir(jvmKernelsResources) }
tasks.named("jvmProcessResources") { dependsOn(buildJvmKernels, verifyJvmKernelResources) }

// Dokka site is the canonical user documentation. Module-level and per-package prose live in
// adjacent .md files referenced here.
dokka {
    moduleName.set("koblas")
    dokkaSourceSets.configureEach {
        sourceLink {
            localDirectory.set(projectDir.resolve("src"))
            val sub = projectDir.relativeTo(rootDir).invariantSeparatorsPath
            val prefix = if (sub.isEmpty()) "src" else "$sub/src"
            remoteUrl("https://github.com/Eignex/${rootProject.name}/blob/main/$prefix")
            remoteLineSuffix.set("#L")
        }
    }
    dokkaSourceSets.named("commonMain") {
        includes.from(
            "module.md",
            "src/commonMain/kotlin/com/eignex/koblas/package.md",
            "src/commonMain/kotlin/com/eignex/koblas/dense/package.md",
            "src/commonMain/kotlin/com/eignex/koblas/sparse/package.md",
        )
    }
}

// JVM SIMD primitives in Primitives.kt use the incubator Vector API. Make the module visible to the
// Kotlin compiler and at test runtime; downstream JVM consumers need the same flag.
tasks.withType<KotlinJvmCompile>().configureEach {
    compilerOptions.freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector")
}
// FFM downcalls are restricted methods: a warning on 25, an error later. The bundled JVM kernels use them.
tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (project.findProperty("koblas.noSimd") != "true") {
        jvmArgs("--add-modules=jdk.incubator.vector")
    }
}

// Vector API carrier objects are scalar-replaced by HotSpot only after C2 compilation. Kover instruments test
// classes before their JVM starts, which prevents that replacement and turns an otherwise allocation-free sparse
// kernel into a coverage artifact. Run this check in its own, uninstrumented JVM with an explicit SIMD engine.
val jvmTestCompilation = (kotlin.targets.getByName("jvm") as KotlinJvmTarget).compilations.getByName("test")
val allocationCheckJavaLauncher = javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) }
val simdSparseAllocationCheck = tasks.register<JavaExec>("simdSparseAllocationCheck") {
    group = "verification"
    description = "Checks allocation-free JVM SIMD indexed sparse dot, gather, and norm kernels outside Kover."
    dependsOn("jvmTestClasses")
    classpath(jvmTestCompilation.output.allOutputs, configurations.getByName("jvmTestRuntimeClasspath"))
    mainClass.set("com.eignex.koblas.sparse.SimdSparseAllocationCheck")
    javaLauncher.set(allocationCheckJavaLauncher)
    jvmArgs(
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-XX:-TieredCompilation",
        "-XX:CompileThreshold=1000",
    )
}
tasks.named("check") { dependsOn(simdSparseAllocationCheck) }

val nativeRuntimeCheck = tasks.register<JavaExec>("nativeRuntimeCheck") {
    group = "verification"
    description = "Checks warmed native allocation and concurrent exact calls outside coverage instrumentation."
    dependsOn("jvmTestClasses", buildJvmKernels)
    classpath(jvmTestCompilation.output.allOutputs, configurations.getByName("jvmTestRuntimeClasspath"))
    mainClass.set("com.eignex.koblas.internal.kernels.JvmNativeRuntimeCheck")
    javaLauncher.set(allocationCheckJavaLauncher)
    jvmArgs("--enable-native-access=ALL-UNNAMED", "-XX:-TieredCompilation", "-XX:CompileThreshold=1000")
}
tasks.named("check") { dependsOn(nativeRuntimeCheck) }

// Coverage instrumentation can retain Vector API carriers. Measure the mixed policy path after C2 warmup
// in a separate JVM, including when CI invokes jvmTest directly instead of the aggregate check task.
val denseDispatchRuntimeCheck = tasks.register<JavaExec>("denseDispatchRuntimeCheck") {
    group = "verification"
    description = "Checks warmed dense selection and mixed runtime/native allocation outside coverage instrumentation."
    dependsOn("jvmTestClasses", buildJvmKernels)
    classpath(jvmTestCompilation.output.allOutputs, configurations.getByName("jvmTestRuntimeClasspath"))
    mainClass.set("com.eignex.koblas.dense.DenseDispatchRuntimeCheck")
    javaLauncher.set(allocationCheckJavaLauncher)
    jvmArgs("--enable-native-access=ALL-UNNAMED", "-XX:-TieredCompilation", "-XX:CompileThreshold=1000", "-Xbatch")
    if (project.findProperty("koblas.noSimd") != "true") {
        jvmArgs("--add-modules=jdk.incubator.vector")
    }
}
tasks.named("jvmTest") { dependsOn(denseDispatchRuntimeCheck) }


// Kotlin emits a `$DefaultImpls` holder for every interface with a body, and a bridge for every method with
// a default argument. Neither is reachable from Kotlin call sites, so both count as permanently uncovered and
// make the report read as though tested code were not: `Blas.syr` shows its real body at 98% next to a bridge
// at 0%. Dropping the holders leaves the report describing code a test can actually reach.
kover {
    reports {
        filters {
            excludes {
                classes($$"*$DefaultImpls")
            }
        }
    }
}

// A stable module name, so a modular consumer sees a named module rather than one named after the jar
// file. That is what lets native access be granted per module instead of blanket ALL-UNNAMED.
tasks.named<Jar>("jvmJar") {
    manifest {
        attributes("Automatic-Module-Name" to "com.eignex.koblas")
    }
}
