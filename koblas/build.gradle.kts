import org.jetbrains.kotlin.konan.target.PlatformManager
import org.jetbrains.kotlin.konan.target.KonanTarget
import org.gradle.api.provider.ValueSource
import org.gradle.api.provider.ValueSourceParameters
import org.gradle.api.tasks.JavaExec
import org.gradle.jvm.toolchain.JavaLanguageVersion
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget

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
    description.set("Dense and sparse BLAS for Kotlin Multiplatform over vendor BLAS with Kotlin Level 1 kernels.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
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

    sourceSets {
        commonMain.dependencies {
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
        }
        commonTest.dependencies {
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
        }
    }
}

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
