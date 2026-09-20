import org.gradle.api.tasks.JavaExec
import org.gradle.jvm.toolchain.JavaLanguageVersion
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget

plugins {
    id("com.eignex.kmp") version "1.3.3"
    kotlin("plugin.serialization") version "2.4.10"
}

eignexPublish {
    description.set("Portable dense and sparse BLAS for Kotlin Multiplatform with JVM SIMD and optional host bindings.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
    explicitApi()
    applyDefaultHierarchyTemplate()
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
        optIn.add("com.eignex.koblas.UnsafeKoblasApi")
    }
    // The vendor binding uses java.lang.foreign, finalized in 22. 25 is the current LTS-track release; this
    // is the floor for JVM consumers.
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
        // Conformance tests compare a vectorised kernel against the portable one that defines its semantics,
        // which means naming an engine. Opting in for the test source sets and not for main is what keeps
        // that a test affordance rather than something production code can reach for by accident.
        all {
            if (name.endsWith("Test")) languageSettings.optIn("com.eignex.koblas.KoblasEngineApi")
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

// The JVM SIMD kernels use the incubator Vector API. Make the module visible to the Kotlin compiler and at
// test runtime; downstream JVM consumers need the same flag.
tasks.withType<KotlinJvmCompile>().configureEach {
    compilerOptions.freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector")
}
// FFM downcalls are restricted methods: a warning on 25, an error later. The vendor binding uses them.
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
/**
 * The dense allocation check in one runtime configuration.
 *
 * A kernel that keeps its accumulators in registers on this machine need not on another, and the two things
 * that decide it are the width of the species and whether a multiply-add is one instruction. Both are fixed
 * when the virtual machine starts, so the only way to check the other configuration is another process. The
 * configurations below are registered rather than left to a reviewer to remember, because the first time a
 * tile spilled without a fused multiply-add it was found by a reviewer and not by this build.
 */
fun registerDenseAllocationCheck(name: String, description: String, vararg extraArgs: String) =
    tasks.register<JavaExec>(name) {
        group = "verification"
        this.description = description
        dependsOn("jvmTestClasses")
        classpath(jvmTestCompilation.output.allOutputs, configurations.getByName("jvmTestRuntimeClasspath"))
        mainClass.set("com.eignex.koblas.dense.SimdDenseAllocationCheck")
        javaLauncher.set(allocationCheckJavaLauncher)
        jvmArgs(
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
            "-XX:-TieredCompilation",
            "-XX:CompileThreshold=1000",
            *extraArgs,
        )
    }

val simdDenseAllocationCheck = registerDenseAllocationCheck(
    "simdDenseAllocationCheck",
    "Checks allocation-free JVM SIMD dense kernels and the whole calls around them, at this machine's width.",
)

// Two lanes rather than this host's four, and the unfused multiply-add a pre-FMA host would take. Both at
// once, because a step that takes two instructions instead of one is what leaves the most values live and
// the narrower species is the fleet's floor.
val simdDenseAllocationCheckNarrowNoFma = registerDenseAllocationCheck(
    "simdDenseAllocationCheckNarrowNoFma",
    "Checks the same kernels with a two-lane species and the fused multiply-add disabled.",
    "-XX:MaxVectorSize=16",
    "-XX:-UseFMA",
)

/**
 * Runs the panel conformance in a process whose species or multiply-add was forced to something other than
 * this machine's own.
 *
 * Both are fixed when the virtual machine starts, so a test task cannot reach either: the narrow-species and
 * unfused paths are generated and taken only in a process started this way. Forcing a flag exercises the
 * path; it says nothing about a machine that genuinely lacks the instruction.
 */
fun registerRuntimePathCheck(name: String, description: String, vararg extraArgs: String) =
    tasks.register<JavaExec>(name) {
        group = "verification"
        this.description = description
        dependsOn("jvmTestClasses")
        classpath(jvmTestCompilation.output.allOutputs, configurations.getByName("jvmTestRuntimeClasspath"))
        mainClass.set("com.eignex.koblas.dense.SimdRuntimePathCheck")
        javaLauncher.set(allocationCheckJavaLauncher)
        jvmArgs("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", *extraArgs)
    }

// Two lanes rather than this host's four, which is the width of the narrowest species in the reference
// fleet, and the unfused multiply-add that a pre-FMA host would take.
val simdNarrowSpeciesCheck = registerRuntimePathCheck(
    "simdNarrowSpeciesCheck",
    "Checks the JVM panels against a species constrained to two lanes.",
    "-XX:MaxVectorSize=16",
)
simdNarrowSpeciesCheck { args("2") }
val simdNoFmaCheck = registerRuntimePathCheck(
    "simdNoFmaCheck",
    "Checks the JVM panels with the fused multiply-add disabled.",
    "-XX:-UseFMA",
)
simdNoFmaCheck { args("", "false") }

tasks.named("check") {
    dependsOn(
        simdSparseAllocationCheck,
        simdDenseAllocationCheck,
        simdDenseAllocationCheckNarrowNoFma,
        simdNarrowSpeciesCheck,
        simdNoFmaCheck,
    )
}

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
