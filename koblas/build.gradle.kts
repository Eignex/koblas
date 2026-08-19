@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

plugins {
    id("com.eignex.kmp") version "1.3.1"
    kotlin("plugin.serialization") version "2.4.10"
}

eignexPublish {
    description.set("Dense/sparse linear algebra for Kotlin/KMP with a pluggable BLAS/LAPACK backend seam.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
    explicitApi()
    applyDefaultHierarchyTemplate()
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }
    // The JVM host-BLAS backend binds through java.lang.foreign, finalized in 22 and used here with
    // Linker.Option.critical. 25 is the current LTS-track release; this is the floor for JVM consumers.
    jvmToolchain(25)
    jvm {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_25)
        }
    }
    js { browser(); nodejs() }
    wasmJs { browser(); nodejs() }
    wasmWasi { nodejs() }
    linuxX64(); linuxArm64()
    macosArm64(); mingwX64()
    iosX64(); iosArm64(); iosSimulatorArm64()

    sourceSets {
        // Everything that is not the JVM: scalar primitive leaves instead of Vector API ones, plus the
        // per-platform threshold defaults and backend reporting that follow from that.
        val scalarMain = create("scalarMain") {
            dependsOn(commonMain.get())
        }
        nativeMain.get().dependsOn(scalarMain)
        wasmWasiMain.get().dependsOn(scalarMain)

        // The host-BLAS backend: Linux and macOS only. It resolves libopenblas with dlopen, which iOS
        // has no use for (no host library to find, and an App Store binary should not carry the call),
        // and mingw does not provide at all. Both keep the portable kernels, and need no source set of
        // their own to say so - the level-1 routing lives in commonMain and resolves to null for them.
        val hostBlasMain = create("hostBlasMain") { dependsOn(nativeMain.get()) }
        linuxMain.get().dependsOn(hostBlasMain)
        macosMain.get().dependsOn(hostBlasMain)

        val hostBlasTest = create("hostBlasTest") { dependsOn(nativeTest.get()) }
        linuxTest.get().dependsOn(hostBlasTest)
        macosTest.get().dependsOn(hostBlasTest)

        // The host-SuiteSparse backend, on the same targets and for the same reason as hostBlasMain. Its own
        // source set rather than a package inside that one because the two libraries are independent: a host
        // may have OpenBLAS without SuiteSparse or the reverse, and a sparse direct solver is not BLAS.
        val hostSparseMain = create("hostSparseMain") { dependsOn(nativeMain.get()) }
        linuxMain.get().dependsOn(hostSparseMain)
        macosMain.get().dependsOn(hostSparseMain)

        val hostSparseTest = create("hostSparseTest") { dependsOn(nativeTest.get()) }
        linuxTest.get().dependsOn(hostSparseTest)
        macosTest.get().dependsOn(hostSparseTest)

        // Backend discovery has to live where it can see the bindings, and the two host source sets are
        // siblings, so the platforms that have both get a source set that depends on both. Everything else
        // takes the no-op from noHostMain. Exactly one of the two reaches each target, which is what
        // `registerPlatformBackends` being an expect declaration requires.
        val hostBackendsMain = create("hostBackendsMain") {
            dependsOn(hostBlasMain)
            dependsOn(hostSparseMain)
        }
        linuxMain.get().dependsOn(hostBackendsMain)
        macosMain.get().dependsOn(hostBackendsMain)

        val noHostMain = create("noHostMain") { dependsOn(scalarMain) }
        mingwMain.get().dependsOn(noHostMain)
        iosMain.get().dependsOn(noHostMain)

        webMain.get().dependsOn(noHostMain)
        wasmWasiMain.get().dependsOn(webMain.get())

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
// FFM downcalls are restricted methods: a warning on 25, an error later. Consumers pass the same flag
// when they want the host-BLAS backend; core koblas needs it only because the backend now ships inside.
tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (project.findProperty("koblas.noSimd") != "true") {
        jvmArgs("--add-modules=jdk.incubator.vector")
    }
}

// Tests marked @Category(HostLibraryTest) need a real OpenBLAS or SuiteSparse, so they are out of the
// default run and opted into with -Pkoblas.hostTests=true. They measure the machine as much as the library,
// so including them would make the everyday result mean different things on a box with SuiteSparse and one
// without. Excluding them and pinning the backend to `reference` keeps that out; the opt-in run accepts the
// noise in exchange for exercising the bindings, and reports coverage like any other run.
tasks.withType<Test>().configureEach {
    if (project.findProperty("koblas.hostTests") == "true") return@configureEach
    systemProperty("koblas.backend", "reference")
    useJUnit {
        excludeCategories("com.eignex.koblas.HostLibraryTest")
    }
}

// Kotlin emits a `$DefaultImpls` holder for every interface with a body, and a bridge for every method with
// a default argument. Neither is reachable from Kotlin call sites, so both count as permanently uncovered and
// make the report read as though tested code were not: `Blas.syr` shows its real body at 98% next to a bridge
// at 0%. Dropping the holders leaves the report describing code a test can actually reach.
kover {
    reports {
        filters {
            excludes {
                classes("*\$DefaultImpls")
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

// CI runs lintDocs as its own step, so `check` passing locally did not mean CI would: an unresolved KDoc
// link fails the Dokka pass and nothing else catches it. Wiring it in makes one local command the same gate.
tasks.named("check") {
    dependsOn(tasks.named("lintDocs"))
}
