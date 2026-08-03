@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

plugins {
    id("com.eignex.kmp") version "1.2.7"
    kotlin("plugin.serialization") version "2.3.20"
}

eignexPublish {
    description.set("Dense/sparse linear algebra for Kotlin/KMP with a pluggable BLAS/LAPACK backend seam.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
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
    js(IR) { browser(); nodejs() }
    wasmJs { browser(); nodejs() }
    wasmWasi { nodejs() }
    linuxX64(); linuxArm64()
    macosX64(); macosArm64(); mingwX64()
    iosX64(); iosArm64(); iosSimulatorArm64()

    sourceSets {
        // Everything that is not the JVM: scalar primitive leaves instead of Vector API ones, plus the
        // per-platform threshold defaults and backend reporting that follow from that.
        val scalarMain by creating {
            dependsOn(commonMain.get())
        }
        nativeMain.get().dependsOn(scalarMain)
        wasmWasiMain.get().dependsOn(scalarMain)
        iosMain.get().dependsOn(scalarMain)
        mingwMain.get().dependsOn(scalarMain)

        // The host-BLAS backend: Linux and macOS only. It resolves libopenblas with dlopen, which iOS
        // has no use for (no host library to find, and an App Store binary should not carry the call),
        // and mingw does not provide at all. Both keep the portable kernels, and need no source set of
        // their own to say so - the level-1 routing lives in commonMain and resolves to null for them.
        val hostBlasMain by creating { dependsOn(nativeMain.get()) }
        linuxMain.get().dependsOn(hostBlasMain)
        macosMain.get().dependsOn(hostBlasMain)

        val hostBlasTest by creating { dependsOn(nativeTest.get()) }
        linuxTest.get().dependsOn(hostBlasTest)
        macosTest.get().dependsOn(hostBlasTest)
        webMain.get().dependsOn(scalarMain)
        wasmWasiMain.get().dependsOn(webMain.get())

        commonMain.dependencies {
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.10.0")
        }
        commonTest.dependencies {
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-core:1.10.0")
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.10.0")
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

// A stable module name, so a modular consumer sees a named module rather than one named after the jar
// file. That is what lets native access be granted per module instead of blanket ALL-UNNAMED.
tasks.named<Jar>("jvmJar") {
    manifest {
        attributes("Automatic-Module-Name" to "com.eignex.koblas")
    }
}
