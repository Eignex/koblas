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
    jvm()
    js(IR) { browser(); nodejs() }
    wasmJs { browser(); nodejs() }
    wasmWasi { nodejs() }
    linuxX64(); linuxArm64()
    macosX64(); macosArm64(); mingwX64()
    iosX64(); iosArm64(); iosSimulatorArm64()

    sourceSets {
        val nonJvmMain by creating {
            dependsOn(commonMain.get())
        }
        nativeMain.get().dependsOn(nonJvmMain)
        wasmWasiMain.get().dependsOn(nonJvmMain)

        val posixMain by creating { dependsOn(nativeMain.get()) }
        appleMain.get().dependsOn(posixMain)
        linuxMain.get().dependsOn(posixMain)

        // The host-BLAS backend: Linux and macOS only. It resolves libopenblas with dlopen, which iOS
        // has no use for (no host library to find, and an App Store binary should not carry the call),
        // and mingw does not provide at all. Both keep the portable kernels.
        val hostBlasMain by creating { dependsOn(posixMain) }
        linuxMain.get().dependsOn(hostBlasMain)
        macosMain.get().dependsOn(hostBlasMain)
        // Everything else non-JVM: no host library to reach, so the primitives call the scalar loops
        // directly and pay nothing for a dispatch they can never use.
        val scalarOnlyMain by creating { dependsOn(nonJvmMain) }
        iosMain.get().dependsOn(scalarOnlyMain)
        mingwMain.get().dependsOn(scalarOnlyMain)
        webMain.get().dependsOn(scalarOnlyMain)

        val hostBlasTest by creating { dependsOn(nativeTest.get()) }
        linuxTest.get().dependsOn(hostBlasTest)
        macosTest.get().dependsOn(hostBlasTest)
        webMain.get().dependsOn(nonJvmMain)
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
tasks.withType<Test>().configureEach {
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
