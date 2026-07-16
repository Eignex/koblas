@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    id("com.eignex.kmp") version "1.2.7"
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
        webMain.get().dependsOn(nonJvmMain)
        wasmWasiMain.get().dependsOn(webMain.get())
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
