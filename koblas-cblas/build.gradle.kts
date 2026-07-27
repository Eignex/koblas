import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
    id("com.eignex.kmp") version "1.2.7"
}

eignexPublish {
    description.set("CBLAS/LAPACKE-backed LinearAlgebra for koblas native targets, linked against the installed OpenBLAS.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
    applyDefaultHierarchyTemplate()
    linuxX64(); linuxArm64()
    macosX64(); macosArm64()

    targets.withType<KotlinNativeTarget>().configureEach {
        compilations.getByName("main").cinterops.create("cblas") {
            definitionFile.set(project.file("src/nativeInterop/cinterop/cblas.def"))
        }
    }

    sourceSets {
        commonMain.dependencies {
            api(project(":koblas"))
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
    }
}
