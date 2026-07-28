plugins {
    id("com.eignex.kmp") version "1.2.7"
}

eignexPublish {
    description.set("CBLAS/LAPACKE-backed LinearAlgebra for koblas native targets, resolving the host OpenBLAS at runtime.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
    applyDefaultHierarchyTemplate()
    // The extra release test binary serves the opt-in microbench (KOBLAS_MICROBENCH=1): debug-binary
    // timings understate the pure-Kotlin reference. Linked only on demand, not by check.
    linuxX64 {
        binaries.test("release", listOf(org.jetbrains.kotlin.gradle.plugin.mpp.NativeBuildType.RELEASE))
    }
    linuxArm64()
    macosX64(); macosArm64()

    sourceSets {
        commonMain.dependencies {
            api(project(":koblas"))
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
    }
}
