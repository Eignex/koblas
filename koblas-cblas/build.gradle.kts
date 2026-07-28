plugins {
    id("com.eignex.kmp") version "1.2.7"
}

eignexPublish {
    description.set("CBLAS/LAPACKE-backed LinearAlgebra for koblas native targets, resolving the host OpenBLAS at runtime.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
    applyDefaultHierarchyTemplate()
    linuxX64(); linuxArm64()
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
