plugins {
    id("com.eignex.jvm") version "1.3.2"
    id("koblas.native-library")
}

eignexPublish {
    description.set("Maven-hosted HiGHS HFactor basis-solver backend for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies {
    api(project(":koblas"))
}

val nativePlatforms = listOf("linux-x86_64", "linux-arm64", "macosx-arm64")

koblasNativeLibrary {
    libraryName = "hfactor"
    resourcePackage = "org/eignex/hfactor"
    lockFile = layout.projectDirectory.file("hfactor.lock")
    buildScript = rootProject.layout.projectDirectory.file("scripts/build-hfactor.sh")
    platformProperty = "koblas.hfactor.platform"
    supportedPlatforms = nativePlatforms
    requiredResources = nativePlatforms.associateWith { platform ->
        listOf(if (platform.startsWith("linux")) "libkoblas_hfactor.so.1" else "libkoblas_hfactor.1.dylib", ".libraries", ".hfactor-source-sha256")
    }
    extraInputFiles.from(layout.projectDirectory.dir("native"))
    compiler("CXX", "c++")
}
