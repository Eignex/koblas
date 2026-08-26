plugins {
    id("com.eignex.jvm") version "1.3.2"
    id("koblas.native-library")
}

eignexPublish {
    description.set("Maven-hosted BASICLU simplex-basis factorization backend for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies {
    api(project(":koblas"))
}

val nativePlatforms = listOf("linux-x86_64", "linux-arm64", "macosx-arm64")

koblasNativeLibrary {
    libraryName = "basiclu"
    resourcePackage = "org/eignex/basiclu"
    lockFile = layout.projectDirectory.file("basiclu.lock")
    buildScript = rootProject.layout.projectDirectory.file("scripts/build-basiclu.sh")
    platformProperty = "koblas.basiclu.platform"
    supportedPlatforms = nativePlatforms
    requiredResources = nativePlatforms.associateWith { platform ->
        listOf(if (platform.startsWith("linux")) "libkoblas_basiclu.so.1" else "libkoblas_basiclu.1.dylib", ".libraries", ".basiclu-source-sha256")
    }
    compiler("CC", "cc")
}
