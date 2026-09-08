plugins {
    id("com.eignex.jvm") version "1.3.3"
    id("koblas.native-library")
}

eignexPublish {
    description.set("Maven-hosted CBLAS loader for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies {
    api(project(":koblas"))
}

val nativePlatforms = listOf("linux-x86_64", "linux-arm64", "macosx-arm64")

koblasNativeLibrary {
    libraryName = "openBlas"
    resourcePackage = "org/bytedeco/openblas"
    lockFile = layout.projectDirectory.file("openblas.lock")
    buildScript = rootProject.layout.projectDirectory.file("scripts/build-openblas.sh")
    platformProperty = "koblas.openblas.platform"
    supportedPlatforms = nativePlatforms
    requiredResources = mapOf(
        "linux-x86_64" to listOf("libopenblas.so.0"),
        "linux-arm64" to listOf("libopenblas.so.0"),
        "macosx-arm64" to listOf("libopenblas.0.dylib"),
    ).mapValues { (platform, resources) -> resources + ".openblas-source-sha256" + ".openblas-build-options" }
    compiler("CC", "cc")
    tool("make", "make")
    testJvmArgs = listOf("--add-modules=jdk.incubator.vector")
}
