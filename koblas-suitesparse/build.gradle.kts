plugins {
    id("com.eignex.jvm") version "1.3.2"
    id("koblas.native-library")
}

eignexPublish {
    description.set("Maven-hosted SuiteSparse loaders for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies {
    api(project(":koblas"))
    implementation(project(":koblas-openblas"))
}

val nativePlatforms = listOf("linux-x86_64", "linux-arm64", "macosx-arm64")
val openBlasBuildDirectory = project(":koblas-openblas").layout.buildDirectory.get().asFile
val blasResources = nativePlatforms.associateWith { platform ->
    openBlasBuildDirectory.resolve(
        "openblas/resources/org/bytedeco/openblas/$platform/" +
            if (platform.startsWith("linux")) "libopenblas.so.0" else "libopenblas.0.dylib",
    )
}

koblasNativeLibrary {
    libraryName = "suiteSparse"
    resourcePackage = "org/eignex/suitesparse"
    lockFile = layout.projectDirectory.file("suitesparse.lock")
    buildScript = rootProject.layout.projectDirectory.file("scripts/build-suitesparse.sh")
    platformProperty = "koblas.suitesparse.platform"
    supportedPlatforms = nativePlatforms
    requiredResources = nativePlatforms.associateWith { platform ->
        val libraries = if (platform.startsWith("linux")) {
            listOf("libklu.so.2", "libumfpack.so.6", "libcholmod.so.5", "libspqr.so.4")
        } else {
            listOf("libklu.2.dylib", "libumfpack.dylib", "libcholmod.5.dylib", "libspqr.4.dylib")
        }
        libraries + listOf(".libraries", ".suitesparse-source-sha256")
    }
    dependsOnTasks = listOf(":koblas-openblas:buildOpenBlas")
    extraInputFiles.from(layout.projectDirectory.file("suitesparse.lock"))
    extraInputFiles.from(blasResources.values)
    platformArguments = blasResources.mapValues { (_, blas) -> listOf("--blas", blas.absolutePath) }
    compiler("CC", "cc")
    compiler("CXX", "c++")
    tool("cmake", "cmake")
}
