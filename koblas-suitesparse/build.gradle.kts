import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar
import java.io.File

plugins {
    id("com.eignex.jvm") version "1.3.2"
}

eignexPublish {
    description.set("Maven-hosted SuiteSparse loaders for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies {
    api(project(":koblas"))
    implementation(project(":koblas-openblas"))
}

val supportedPlatforms = listOf("linux-x86_64", "linux-arm64", "macosx-arm64")
val hostPlatform = when {
    System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("amd64", "x86_64") -> "linux-x86_64"
    System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("aarch64", "arm64") -> "linux-arm64"
    System.getProperty("os.name").startsWith("Mac", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("aarch64", "arm64") -> "macosx-arm64"
    else -> error("koblas-suitesparse cannot build SuiteSparse for this host")
}
val suiteSparsePlatform = providers.gradleProperty("koblas.suitesparse.platform").orElse(hostPlatform)
val suiteSparseResources = layout.buildDirectory.dir("suitesparse/resources")
val lintOnly = gradle.startParameter.taskNames.let { names ->
    names.isNotEmpty() && names.all { it.substringAfterLast(':') == "lintDocs" }
}

fun toolVersion(command: String) = providers.exec { commandLine(command, "--version") }.standardOutput.asText

val cCompiler = providers.environmentVariable("CC").orElse("cc")

// SPQR is C++, so this build wants a C++ compiler alongside the C one every other package uses.
val cxxCompiler = providers.environmentVariable("CXX").orElse("c++")
val buildSuiteSparse = tasks.register<Exec>("buildSuiteSparse") {
    val platform = suiteSparsePlatform.get()
    require(platform in supportedPlatforms) { "unsupported SuiteSparse platform $platform" }
    inputs.file(layout.projectDirectory.file("suitesparse.lock"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/build-suitesparse.sh"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/third-party-notices.sh"))
    inputs.file(rootProject.project(":koblas-openblas").layout.projectDirectory.file("openblas.lock"))
    inputs.property("platform", platform)
    inputs.property("cc", toolVersion(cCompiler.get()))
    inputs.property("cxx", toolVersion(cxxCompiler.get()))
    inputs.property("cmake", toolVersion("cmake"))
    outputs.dir(suiteSparseResources.map { it.dir("org/eignex/suitesparse/$platform") })
    outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
    dependsOn(":koblas-openblas:buildOpenBlas")
    val blas = project(":koblas-openblas").layout.buildDirectory
        .file("openblas/resources/org/bytedeco/openblas/$platform/${if (platform.startsWith("linux")) "libopenblas.so.0" else "libopenblas.0.dylib"}")
    inputs.file(blas)
    commandLine(
        "bash", rootProject.layout.projectDirectory.file("scripts/build-suitesparse.sh").asFile.absolutePath,
        "--platform", platform, "--output", suiteSparseResources.get().asFile.absolutePath,
        "--blas", blas.get().asFile.absolutePath,
    )
    environment(
        "CC" to cCompiler.get(), "CXX" to cxxCompiler.get(),
        "CFLAGS" to "", "CXXFLAGS" to "", "LDFLAGS" to "",
        "LC_ALL" to "C", "TZ" to "UTC", "SOURCE_DATE_EPOCH" to "0",
    )
}

sourceSets.named("main") { resources.srcDir(suiteSparseResources) }
tasks.named("processResources") { if (!lintOnly) dependsOn(buildSuiteSparse) }
tasks.named<Jar>("sourcesJar") { dependsOn(buildSuiteSparse) }

// CHOLMOD and SPQR carry no binding yet, so this task is the only thing that would notice them going
// missing from a platform's build.
val bundledLibraries = mapOf(
    "linux" to listOf("libklu.so.2", "libumfpack.so.6", "libcholmod.so.5", "libspqr.so.4"),
    "macos" to listOf("libklu.2.dylib", "libumfpack.dylib", "libcholmod.5.dylib", "libspqr.4.dylib"),
)

val verifySuiteSparseResources = tasks.register("verifySuiteSparseResources") {
    dependsOn(buildSuiteSparse)
    val notices = suiteSparseResources.get().file("THIRD-PARTY-NOTICES.txt").asFile
    inputs.file(notices)
    val requiredResources = supportedPlatforms.associateWith { platform ->
        val libraries = bundledLibraries.getValue(if (platform.startsWith("linux")) "linux" else "macos")
        libraries + listOf(".libraries", ".suitesparse-source-sha256")
    }
    val resourceDirectories = supportedPlatforms.associateWith { platform ->
        suiteSparseResources.get().dir("org/eignex/suitesparse/$platform").asFile.absolutePath
    }
    inputs.files(resourceDirectories.values)
    inputs.property("requiredResources", requiredResources)
    inputs.property("resourceDirectories", resourceDirectories)
    doLast {
        check(notices.isFile && notices.length() > 0L) {
            "missing consolidated third-party notices for koblas-suitesparse"
        }
        @Suppress("UNCHECKED_CAST")
        val required = inputs.properties.getValue("requiredResources") as Map<String, List<String>>
        @Suppress("UNCHECKED_CAST")
        val directories = inputs.properties.getValue("resourceDirectories") as Map<String, String>
        required.forEach { (platform, resources) ->
            val directory = File(directories.getValue(platform))
            val missing = resources.filterNot {
                directory.resolve(it).isFile && directory.resolve(it).length() > 0L
            }
            check(missing.isEmpty()) {
                "missing bundled SuiteSparse resources for $platform: ${missing.joinToString()}; " +
                    "build them on that platform before publishing"
            }
        }
    }
}

tasks.configureEach {
    if (name.startsWith("publish")) dependsOn(verifySuiteSparseResources)
}

tasks.withType<Jar>().configureEach {
    manifest { attributes("Automatic-Module-Name" to "com.eignex.koblas.suitesparse") }
}
tasks.withType<Test>().configureEach { jvmArgs("--enable-native-access=ALL-UNNAMED") }
