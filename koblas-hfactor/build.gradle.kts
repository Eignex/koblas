import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar
import java.io.File

plugins {
    id("com.eignex.jvm") version "1.3.2"
}

eignexPublish {
    description.set("Maven-hosted HiGHS HFactor basis-solver backend for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies { api(project(":koblas")) }

val supportedPlatforms = listOf("linux-x86_64", "linux-arm64", "macosx-arm64")
val hostPlatform = when {
    System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("amd64", "x86_64") -> "linux-x86_64"
    System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("aarch64", "arm64") -> "linux-arm64"
    System.getProperty("os.name").startsWith("Mac", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("aarch64", "arm64") -> "macosx-arm64"
    else -> error("koblas-hfactor cannot build HFactor for this host")
}
val hfactorPlatform = providers.gradleProperty("koblas.hfactor.platform").orElse(hostPlatform)
val hfactorResources = layout.buildDirectory.dir("hfactor/resources")
val lintOnly = gradle.startParameter.taskNames.let { names ->
    names.isNotEmpty() && names.all { it.substringAfterLast(':') == "lintDocs" }
}

fun toolVersion(command: String) = providers.exec { commandLine(command, "--version") }.standardOutput.asText

// HFactor is C++, unlike every other library koblas bundles, so this build wants a C++ compiler.
val cxxCompiler = providers.environmentVariable("CXX").orElse("c++")
val buildHfactor = tasks.register<Exec>("buildHfactor") {
    val platform = hfactorPlatform.get()
    require(platform in supportedPlatforms) { "unsupported HFactor platform $platform" }
    inputs.file(layout.projectDirectory.file("hfactor.lock"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/build-hfactor.sh"))
    inputs.dir(layout.projectDirectory.dir("native"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/third-party-notices.sh"))
    inputs.property("platform", platform)
    inputs.property("cxx", toolVersion(cxxCompiler.get()))
    outputs.dir(hfactorResources.map { it.dir("org/eignex/hfactor/$platform") })
    outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
    commandLine(
        "bash", rootProject.layout.projectDirectory.file("scripts/build-hfactor.sh").asFile.absolutePath,
        "--platform", platform, "--output", hfactorResources.get().asFile.absolutePath,
    )
    environment("CXX" to cxxCompiler.get(), "CXXFLAGS" to "", "LDFLAGS" to "", "LC_ALL" to "C", "TZ" to "UTC")
}

sourceSets.named("main") { resources.srcDir(hfactorResources) }
tasks.named("processResources") { if (!lintOnly) dependsOn(buildHfactor) }
tasks.named<Jar>("sourcesJar") { dependsOn(buildHfactor) }

val verifyHfactorResources = tasks.register("verifyHfactorResources") {
    dependsOn(buildHfactor)
    val notices = hfactorResources.get().file("THIRD-PARTY-NOTICES.txt").asFile
    inputs.file(notices)
    val requiredResources = supportedPlatforms.associateWith { platform ->
        listOf(if (platform.startsWith("linux")) "libkoblas_hfactor.so.1" else "libkoblas_hfactor.1.dylib", ".libraries", ".hfactor-source-sha256")
    }
    val resourceDirectories = supportedPlatforms.associateWith { platform ->
        hfactorResources.get().dir("org/eignex/hfactor/$platform").asFile.absolutePath
    }
    inputs.files(resourceDirectories.values)
    inputs.property("requiredResources", requiredResources)
    inputs.property("resourceDirectories", resourceDirectories)
    doLast {
        check(notices.isFile && notices.length() > 0L) { "missing consolidated third-party notices for koblas-hfactor" }
        @Suppress("UNCHECKED_CAST") val required = inputs.properties.getValue("requiredResources") as Map<String, List<String>>
        @Suppress("UNCHECKED_CAST") val directories = inputs.properties.getValue("resourceDirectories") as Map<String, String>
        required.forEach { (platform, resources) ->
            val directory = File(directories.getValue(platform))
            check(resources.all { directory.resolve(it).isFile && directory.resolve(it).length() > 0L }) {
                "missing bundled HFactor resources for $platform; build them on that platform before publishing"
            }
        }
    }
}

tasks.configureEach { if (name.startsWith("publish")) dependsOn(verifyHfactorResources) }
tasks.withType<Jar>().configureEach { manifest { attributes("Automatic-Module-Name" to "com.eignex.koblas.hfactor") } }
tasks.withType<Test>().configureEach { jvmArgs("--enable-native-access=ALL-UNNAMED") }
