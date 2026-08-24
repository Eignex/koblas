import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar
import java.io.File

plugins {
    id("com.eignex.jvm") version "1.3.2"
}

eignexPublish {
    description.set("Maven-hosted BASICLU simplex-basis factorization backend for koblas on the JVM.")
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
    else -> error("koblas-basiclu cannot build BASICLU for this host")
}
val basicluPlatform = providers.gradleProperty("koblas.basiclu.platform").orElse(hostPlatform)
val basicluResources = layout.buildDirectory.dir("basiclu/resources")
val lintOnly = gradle.startParameter.taskNames.let { names ->
    names.isNotEmpty() && names.all { it.substringAfterLast(':') == "lintDocs" }
}

fun toolVersion(command: String) = providers.exec { commandLine(command, "--version") }.standardOutput.asText

val cCompiler = providers.environmentVariable("CC").orElse("cc")
val buildBasiclu = tasks.register<Exec>("buildBasiclu") {
    val platform = basicluPlatform.get()
    require(platform in supportedPlatforms) { "unsupported BASICLU platform $platform" }
    inputs.file(layout.projectDirectory.file("basiclu.lock"))
    inputs.file(layout.projectDirectory.file("native/basiclu_shim.c"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/build-basiclu.sh"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/third-party-notices.sh"))
    inputs.property("platform", platform)
    inputs.property("cc", toolVersion(cCompiler.get()))
    outputs.dir(basicluResources.map { it.dir("org/eignex/basiclu/$platform") })
    outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
    commandLine(
        "bash", rootProject.layout.projectDirectory.file("scripts/build-basiclu.sh").asFile.absolutePath,
        "--platform", platform, "--output", basicluResources.get().asFile.absolutePath,
    )
    environment("CC" to cCompiler.get(), "CFLAGS" to "", "LDFLAGS" to "", "LC_ALL" to "C", "TZ" to "UTC")
}

sourceSets.named("main") { resources.srcDir(basicluResources) }
tasks.named("processResources") { if (!lintOnly) dependsOn(buildBasiclu) }
tasks.named<Jar>("sourcesJar") { dependsOn(buildBasiclu) }

val verifyBasicluResources = tasks.register("verifyBasicluResources") {
    val notices = basicluResources.get().file("THIRD-PARTY-NOTICES.txt").asFile
    inputs.file(notices)
    val requiredResources = supportedPlatforms.associateWith { platform ->
        listOf(if (platform.startsWith("linux")) "libkoblas_basiclu.so.1" else "libkoblas_basiclu.1.dylib", ".libraries", ".basiclu-source-sha256")
    }
    val resourceDirectories = supportedPlatforms.associateWith { platform ->
        basicluResources.get().dir("org/eignex/basiclu/$platform").asFile.absolutePath
    }
    inputs.files(resourceDirectories.values)
    inputs.property("requiredResources", requiredResources)
    inputs.property("resourceDirectories", resourceDirectories)
    doLast {
        check(notices.isFile && notices.length() > 0L) { "missing consolidated third-party notices for koblas-basiclu" }
        @Suppress("UNCHECKED_CAST") val required = inputs.properties.getValue("requiredResources") as Map<String, List<String>>
        @Suppress("UNCHECKED_CAST") val directories = inputs.properties.getValue("resourceDirectories") as Map<String, String>
        required.forEach { (platform, resources) ->
            val directory = File(directories.getValue(platform))
            check(resources.all { directory.resolve(it).isFile && directory.resolve(it).length() > 0L }) {
                "missing bundled BASICLU resources for $platform; build them on that platform before publishing"
            }
        }
    }
}

tasks.configureEach { if (name.startsWith("publish")) dependsOn(verifyBasicluResources) }
tasks.withType<Jar>().configureEach { manifest { attributes("Automatic-Module-Name" to "com.eignex.koblas.basiclu") } }
tasks.withType<Test>().configureEach { jvmArgs("--enable-native-access=ALL-UNNAMED") }
