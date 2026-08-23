import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar
import java.io.File

plugins {
    id("com.eignex.jvm") version "1.3.2"
}

eignexPublish {
    description.set("LGPL-2.1-or-later Maven-hosted SuiteSparse KLU loader for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
    licenses {
        license {
            name.set("Apache-2.0")
            url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
            distribution.set("repo")
        }
        license {
            name.set("BSD-3-Clause")
            url.set("https://opensource.org/license/bsd-3-clause")
            distribution.set("repo")
        }
        license {
            name.set("LGPL-2.1-or-later")
            url.set("https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html")
            distribution.set("repo")
        }
    }
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
    else -> error("koblas-klu cannot build SuiteSparse KLU for this host")
}
val kluPlatform = providers.gradleProperty("koblas.klu.platform").orElse(hostPlatform)
val kluResources = layout.buildDirectory.dir("klu/resources")
val lintOnly = gradle.startParameter.taskNames.let { names ->
    names.isNotEmpty() && names.all { it.substringAfterLast(':') == "lintDocs" }
}

fun toolVersion(command: String) = providers.exec { commandLine(command, "--version") }.standardOutput.asText

val cCompiler = providers.environmentVariable("CC").orElse("cc")
val buildKlu = tasks.register<Exec>("buildKlu") {
    val platform = kluPlatform.get()
    require(platform in supportedPlatforms) { "unsupported KLU platform $platform" }
    inputs.file(layout.projectDirectory.file("klu.lock"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/build-klu.sh"))
    inputs.property("platform", platform)
    inputs.property("cc", toolVersion(cCompiler.get()))
    inputs.property("cmake", toolVersion("cmake"))
    outputs.dir(kluResources.map { it.dir("org/eignex/klu/$platform") })
    outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
    commandLine(
        "bash", rootProject.layout.projectDirectory.file("scripts/build-klu.sh").asFile.absolutePath,
        "--platform", platform, "--output", kluResources.get().asFile.absolutePath,
    )
    environment(
        "CC" to cCompiler.get(), "CFLAGS" to "", "CXXFLAGS" to "", "LDFLAGS" to "",
        "LC_ALL" to "C", "TZ" to "UTC", "SOURCE_DATE_EPOCH" to "0",
    )
}

sourceSets.named("main") { resources.srcDir(kluResources) }
tasks.named("processResources") { if (!lintOnly) dependsOn(buildKlu) }
tasks.named<Jar>("sourcesJar") { dependsOn(buildKlu) }

val kluVersion = layout.projectDirectory.file("klu.lock").asFile.useLines { lines ->
    lines.first { it.startsWith("version=") }.removePrefix("version=")
}

val verifyKluResources = tasks.register("verifyKluResources") {
    val requiredResources = supportedPlatforms.associateWith { platform ->
        listOf(
            if (platform.startsWith("linux")) "libklu.so.2" else "libklu.2.dylib",
            ".libraries",
            ".suitesparse-source-sha256",
            "LICENSE.klu-$kluVersion.txt",
        )
    }
    val resourceDirectories = supportedPlatforms.associateWith { platform ->
        kluResources.get().dir("org/eignex/klu/$platform").asFile.absolutePath
    }
    inputs.files(resourceDirectories.values)
    inputs.property("requiredResources", requiredResources)
    inputs.property("resourceDirectories", resourceDirectories)
    doLast {
        @Suppress("UNCHECKED_CAST")
        val required = inputs.properties.getValue("requiredResources") as Map<String, List<String>>
        @Suppress("UNCHECKED_CAST")
        val directories = inputs.properties.getValue("resourceDirectories") as Map<String, String>
        required.forEach { (platform, resources) ->
            val directory = File(directories.getValue(platform))
            check(resources.all { directory.resolve(it).isFile && directory.resolve(it).length() > 0L }) {
                "missing bundled KLU resources for $platform; build them on that platform before publishing"
            }
        }
    }
}

tasks.configureEach {
    if (name.startsWith("publish")) dependsOn(verifyKluResources)
}

tasks.withType<Jar>().configureEach {
    manifest { attributes("Automatic-Module-Name" to "com.eignex.koblas.klu") }
}
tasks.withType<Test>().configureEach { jvmArgs("--enable-native-access=ALL-UNNAMED") }
