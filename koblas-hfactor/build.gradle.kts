import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar
import org.gradle.api.tasks.testing.Test

plugins {
    id("com.eignex.jvm") version "1.3.3"
}

eignexPublish {
    description.set("Optional HiGHS HFactor sparse and basis solver API for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies {
    api(project(":koblas"))
}

val nativePlatforms = listOf("linux-x86_64", "linux-arm64", "macosx-arm64")
val hostPlatform = providers.systemProperty("os.name").zip(providers.systemProperty("os.arch")) { osName, architecture ->
    when {
        osName.startsWith("Linux", ignoreCase = true) && architecture in setOf("amd64", "x86_64") ->
            "linux-x86_64"
        osName.startsWith("Linux", ignoreCase = true) && architecture in setOf("aarch64", "arm64") ->
            "linux-arm64"
        osName.startsWith("Mac", ignoreCase = true) && architecture in setOf("aarch64", "arm64") ->
            "macosx-arm64"
        else -> error("unsupported koblas-hfactor host $osName/$architecture")
    }
}
val hfactorPlatform = providers.gradleProperty("koblas.hfactor.platform").orElse(hostPlatform)
val hfactorResources = layout.buildDirectory.dir("hfactor/resources")
val hfactorBuildScript = layout.projectDirectory.file("native/build.sh")
val cxxCompiler = providers.environmentVariable("CXX").orElse("c++")
val cxxVersion = providers.exec {
    commandLine(cxxCompiler.get(), "--version")
}.standardOutput.asText
val lintOnly = gradle.startParameter.taskNames.let { names ->
    names.isNotEmpty() && names.all { it.substringAfterLast(':') == "lintDocs" }
}

val buildHfactor = tasks.register<Exec>("buildHfactor") {
    val output = hfactorResources.get().asFile.absolutePath
    val platform = hfactorPlatform.get()
    inputs.file(layout.projectDirectory.file("hfactor.lock"))
    inputs.file(hfactorBuildScript)
    inputs.file(rootProject.layout.projectDirectory.file("scripts/third-party-notices.sh"))
    inputs.dir(layout.projectDirectory.dir("native"))
    inputs.property("platform", hfactorPlatform)
    inputs.property("resourceLayout", "complete-resources-v1")
    inputs.property("os.name", providers.systemProperty("os.name"))
    inputs.property("os.arch", providers.systemProperty("os.arch"))
    inputs.property("CXX", cxxCompiler)
    inputs.property("tool-cxx", cxxVersion)
    outputs.dir(hfactorResources)
    outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
    commandLine(
        "bash",
        hfactorBuildScript.asFile.absolutePath,
        "--platform",
        platform,
        "--output",
        output,
    )
    environment("CXX", cxxCompiler.get())
    environment(
        mapOf(
            "CFLAGS" to "",
            "CXXFLAGS" to "",
            "FFLAGS" to "",
            "LDFLAGS" to "",
            "MAKEFLAGS" to "",
            "LC_ALL" to "C",
            "TZ" to "UTC",
            "SOURCE_DATE_EPOCH" to "0",
        ),
    )
}

sourceSets.named("main") { resources.srcDir(hfactorResources) }
tasks.named("processResources") {
    if (!lintOnly) dependsOn(buildHfactor)
}
tasks.named<Jar>("sourcesJar") { dependsOn(buildHfactor) }

val requiredHfactorResources = nativePlatforms.flatMap { platform ->
    val library = if (platform.startsWith("linux")) "libkoblas_hfactor.so.1" else "libkoblas_hfactor.1.dylib"
    listOf(library, ".libraries", ".hfactor-source-sha256").map { resource ->
        hfactorResources.map { it.file("org/eignex/hfactor/$platform/$resource") }
    }
}
val verifyHfactorResources = tasks.register<Exec>("verifyHfactorResources") {
    dependsOn(buildHfactor)
    val notices = hfactorResources.map { it.file("THIRD-PARTY-NOTICES.txt") }
    inputs.file(notices)
    inputs.files(requiredHfactorResources)
    commandLine(
        "bash",
        "-c",
        "set -euo pipefail; notices=\"\$1\"; shift; " +
            "test -s \"\$notices\" || { echo \"missing consolidated third-party notices for koblas-hfactor\" >&2; exit 1; }; " +
            "for resource in \"\$@\"; do test -s \"\$resource\" || { echo \"missing bundled hfactor resource: \$resource\" >&2; exit 1; }; done",
        "verify-hfactor-resources",
        notices.get().asFile.absolutePath,
        *requiredHfactorResources.map { it.get().asFile.absolutePath }.toTypedArray(),
    )
}

tasks.configureEach {
    if (name.startsWith("publish")) dependsOn(verifyHfactorResources)
}
tasks.withType<Jar>().configureEach {
    manifest { attributes("Automatic-Module-Name" to "com.eignex.koblas.hfactor") }
}
tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
