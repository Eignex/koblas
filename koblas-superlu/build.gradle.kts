import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar
import java.io.File

plugins {
    id("com.eignex.jvm") version "1.3.2"
}

eignexPublish {
    description.set("BSD-3-Clause Maven-hosted SuperLU loader for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
    licenses {
        license {
            name.set("BSD-3-Clause")
            url.set("https://opensource.org/license/bsd-3-clause")
            distribution.set("repo")
        }
    }
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
    else -> error("koblas-superlu cannot build SuperLU for this host")
}
val superluPlatform = providers.gradleProperty("koblas.superlu.platform").orElse(hostPlatform)
val superluResources = layout.buildDirectory.dir("superlu/resources")
val lintOnly = gradle.startParameter.taskNames.let { names ->
    names.isNotEmpty() && names.all { it.substringAfterLast(':') == "lintDocs" }
}

fun toolVersion(command: String) = providers.exec { commandLine(command, "--version") }.standardOutput.asText

val cCompiler = providers.environmentVariable("CC").orElse("cc")
val buildSuperlu = tasks.register<Exec>("buildSuperlu") {
    val platform = superluPlatform.get()
    require(platform in supportedPlatforms) { "unsupported SuperLU platform $platform" }
    inputs.file(layout.projectDirectory.file("superlu.lock"))
    inputs.file(layout.projectDirectory.file("native/koblas_superlu.c"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/build-superlu.sh"))
    inputs.property("platform", platform)
    inputs.property("cc", toolVersion(cCompiler.get()))
    inputs.property("cmake", toolVersion("cmake"))
    outputs.dir(superluResources.map { it.dir("org/eignex/superlu/$platform") })
    outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
    dependsOn(":koblas-openblas:buildOpenBlas")
    val blas = project(":koblas-openblas").layout.buildDirectory
        .file("openblas/resources/org/bytedeco/openblas/$platform/${if (platform.startsWith("linux")) "libopenblas.so.0" else "libopenblas.0.dylib"}")
    inputs.file(blas)
    commandLine(
        "bash", rootProject.layout.projectDirectory.file("scripts/build-superlu.sh").asFile.absolutePath,
        "--platform", platform, "--output", superluResources.get().asFile.absolutePath,
        "--blas", blas.get().asFile.absolutePath,
    )
    environment(
        "CC" to cCompiler.get(), "CFLAGS" to "", "CXXFLAGS" to "", "LDFLAGS" to "",
        "LC_ALL" to "C", "TZ" to "UTC", "SOURCE_DATE_EPOCH" to "0",
    )
}

sourceSets.named("main") { resources.srcDir(superluResources) }
tasks.named("processResources") { if (!lintOnly) dependsOn(buildSuperlu) }

val superluVersion = layout.projectDirectory.file("superlu.lock").asFile.useLines { lines ->
    lines.first { it.startsWith("version=") }.removePrefix("version=")
}

val verifySuperluResources = tasks.register("verifySuperluResources") {
    val requiredResources = supportedPlatforms.associateWith { platform ->
        listOf(
            if (platform.startsWith("linux")) "libkoblas-superlu.so" else "libkoblas-superlu.dylib",
            ".libraries",
            ".superlu-source-sha256",
            "LICENSE.superlu-$superluVersion.txt",
        )
    }
    val resourceDirectories = supportedPlatforms.associateWith { platform ->
        superluResources.get().dir("org/eignex/superlu/$platform").asFile.absolutePath
    }
    inputs.files(resourceDirectories.values)
    inputs.property("requiredResources", requiredResources)
    inputs.property("resourceDirectories", resourceDirectories)
    doLast {
        @Suppress("UNCHECKED_CAST")
        val required = inputs.properties.getValue("requiredResources") as Map<String, List<String>>
        @Suppress("UNCHECKED_CAST")
        val directories = inputs.properties.getValue("resourceDirectories") as Map<String, String>
        for ((platform, names) in required) {
            val directory = File(directories.getValue(platform))
            check(directory.isDirectory) {
                "missing bundled SuperLU resources for $platform; build them on that platform before publishing"
            }
            for (name in names) check(File(directory, name).isFile) { "missing $name for $platform" }
        }
    }
}

tasks.withType<Jar>().configureEach {
    manifest { attributes("Automatic-Module-Name" to "com.eignex.koblas.superlu") }
}
tasks.configureEach { if (name.startsWith("publish")) dependsOn(verifySuperluResources) }
