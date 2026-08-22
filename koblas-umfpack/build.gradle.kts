import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar
import java.io.File

plugins {
    id("com.eignex.jvm") version "1.3.2"
}

eignexPublish {
    description.set("GPL-3.0-only Maven-hosted SuiteSparse UMFPACK loader for koblas on the JVM.")
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
            name.set("GPL-2.0-or-later")
            url.set("https://www.gnu.org/licenses/old-licenses/gpl-2.0.html")
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
    else -> error("koblas-umfpack cannot build SuiteSparse for this host")
}
val umfpackPlatform = providers.gradleProperty("koblas.umfpack.platform").orElse(hostPlatform)
val umfpackResources = layout.buildDirectory.dir("umfpack/resources")
val lintOnly = gradle.startParameter.taskNames.let { names ->
    names.isNotEmpty() && names.all { it.substringAfterLast(':') == "lintDocs" }
}

fun toolVersion(command: String) = providers.exec { commandLine(command, "--version") }.standardOutput.asText

val cCompiler = providers.environmentVariable("CC").orElse("cc")
val buildUmfpack = tasks.register<Exec>("buildUmfpack") {
    val platform = umfpackPlatform.get()
    require(platform in supportedPlatforms) { "unsupported UMFPACK platform $platform" }
    inputs.file(layout.projectDirectory.file("umfpack.lock"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/build-umfpack.sh"))
    inputs.property("platform", platform)
    inputs.property("cc", toolVersion(cCompiler.get()))
    inputs.property("cmake", toolVersion("cmake"))
    outputs.dir(umfpackResources.map { it.dir("org/eignex/umfpack/$platform") })
    outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
    dependsOn(":koblas-openblas:buildOpenBlas")
    val blas = project(":koblas-openblas").layout.buildDirectory
        .file("openblas/resources/org/bytedeco/openblas/$platform/${if (platform.startsWith("linux")) "libopenblas.so.0" else "libopenblas.0.dylib"}")
    inputs.file(blas)
    commandLine(
        "bash", rootProject.layout.projectDirectory.file("scripts/build-umfpack.sh").asFile.absolutePath,
        "--platform", platform, "--output", umfpackResources.get().asFile.absolutePath,
        "--blas", blas.get().asFile.absolutePath,
    )
    environment(
        "CC" to cCompiler.get(), "CFLAGS" to "", "CXXFLAGS" to "", "LDFLAGS" to "",
        "LC_ALL" to "C", "TZ" to "UTC", "SOURCE_DATE_EPOCH" to "0",
    )
}

sourceSets.named("main") { resources.srcDir(umfpackResources) }
tasks.named("processResources") { if (!lintOnly) dependsOn(buildUmfpack) }

val umfpackVersion = layout.projectDirectory.file("umfpack.lock").asFile.useLines { lines ->
    lines.first { it.startsWith("version=") }.removePrefix("version=")
}

val verifyUmfpackResources = tasks.register("verifyUmfpackResources") {
    val requiredResources = supportedPlatforms.associateWith { platform ->
        listOf(
            if (platform.startsWith("linux")) "libumfpack.so.6" else "libumfpack.dylib",
            ".libraries",
            ".suitesparse-source-sha256",
            "LICENSE.suitesparse-$umfpackVersion.txt",
        )
    }
    val resourceDirectories = supportedPlatforms.associateWith { platform ->
        umfpackResources.get().dir("org/eignex/umfpack/$platform").asFile.absolutePath
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
                "missing bundled UMFPACK resources for $platform; build them on that platform before publishing"
            }
        }
    }
}

tasks.configureEach {
    if (name.startsWith("publish")) dependsOn(verifyUmfpackResources)
}

tasks.withType<Jar>().configureEach {
    manifest { attributes("Automatic-Module-Name" to "com.eignex.koblas.umfpack") }
}
tasks.withType<Test>().configureEach { jvmArgs("--enable-native-access=ALL-UNNAMED") }
