import org.gradle.api.publish.maven.MavenPublication
import org.gradle.api.publish.maven.tasks.GenerateMavenPom
import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar

plugins {
    id("com.eignex.jvm") version "1.3.1"
}

eignexPublish {
    description.set("GPL-3.0-only Maven-hosted SuiteSparse UMFPACK loader for koblas on the JVM.")
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

tasks.withType<Jar>().configureEach {
    manifest { attributes("Automatic-Module-Name" to "com.eignex.koblas.umfpack") }
}
tasks.withType<Test>().configureEach { jvmArgs("--enable-native-access=ALL-UNNAMED") }

afterEvaluate {
    publishing.publications.withType(MavenPublication::class.java).configureEach {
        pom.withXml {
            val license = asNode().appendNode("licenses").appendNode("license")
            license.appendNode("name", "GPL-3.0-only")
            license.appendNode("url", "https://www.gnu.org/licenses/gpl-3.0.html")
            license.appendNode("distribution", "repo")
        }
    }
}

tasks.withType<GenerateMavenPom>().configureEach {
    doLast {
        val pom = destination
        pom.writeText(
            pom.readText().replace(
                Regex("""  <licenses>\s*<license>\s*<name>Apache-2.0</name>.*?</licenses>\s*""", RegexOption.DOT_MATCHES_ALL),
                "",
            ),
        )
    }
}
