import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Jar

plugins {
    id("com.eignex.jvm") version "1.3.2"
}

eignexPublish {
    description.set("Maven-hosted OpenBLAS and LAPACKE loader for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies { api(project(":koblas")) }

val supportedPlatforms = listOf("linux-x86_64", "linux-arm64", "macosx-arm64")
val requiredResources = mapOf(
    "linux-x86_64" to listOf("libopenblas.so.0", "libgfortran.so.5", "libquadmath.so.0", "libgcc_s.so.1"),
    "linux-arm64" to listOf("libopenblas.so.0", "libgfortran.so.5", "libgcc_s.so.1"),
    "macosx-arm64" to listOf(
        "libopenblas.0.dylib",
        "libgfortran.dylib",
        "libgfortran.5.dylib",
        "libquadmath.0.dylib",
        "libgcc_s.1.1.dylib",
    ),
)
val hostPlatform = when {
    System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("amd64", "x86_64") -> "linux-x86_64"
    System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("aarch64", "arm64") -> "linux-arm64"
    System.getProperty("os.name").startsWith("Mac", ignoreCase = true) &&
        System.getProperty("os.arch") in setOf("aarch64", "arm64") -> "macosx-arm64"
    else -> error("koblas-openblas cannot build OpenBLAS for this host")
}
val openBlasPlatform = providers.gradleProperty("koblas.openblas.platform").orElse(hostPlatform)
val openBlasResources = layout.buildDirectory.dir("openblas/resources")
val lintOnly = gradle.startParameter.taskNames.let { taskNames ->
    taskNames.isNotEmpty() && taskNames.all { it.substringAfterLast(':') == "lintDocs" }
}
fun toolVersion(command: String) = providers.exec {
    commandLine(command, "--version")
}.standardOutput.asText

val fortranCompiler = providers.environmentVariable("FC").orElse("gfortran")
val cCompiler = providers.environmentVariable("CC").orElse("cc")

val buildOpenBlas = tasks.register<Exec>("buildOpenBlas") {
    val platform = openBlasPlatform.get()
    require(platform in supportedPlatforms) { "unsupported OpenBLAS platform $platform" }
    inputs.file(layout.projectDirectory.file("openblas.lock"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/build-openblas.sh"))
    inputs.file(rootProject.layout.projectDirectory.file("scripts/third-party-notices.sh"))
    inputs.property("platform", platform)
    inputs.property("cc", toolVersion(cCompiler.get()))
    inputs.property("fortran", toolVersion(fortranCompiler.get()))
    inputs.property("make", toolVersion("make"))
    inputs.property("os.name", System.getProperty("os.name"))
    inputs.property("os.arch", System.getProperty("os.arch"))
    outputs.dir(openBlasResources.map { it.dir("org/bytedeco/openblas/$platform") })
    outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
    commandLine(
        "bash",
        rootProject.layout.projectDirectory.file("scripts/build-openblas.sh").asFile.absolutePath,
        "--platform", platform,
        "--output", openBlasResources.get().asFile.absolutePath,
    )
    environment(
        "CC" to cCompiler.get(),
        "FC" to fortranCompiler.get(),
        "CFLAGS" to "",
        "CXXFLAGS" to "",
        "FFLAGS" to "",
        "LDFLAGS" to "",
        "MAKEFLAGS" to "",
        "LC_ALL" to "C",
        "TZ" to "UTC",
        "SOURCE_DATE_EPOCH" to "0",
    )
}

sourceSets.named("main") {
    resources.srcDir(openBlasResources)
}

tasks.named("processResources") {
    if (!lintOnly) dependsOn(buildOpenBlas)
}
tasks.named<Jar>("sourcesJar") { dependsOn(buildOpenBlas) }

val verifyOpenBlasResources = tasks.register("verifyOpenBlasResources") {
    dependsOn(buildOpenBlas)
    val notices = openBlasResources.get().file("THIRD-PARTY-NOTICES.txt").asFile
    inputs.file(notices)
    val requiredOpenBlasResources = supportedPlatforms.associateWith { platform ->
        requiredResources.getValue(platform) + ".openblas-source-sha256" + ".openblas-build-options"
    }
    val resourceDirectories = supportedPlatforms.associateWith { platform ->
        openBlasResources.get().dir("org/bytedeco/openblas/$platform").asFile.absolutePath
    }
    inputs.files(resourceDirectories.values)
    inputs.property("requiredResources", requiredOpenBlasResources)
    inputs.property("resourceDirectories", resourceDirectories)
    doLast {
        check(notices.isFile && notices.length() > 0L) {
            "missing consolidated third-party notices for koblas-openblas"
        }
        @Suppress("UNCHECKED_CAST")
        val required = inputs.properties.getValue("requiredResources") as Map<String, List<String>>
        @Suppress("UNCHECKED_CAST")
        val directories = inputs.properties.getValue("resourceDirectories") as Map<String, String>
        required.forEach { (platform, resources) ->
            val directory = File(directories.getValue(platform))
            check(resources.all { directory.resolve(it).isFile && directory.resolve(it).length() > 0L }) {
                "missing bundled OpenBLAS resources for $platform; build them on that platform before publishing"
            }
        }
    }
}

tasks.configureEach {
    if (name.startsWith("publish")) dependsOn(verifyOpenBlasResources)
}

tasks.withType<Jar>().configureEach {
    manifest {
        attributes("Automatic-Module-Name" to "com.eignex.koblas.openblas")
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}
