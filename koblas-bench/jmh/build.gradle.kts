import java.security.MessageDigest

plugins { `java-library` }

val sourceRevision = "a194eead0136bb66e5e59e4fdb2e18543e730929"
val sourceChecksum = "619143e9da4765499f21cad94f732740dd8ff5f6637554a2b1ff37cae374bd80"
version = "1.38-SNAPSHOT+${sourceRevision.take(12)}"

// The core and annotation processor come from one upstream revision so generated code and runtime agree.
val upstreamSource = configurations.create("upstreamSource") {
    isCanBeConsumed = false
    isTransitive = false
}
repositories {
    mavenCentral()
    exclusiveContent {
        forRepository {
            ivy {
                name = "openjdkJmhSource"
                url = uri("https://codeload.github.com/openjdk/jmh/zip")
                patternLayout { artifact("[revision]") }
                metadataSources { artifact() }
            }
        }
        filter { includeModule("openjdk.source", "jmh") }
    }
}
dependencies {
    upstreamSource("openjdk.source:jmh:$sourceRevision@zip")
    api("net.sf.jopt-simple:jopt-simple:5.0.4")
    api("org.apache.commons:commons-math3:3.6.1")
}

/** Verifies the immutable source archive before any upstream files enter the compilation. */
@CacheableTask
abstract class VerifyJmhSource : DefaultTask() {
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val archive: RegularFileProperty

    @get:Input
    abstract val expectedSha256: Property<String>

    @get:OutputFile
    abstract val marker: RegularFileProperty

    @TaskAction
    fun verify() {
        val digest = MessageDigest.getInstance("SHA-256")
        archive.get().asFile.inputStream().use { input ->
            val buffer = ByteArray(8192)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        check(actual == expectedSha256.get()) { "JMH source archive checksum mismatch: $actual" }
        marker.get().asFile.apply { parentFile.mkdirs(); writeText(actual) }
    }
}

val verifyUpstreamSource = tasks.register<VerifyJmhSource>("verifyUpstreamSource") {
    archive.fileProvider(upstreamSource.elements.map { it.single().asFile })
    expectedSha256.set(sourceChecksum)
    marker.set(layout.buildDirectory.file("upstream/verified.sha256"))
}

val unpackUpstreamSource = tasks.register<Sync>("unpackUpstreamSource") {
    dependsOn(verifyUpstreamSource)
    from(zipTree(verifyUpstreamSource.flatMap { it.archive })) {
        include("jmh-$sourceRevision/jmh-core/src/main/**")
        include("jmh-$sourceRevision/jmh-generator-annprocess/src/main/**")
        eachFile { relativePath = RelativePath(true, *relativePath.segments.drop(4).toTypedArray()) }
        includeEmptyDirs = false
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    into(layout.buildDirectory.dir("upstream/main"))
}

java {
    toolchain.languageVersion.set(JavaLanguageVersion.of(25))
}
tasks.withType<JavaCompile>().configureEach {
    // Keep the upstream harness compatible with Java 11; benchmark code and execution use JDK 25.
    options.release.set(11)
}
sourceSets.main {
    java.setSrcDirs(listOf(unpackUpstreamSource.map { it.destinationDir.resolve("java") }))
    resources.setSrcDirs(listOf(unpackUpstreamSource.map { it.destinationDir.resolve("resources") }))
}
tasks.processResources {
    val jmhVersion = project.version.toString()
    inputs.property("jmhVersion", jmhVersion)
    filesMatching("jmh.properties") {
        expand("project" to mapOf("version" to jmhVersion), "buildDate" to "2026/07/14")
    }
}
