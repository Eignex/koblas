import java.io.ByteArrayOutputStream
import java.io.FileOutputStream
import java.net.URI
import java.security.MessageDigest
import javax.inject.Inject
import org.gradle.process.ExecOperations

plugins {
    kotlin("jvm")
    id("com.eignex.publish") version "1.3.3"
}

eignexPublish {
    description.set("Optional vendor BLAS runtimes packaged for koblas, one artifact per platform.")
    githubRepo.set("Eignex/koblas")
}

/**
 * One packaged vendor runtime.
 *
 * `files` maps a path inside the vendor's `.deb` to the name it takes in the payload directory, and
 * `libraries` names the subset that makes up the runtime, entry point first. Everything else is a notice that
 * ships beside the binaries and is never extracted at run time. The manifest the loader reads is `libraries`,
 * written by [PackagePayload] rather than kept as a second list here.
 *
 * `published` is a licence question, not a technical one. Every payload here is built the same way and loads
 * the same way; only oneMKL's terms allow Koblas to hand the binaries on. See the module README.
 */
data class Payload(
    val vendor: String,
    val platform: String,
    val url: String,
    val sha256: String,
    val version: String,
    val files: Map<String, String>,
    val libraries: List<String>,
    val published: Boolean,
    /** A `.deb` inside the downloaded tar, for a vendor that ships one archive per compiler. */
    val innerArchive: String? = null,
    /** A self-extracting installer inside the downloaded tar, whose own payload holds [innerArchive]. */
    val installer: String? = null,
    /** Paths in the downloaded tar itself, for notices a vendor keeps outside its `.deb`. */
    val outerFiles: Map<String, String> = emptyMap(),
)

// oneMKL 2026.1. The four instruction-set layers are all packaged: mkl_rt opens the one matching the host at
// run time, and a payload missing the host's layer does not fail, it quietly runs a narrower one. The bundle
// has to serve any x86-64 host, so leaving one out would buy artifact size with a performance cliff nobody
// could see. Redistribution of the binaries unmodified, with the notice reproduced, is what the Intel
// Simplified Software License in LICENSE.txt permits.
private val oneMklLibrary = "/opt/intel/oneapi/mkl/2026.1/lib"
private val oneMklDoc = "/opt/intel/oneapi/mkl/2026.1/share/doc/mkl/licensing"
private val oneMklFiles = listOf(
    "libmkl_rt.so.3",
    "libmkl_core.so.3",
    "libmkl_sequential.so.3",
    "libmkl_intel_lp64.so.3",
    "libmkl_def.so.3",
    "libmkl_avx2.so.3",
    "libmkl_avx512.so.3",
    "libmkl_avx10.so.3",
)

// AOCL 5.3.0 and ArmPL 25.07 are the single-threaded LP64 builds, each one self-contained library. Both are
// described here and neither is published: their terms allow an operator to build the payload into their own
// application, which these tasks do, and not Koblas to hand the binaries on. See the module README.
private val aoclRoot = "/opt/AMD/aocl/aocl-linux-gcc-5.3.0/gcc/ST"
private val armPlRoot = "/opt/arm/armpl_25.07_gcc"
private val armPlInstaller = "./arm-performance-libraries_25.07_deb/arm-performance-libraries_25.07_deb.sh"
private val armPlLicences = "./arm-performance-libraries_25.07_deb/license_terms"

val payloads = listOf(
    Payload(
        vendor = "onemkl",
        platform = "linux-x86_64",
        url = "https://apt.repos.intel.com/oneapi/pool/main/" +
            "intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb",
        sha256 = "3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93",
        version = "2026.1.0-236",
        files = oneMklFiles.associate { ".$oneMklLibrary/$it" to it } + mapOf(
            ".$oneMklDoc/license.txt" to "LICENSE.txt",
            ".$oneMklDoc/third-party-programs.txt" to "THIRD-PARTY-PROGRAMS.txt",
        ),
        libraries = oneMklFiles,
        published = true,
    ),
    Payload(
        vendor = "aocl",
        platform = "linux-x86_64",
        url = "https://download.amd.com/developer/eula/aocl/aocl-5-3/aocl-linux-gcc-5.3.0_1_amd64.deb",
        sha256 = "2923cc8e69b53996c48ccd87925a45d538c9b50824513ee7bc0110fa7590f1e3",
        version = "5.3.0",
        files = mapOf(
            // The serial LP64 build, taken under the soname it carries.
            ".$aoclRoot/lib_LP64/libblis.so.5.3.0" to "libblis.so.5",
            ".$aoclRoot/amd-blis/EULA.txt" to "EULA.txt",
            ".$aoclRoot/amd-blis/NOTICE.txt" to "NOTICE.txt",
        ),
        libraries = listOf("libblis.so.5"),
        published = false,
    ),
    Payload(
        vendor = "armpl",
        platform = "linux-arm64",
        url = "https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/" +
            "Version_25.07/arm-performance-libraries_25.07_deb_gcc.tar",
        sha256 = "28a0cdf84b1f8e61d1d1ea484f4e2ecf645f7d916bb002ed34d46f6eb2e41345",
        version = "25.07",
        files = mapOf(".$armPlRoot/lib/libarmpl_lp64.so" to "libarmpl_lp64.so"),
        libraries = listOf("libarmpl_lp64.so"),
        published = false,
        installer = armPlInstaller,
        innerArchive = "armpl_25.07_gcc.deb",
        // Arm keeps its licence beside the installer rather than inside the package.
        outerFiles = mapOf(
            "$armPlLicences/license_agreement.txt" to "LICENSE.txt",
            "$armPlLicences/third_party_licenses.txt" to "THIRD-PARTY-LICENSES.txt",
        ),
    ),
)

/**
 * Downloads a payload archive once, checks it against its pinned hash, and lays its files out for packaging.
 *
 * The hash is checked before anything is read out of the archive, so a mirror that served something else
 * fails here rather than producing an artifact that carries it. A `.deb` is an `ar` archive of tarballs, and
 * `ar` and `tar` do the reading: Gradle's own archive handling covers neither `ar` nor the compression these
 * use, and a Linux host building a Linux payload has both.
 */
abstract class PackagePayload @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:Input abstract val url: Property<String>

    @get:Input abstract val sha256: Property<String>

    @get:Input @get:Optional abstract val innerArchive: Property<String>

    @get:Input @get:Optional abstract val installer: Property<String>

    /** Archive path to packaged name for files in the downloaded tar itself, as [Payload.outerFiles]. */
    @get:Input abstract val outerFiles: MapProperty<String, String>

    /** Archive path to packaged name, as [Payload.files]. */
    @get:Input abstract val files: MapProperty<String, String>

    /** The packaged names that make up the runtime, entry point first. */
    @get:Input abstract val libraries: ListProperty<String>

    /** Where the downloaded archives are kept between builds. */
    @get:Internal abstract val downloads: DirectoryProperty

    @get:OutputDirectory abstract val destination: DirectoryProperty

    @TaskAction
    fun package_() {
        val archive = download()
        val work = temporaryDir.resolve("extract").apply { deleteRecursively(); mkdirs() }
        val outer = work.resolve("outer").apply { mkdirs() }
        val container = container(archive, work, outer)
        val data = dataMember(container)
        val payload = work.resolve("payload.tar")
        FileOutputStream(payload).use { out ->
            exec.exec {
                commandLine("ar", "p", container.absolutePath, data)
                standardOutput = out
            }
        }
        val extracted = work.resolve("files").apply { mkdirs() }
        exec.exec {
            commandLine(
                listOf("tar", "-xf", payload.absolutePath, "-C", extracted.absolutePath) + files.get().keys,
            )
        }

        val target = destination.get().asFile.apply { deleteRecursively(); mkdirs() }
        for ((path, name) in files.get()) {
            val source = extracted.resolve(path)
            check(source.isFile) { "$path is missing from ${url.get()}" }
            source.copyTo(target.resolve(name), overwrite = true)
        }
        for ((path, name) in outerFiles.get()) {
            val source = outer.resolve(path)
            check(source.isFile) { "$path is missing from ${url.get()}" }
            source.copyTo(target.resolve(name), overwrite = true)
        }
        // The manifest the loader reads back. Written from what was packaged, so a payload cannot claim a file
        // this task did not put beside it.
        target.resolve("payload").writeText(libraries.get().joinToString("\n", postfix = "\n"))
    }

    /**
     * The `.deb` the files come out of, and the notices laid out in [outer] beside it.
     *
     * A vendor that distributes a bare package is already there. Arm wraps one build per compiler in a tar, and
     * inside that a self-extracting installer whose own archive follows a marker line: the bytes after that
     * line are an ordinary gzipped tar, so the payload is carved out rather than the installer run. Running it
     * would mean accepting a licence on the operator's behalf and writing into system directories, neither of
     * which a build task should do.
     */
    private fun container(archive: File, work: File, outer: File): File {
        val inner = innerArchive.orNull ?: return archive
        val installerPath = installer.orNull
            ?: run {
                exec.exec { commandLine("tar", "-xf", archive.absolutePath, "-C", work.absolutePath, inner) }
                return work.resolve(inner)
            }
        val wanted = listOf(installerPath) + outerFiles.get().keys
        exec.exec { commandLine(listOf("tar", "-xf", archive.absolutePath, "-C", outer.absolutePath) + wanted) }

        val installerFile = outer.resolve(installerPath)
        val carved = work.resolve("installer.tar.gz")
        installerFile.inputStream().buffered().use { input ->
            skipPastMarker(input, installerFile)
            carved.outputStream().buffered().use { output -> input.copyTo(output) }
        }
        exec.exec { commandLine("tar", "-xzf", carved.absolutePath, "-C", work.absolutePath, inner) }
        return work.resolve(inner)
    }

    /**
     * Reads [input] up to and including the line after which the installer's own archive begins.
     *
     * The marker has to start a line, as the installer's own search for it does. The name appears earlier in
     * the script, inside the very command that looks for it, and a scan that took the first occurrence
     * anywhere would carve the archive out of the middle of the shell code.
     */
    private fun skipPastMarker(input: java.io.InputStream, installer: File) {
        val marker = "__START_OF_PAYLOAD__"
        val line = StringBuilder()
        while (true) {
            val byte = input.read()
            check(byte >= 0) { "no payload marker in $installer" }
            if (byte != '\n'.code) {
                line.append(byte.toChar())
                continue
            }
            if (line.startsWith(marker)) return
            line.clear()
        }
    }

    /** The `ar` member holding the files, whose compression suffix differs between vendors. */
    private fun dataMember(archive: File): String {
        val listing = ByteArrayOutputStream()
        exec.exec {
            commandLine("ar", "t", archive.absolutePath)
            standardOutput = listing
        }
        val members = listing.toString(Charsets.UTF_8).lines().map { it.trim() }
        return members.firstOrNull { it.startsWith("data.tar") }
            ?: error("no data member in $archive, found $members")
    }

    /**
     * The archive, downloaded unless a previous build left a copy with the right hash.
     *
     * Kept outside the task's own output directory so that rebuilding the artifact does not fetch hundreds of
     * megabytes again, and re-fetched whenever what is on disk is not what the hash says it should be.
     */
    private fun download(): File {
        val directory = downloads.get().asFile.apply { mkdirs() }
        val file = directory.resolve(url.get().substringAfterLast('/'))
        if (file.isFile && hash(file) == sha256.get()) return file
        logger.lifecycle("Fetching ${url.get()}")
        URI(url.get()).toURL().openStream().use { input ->
            file.outputStream().use { output -> input.copyTo(output) }
        }
        val actual = hash(file)
        check(actual == sha256.get()) {
            "${url.get()} hashed $actual, expected ${sha256.get()}"
        }
        return file
    }

    private fun hash(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1 shl 16)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}

val downloadCache = layout.buildDirectory.dir("payload-downloads")

fun camel(text: String) = text.split('-', '_').joinToString("") { part -> part.replaceFirstChar { it.uppercase() } }

val packageTasks = payloads.associateWith { payload ->
    tasks.register<PackagePayload>("package${camel(payload.vendor)}${camel(payload.platform)}") {
        group = "vendor runtime"
        description = "Downloads ${payload.vendor} ${payload.version} and lays out its ${payload.platform} payload."
        url.set(payload.url)
        sha256.set(payload.sha256)
        payload.innerArchive?.let { innerArchive.set(it) }
        payload.installer?.let { installer.set(it) }
        files.set(payload.files)
        outerFiles.set(payload.outerFiles)
        libraries.set(payload.libraries)
        downloads.set(downloadCache)
        destination.set(layout.buildDirectory.dir("payloads/${payload.platform}/${payload.vendor}"))
    }
}

/** Packages [chosen] under the layout `com.eignex.koblas.vendor.Bundle` reads back. */
fun payloadJar(name: String, classifier: String, platform: String, chosen: List<Payload>) =
    tasks.register<Jar>(name) {
        group = "vendor runtime"
        description = "Packages ${chosen.joinToString(", ") { it.vendor }} for $platform."
        archiveClassifier.set(classifier)
        destinationDirectory.set(layout.buildDirectory.dir("artifacts"))
        for (payload in chosen) {
            from(packageTasks.getValue(payload)) { into("com/eignex/koblas/vendor/$platform/${payload.vendor}") }
        }
    }

// One published artifact per platform, carrying the payloads Koblas may hand on. A classpath resource on the
// JVM, the same directory on disk for a Native binary.
val publishedJars = payloads.filter { it.published }.groupBy { it.platform }.mapValues { (platform, chosen) ->
    payloadJar("jar${camel(platform)}", platform, platform, chosen)
}

// One task per payload Koblas may not hand on, so that an operator can still build it into their own
// application. Nothing here is an artifact of this project: no publication references these.
val localJars = payloads.filterNot { it.published }.associateWith { payload ->
    payloadJar(
        "assemble${camel(payload.vendor)}${camel(payload.platform)}",
        "${payload.platform}-${payload.vendor}",
        payload.platform,
        listOf(payload),
    )
}

publishing.publications.register<MavenPublication>("vendorRuntime") {
    publishedJars.values.forEach { artifact(it) }
    // The convention plugin declares Apache-2.0 on every publication, which describes Koblas and not what this
    // artifact carries: there is no Koblas code in it, only unmodified vendor runtimes under their own terms.
    // Appending would read as a choice between the two, so the element is replaced once the plugin has written
    // it. A declared licence a consumer's audit reads is worth the XML.
    pom.withXml {
        val root = asNode()
        (root.get("licenses") as groovy.util.NodeList).toList().forEach { root.remove(it as groovy.util.Node) }
        val license = root.appendNode("licenses").appendNode("license")
        license.appendNode("name", "Vendor licences, one per payload")
        license.appendNode("url", "https://github.com/Eignex/koblas/blob/main/koblas-vendor-runtime/README.md")
        license.appendNode(
            "comments",
            "Unmodified vendor BLAS runtimes. Each payload directory carries the licence it is distributed " +
                "under, beside its binaries. No Koblas code is in this artifact.",
        )
    }
}

// The payloads are hundreds of megabytes and are fetched from vendor servers, so nothing here hangs off
// `assemble` or `check`. Building an artifact and verifying one are both asked for by name.
tasks.named("assemble") { setDependsOn(emptyList<Any>()) }

kotlin { jvmToolchain(25) }

// Only the verification test compiles here; the artifact itself carries no Koblas code.
repositories { mavenCentral() }

dependencies {
    testImplementation(kotlin("test-junit5"))
    testImplementation(project(":koblas"))
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:6.1.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:6.1.3")
}

// Proves the built artifact, rather than the layout convention: the payload jar goes on the test classpath and
// a vendor is opened out of it. `user.home` points at an empty directory so that every installed candidate
// misses and the bundled payload is the only route left; a host with oneMKL installed would otherwise resolve
// the installed copy, which is the correct precedence and would prove nothing about the artifact.
val hostPlatform = if (System.getProperty("os.arch").lowercase() in setOf("aarch64", "arm64")) {
    "linux-arm64"
} else {
    "linux-x86_64"
}

val hostPayloads = payloads.filter { it.platform == hostPlatform }

// What a Native process reads: the same layout as the jar, unpacked on disk. `:koblas`'s native tests run in
// this directory when `-Pkoblas.nativePayload=true` is passed, which is the Native half of the artifact check
// and cannot be replaced by the JVM's classpath lookup.
tasks.register<Sync>("stageNativePayload") {
    group = "verification"
    description = "Lays the $hostPlatform payloads out on disk for the Native loader to find."
    into(layout.buildDirectory.dir("native-payload"))
    for (payload in hostPayloads) {
        from(packageTasks.getValue(payload)) {
            into("com/eignex/koblas/vendor/${payload.platform}/${payload.vendor}")
        }
    }
}

tasks.register<Test>("verifyPayload") {
    group = "verification"
    description = "Loads every $hostPlatform payload this host can build and computes with each."
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    val jars = listOfNotNull(publishedJars[hostPlatform]) + hostPayloads.mapNotNull { localJars[it] }
    jars.forEach {
        dependsOn(it)
        classpath += files(it)
    }
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("koblas.payload.vendors", hostPayloads.joinToString(",") { it.vendor })
    doFirst {
        // Empty, so that every installed candidate under a real home directory misses.
        systemProperty("user.home", temporaryDir.resolve("no-home").apply { mkdirs() }.absolutePath)
    }
}

// `check` builds no payload: the ordinary test task has nothing to run, and the artifact verification is a
// hundreds-of-megabytes download from vendor servers that is asked for by name.
tasks.named<Test>("test") { enabled = false }
