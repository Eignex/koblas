@file:OptIn(org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi::class)

import org.gradle.api.tasks.JavaExec
import org.gradle.jvm.toolchain.JavaLanguageVersion
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget
import org.gradle.process.ExecOperations
import org.gradle.jvm.toolchain.JavaLauncher
import javax.inject.Inject

plugins {
    id("com.eignex.kmp")
    id("com.eignex.publish.github")
    kotlin("plugin.serialization")
}

val fullTargets = providers.gradleProperty("targets.full").isPresent

eignexBuild {
    abiValidationEnabled.set(fullTargets)
    jvmToolchain.set(25)
    jvmTarget.set(17)
    multiReleaseJvmVersion.set(25)
    jvmModuleDescriptor.set(layout.projectDirectory.file("src/jvmModule/module-info.java"))
    jvmModuleVersioned.set(true)
    packagedTestJdks.set(listOf(17))
}

eignexPublish {
    description.set("Portable dense and sparse BLAS for Kotlin Multiplatform with JVM SIMD and optional host bindings.")
    githubRepo.set("Eignex/koblas")
    // Kotlin/Native has no Linux ARM64 host, so the multiplatform publication is never created there. Releases
    // are built elsewhere; skipping publication lets ARM benchmark workers configure the JVM build.
    publish.set(
        !(System.getProperty("os.name").startsWith("Linux") && System.getProperty("os.arch") in setOf("aarch64", "arm64")),
    )
}

kotlin {
    explicitApi()
    compilerOptions {
        allWarningsAsErrors.set(true)
        freeCompilerArgs.add("-Xexpect-actual-classes")
        optIn.add("com.eignex.koblas.UnsafeKoblasApi")
    }
    jvm()
    linuxX64()
    if (fullTargets) {
        iosArm64()
        iosSimulatorArm64()
    }
    if (fullTargets || gradle.startParameter.taskNames.any { it.endsWith("LinuxArm64") }) linuxArm64()
    if (fullTargets || System.getProperty("os.name").startsWith("Windows")) mingwX64()
    if (fullTargets || System.getProperty("os.name").startsWith("Mac")) macosArm64()

    applyDefaultHierarchyTemplate {
        common {
            group("native") {
                group("posix") {
                    withLinux()
                    withApple()
                }
            }
        }
    }

    sourceSets {
        commonMain.dependencies {
            api("org.jetbrains.kotlinx:kotlinx-serialization-core")
        }
        commonTest.dependencies {
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json")
        }
    }
}

// Module and package overviews share one Dokka include.
dokka {
    dokkaSourceSets.configureEach {
        sourceLink {
            localDirectory.set(projectDir.resolve("src"))
            val sub = projectDir.relativeTo(rootDir).invariantSeparatorsPath
            val prefix = if (sub.isEmpty()) "src" else "$sub/src"
            remoteUrl("https://github.com/Eignex/${rootProject.name}/blob/main/$prefix")
            remoteLineSuffix.set("#L")
        }
    }
    dokkaSourceSets.named("commonMain") {
        includes.from("module.md")
    }
}

val desktopJavaLauncher = javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) }

abstract class InternalAssertionPolicy @Inject constructor(private val execution: ExecOperations) : Action<Task> {
    lateinit var launcher: Provider<JavaLauncher>
    lateinit var toolClasspath: FileCollection
    override fun execute(task: Task) {
        execution.javaexec {
            executable = launcher.get().executablePath.asFile.absolutePath
            classpath = toolClasspath
            mainClass.set("com.eignex.koblas.bytecode.InternalAssertions")
            jvmArgs("--add-modules=jdk.incubator.vector")
            args((task as Jar).archiveFile.get().asFile.absolutePath)
        }
    }
}

val internalAssertionTool = configurations.create("internalAssertionTool")
dependencies { add(internalAssertionTool.name, project(":jvm-bytecode")) }
val verifyKernelBoundaries = tasks.register<JavaExec>("verifyKernelBoundaries") {
    val artifact = tasks.named<Jar>("jvmJar")
    dependsOn(artifact)
    javaLauncher.set(desktopJavaLauncher)
    classpath = internalAssertionTool
    mainClass.set("com.eignex.koblas.bytecode.KernelBoundaries")
    inputs.file(artifact.flatMap { it.archiveFile })
    args(artifact.get().archiveFile.get().asFile.absolutePath)
}
tasks.named("check") { dependsOn(verifyKernelBoundaries) }

if (providers.gradleProperty("koblas.stripInternalAssertions").orNull == "true") {
    val action = objects.newInstance<InternalAssertionPolicy>().apply {
        launcher = desktopJavaLauncher
        toolClasspath = internalAssertionTool + configurations.getByName("jvmCompileClasspath")
    }
    tasks.named<Jar>("jvmJar") {
        dependsOn(action.toolClasspath)
        inputs.files(action.toolClasspath).withPropertyName("internalAssertionTool").withNormalizer(ClasspathNormalizer::class)
        inputs.property("stripInternalAssertions", true)
        inputs.property("internalAssertionJava", desktopJavaLauncher.map { "${it.metadata.vendor} ${it.metadata.javaRuntimeVersion}" })
        doLast(action)
    }
}

val desktopJvmArguments = listOf("--enable-native-access=ALL-UNNAMED") +
    if (providers.gradleProperty("koblas.noSimd").orNull == "true") emptyList()
    else listOf("--add-modules=jdk.incubator.vector")

val packagedDesktopClasspath = with(kotlin.targets.getByName("jvm") as KotlinJvmTarget) {
    val main = compilations.getByName("main")
    val baselineTest = compilations.getByName("test")
    val productionOutputs = files(main.output.allOutputs)
    val desktopTest = compilations.create("desktopTest") {
        associateWith(main)
        associateWith(baselineTest)
        compileTaskProvider.configure {
            compilerOptions.jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_25)
            compilerOptions.freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector")
        }
        compileJavaTaskProvider?.configure { options.release.set(25) }
    }
    // Kbuild creates the MR compilation during its lifecycle hook.
    compilations.matching { it.name == "java25" }.all {
        productionOutputs.from(output.allOutputs)
        compileTaskProvider.configure {
            // Opaque null assertions can force Vector API values onto the heap when C2 exhausts its inlining budget.
            compilerOptions.freeCompilerArgs.addAll(
                "-Xadd-modules=jdk.incubator.vector",
                "-Xno-param-assertions",
                "-Xno-call-assertions",
                "-Xno-receiver-assertions",
            )
        }
        desktopTest.associateWith(this)
        val versionedOutput = output.classesDirs
        tasks.named<KotlinJvmCompile>(desktopTest.compileKotlinTaskName) {
            friendPaths.setFrom(main.output.classesDirs, baselineTest.output.classesDirs, versionedOutput)
        }
    }
    val jvmJar = tasks.named<Jar>("jvmJar")
    val packagedBaselineClasspath = files(jvmJar, baselineTest.output.allOutputs, baselineTest.runtimeDependencyFiles) - productionOutputs
    val desktopClasspath = files(
        packagedBaselineClasspath, desktopTest.output.allOutputs, desktopTest.runtimeDependencyFiles,
    ) - productionOutputs

    tasks.named<Test>("jvmTest") {
        testClassesDirs = files(baselineTest.output.classesDirs, desktopTest.output.classesDirs)
        classpath = desktopClasspath
        javaLauncher.set(desktopJavaLauncher)
        jvmArgs(desktopJvmArguments)
    }
    tasks.withType<Test>().matching { it.name == "jvmTestJarOnJava17" }.configureEach {
        testClassesDirs = baselineTest.output.classesDirs
        classpath = packagedBaselineClasspath
    }
    desktopClasspath
}

/** Each profile is a separate uninstrumented process with its own fixed runtime policy. */
data class JvmCheckProfile(
    val name: String,
    val main: String,
    val arguments: List<String> = emptyList(),
    val vmArguments: List<String> = emptyList(),
    val heap: String? = null,
    val environment: Map<String, String> = emptyMap(),
    val oneMkl: Boolean = false,
    val runtimeHome: String? = null,
)

val vectorArguments = listOf("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
val allocationArguments = vectorArguments + listOf("-XX:-TieredCompilation", "-XX:CompileThreshold=1000")
val profiles = buildList {
    for (family in listOf("Dense", "Sparse")) {
        for ((suffix, flags) in listOf(
            "" to emptyList(),
            "NoFma" to listOf("-XX:-UseFMA"),
            "NarrowNoFma" to listOf("-XX:MaxVectorSize=16", "-XX:-UseFMA"),
        )) {
            add(JvmCheckProfile(
                "simd${family}AllocationCheck$suffix",
                "com.eignex.koblas.${family.lowercase()}.Simd${family}AllocationCheck",
                vmArguments = allocationArguments + flags,
            ))
        }
    }
    for ((name, args, flags) in listOf(
        Triple("simdNarrowSpeciesCheck", listOf("2"), listOf("-XX:MaxVectorSize=16")),
        Triple("simdNoFmaCheck", listOf("", "false"), listOf("-XX:-UseFMA")),
    )) {
        add(JvmCheckProfile(name, "com.eignex.koblas.dense.SimdRuntimePathCheck", args, vectorArguments + flags))
    }
    for ((name, mode, flags) in listOf(
        Triple("simdG1AlignmentCheck", "g1", listOf("-XX:+UseG1GC", "-XX:G1HeapRegionSize=1m")),
        Triple("simdSerialAlignmentCheck", "fallback", listOf("-XX:+UseSerialGC")),
        Triple("simdZgcAlignmentCheck", "zgc", listOf("-XX:+UseZGC")),
        Triple("simdObjectAlignmentCheck", "object64", listOf("-XX:+UseSerialGC", "-XX:ObjectAlignmentInBytes=64")),
    )) {
        add(JvmCheckProfile(
            name, "com.eignex.koblas.dense.SimdReductionAlignmentCheck", listOf(mode),
            allocationArguments + flags, heap = "128m",
            runtimeHome = providers.gradleProperty("koblas.runtimeCheckJavaHome").orNull,
        ))
    }
    for (mode in listOf("inherited", "initialized")) {
        add(JvmCheckProfile(
            "oneMkl${mode.replaceFirstChar(Char::uppercase)}InterfaceCheck",
            "com.eignex.koblas.internal.host.OneMklInterfaceCheck", listOf(mode),
            listOf("--enable-native-access=ALL-UNNAMED", "-XX:ActiveProcessorCount=1"),
            "128m", mapOf("MKL_INTERFACE_LAYER" to "ILP64"), oneMkl = true,
        ))
    }
}
val isolatedChecks = profiles.map { profile ->
    tasks.register<JavaExec>(profile.name) {
        group = "verification"
        description = "Runs the isolated ${profile.name} runtime profile."
        classpath = packagedDesktopClasspath
        mainClass.set(profile.main)
        if (profile.runtimeHome == null) javaLauncher.set(desktopJavaLauncher)
        else setExecutable("${profile.runtimeHome}/bin/java")
        args(profile.arguments)
        jvmArgs(profile.vmArguments)
        profile.heap?.let { maxHeapSize = it }
        environment(profile.environment)
        if (profile.oneMkl) {
            systemProperty("koblas.requireOneMkl", providers.gradleProperty("koblas.requireOneMkl").orElse("false").get())
        }
    }
}
if (fullTargets) tasks.named("check") { dependsOn(isolatedChecks) }

// Kotlin emits a `$DefaultImpls` holder for every interface with a body, and a bridge for every method with
// a default argument. Neither is reachable from Kotlin call sites, so both count as permanently uncovered and
// make the report read as though tested code were not: `Blas.syr` shows its real body at 98% next to a bridge
// at 0%. Dropping the holders leaves the report describing code a test can actually reach.
kover {
    currentProject {
        sources {
            // Kover otherwise treats custom compilations as production report inputs.
            excludedSourceSets.addAll("desktopTest", "jvmDesktopTest")
        }
    }
    reports {
        filters {
            excludes {
                classes($$"*$DefaultImpls")
            }
        }
    }
}

// A stable module name, so a modular consumer sees a named module rather than one named after the jar
// file. That is what lets native access be granted per module instead of blanket ALL-UNNAMED.
tasks.named<Jar>("jvmJar") {
    manifest {
        attributes("Automatic-Module-Name" to "com.eignex.koblas")
    }
}

// Associated baseline test sources are compiled separately; only the custom source directory is linted here.
tasks.withType<dev.detekt.gradle.Detekt>().matching {
    it.name == "detektDesktopTestJvm"
}.configureEach {
    autoCorrect.set(false)
    mustRunAfter("detektCommonTestSourceSet", "detektJvmTestSourceSet", "detektJvmDesktopTestSourceSet")
}

// Formatting a physical source set must finish before type-resolved analysis reads its associated sources.
tasks.withType<dev.detekt.gradle.Detekt>().matching {
    it.name == "detektJava25Jvm"
}.configureEach {
    autoCorrect.set(false)
    mustRunAfter("detektCommonMainSourceSet", "detektJvmMainSourceSet", "detektJvmJava25SourceSet")
}

// CI selects an installed simulator so the test does not depend on a hard-coded iPhone model.
tasks.withType<org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeSimulatorTest>().configureEach {
    providers.gradleProperty("iosSimulatorDevice").orNull?.let { device.set(it) }
}
