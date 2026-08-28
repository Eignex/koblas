package com.eignex.koblas.buildlogic

import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.model.ObjectFactory
import org.gradle.api.plugins.JavaPluginExtension
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.MapProperty
import org.gradle.api.provider.Property
import org.gradle.api.provider.Provider
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.InputFiles
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.api.tasks.bundling.Jar
import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.testing.Test
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.listProperty
import org.gradle.kotlin.dsl.mapProperty
import org.gradle.kotlin.dsl.property
import org.gradle.kotlin.dsl.register
import org.gradle.kotlin.dsl.named
import org.gradle.kotlin.dsl.withType
import java.io.File
import javax.inject.Inject

abstract class KoblasNativeLibraryExtension @Inject constructor(objects: ObjectFactory) {
    val libraryName: Property<String> = objects.property()
    val resourcePackage: Property<String> = objects.property()
    val lockFile: RegularFileProperty = objects.fileProperty()
    val buildScript: RegularFileProperty = objects.fileProperty()
    val platformProperty: Property<String> = objects.property<String>().convention("koblas.platform")
    val supportedPlatforms: ListProperty<String> = objects.listProperty()
    val requiredResources: MapProperty<String, List<String>> = objects.mapProperty()
    val forbiddenResources: MapProperty<String, List<String>> = objects.mapProperty()
    val forbiddenNoticeText: ListProperty<String> = objects.listProperty()
    val compilerDefaults: MapProperty<String, String> = objects.mapProperty()
    val toolCommands: MapProperty<String, String> = objects.mapProperty()
    val extraArguments: ListProperty<String> = objects.listProperty()
    val platformArguments: MapProperty<String, List<String>> = objects.mapProperty()
    val dependsOnTasks: ListProperty<String> = objects.listProperty()
    val testJvmArgs: ListProperty<String> = objects.listProperty()
    val extraInputFiles: ConfigurableFileCollection = objects.fileCollection()

    fun compiler(environment: String, defaultCommand: String) {
        compilerDefaults.put(environment, defaultCommand)
        toolCommands.put(environment.lowercase(), defaultCommand)
    }

    fun tool(name: String, command: String) {
        toolCommands.put(name, command)
    }
}

abstract class VerifyKoblasNativeResourcesTask : DefaultTask() {
    @get:InputFile
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val notices: RegularFileProperty

    @get:Input
    abstract val requiredResources: MapProperty<String, List<String>>

    @get:Input
    abstract val resourceDirectories: MapProperty<String, String>

    @get:Input
    abstract val forbiddenResources: MapProperty<String, List<String>>

    @get:Input
    abstract val forbiddenNoticeText: ListProperty<String>

    @get:InputFiles
    @get:Optional
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val resourceFiles: ConfigurableFileCollection

    @TaskAction
    fun verify() {
        check(notices.asFile.get().isFile && notices.asFile.get().length() > 0L) {
            "missing consolidated third-party notices for ${project.name}"
        }
        requiredResources.get().forEach { (platform, resources) ->
            val directory = File(resourceDirectories.get().getValue(platform))
            val missing = resources.filterNot { resource ->
                directory.resolve(resource).isFile && directory.resolve(resource).length() > 0L
            }
            check(missing.isEmpty()) {
                "missing bundled ${project.name.removePrefix("koblas-")} resources for $platform: " +
                    "${missing.joinToString()}; build them on that platform before publishing"
            }
            val forbidden = forbiddenResources.get().getOrDefault(platform, emptyList())
            val present = directory.listFiles().orEmpty().filter { file ->
                forbidden.any { pattern -> file.name.contains(pattern, ignoreCase = true) }
            }
            check(present.isEmpty()) {
                "forbidden resources remain in the $platform bundle: ${present.joinToString { it.name }}"
            }
        }
        val noticesText = notices.asFile.get().readText()
        forbiddenNoticeText.get().forEach { text ->
            check(text !in noticesText) { "$text notices remain although the resource is not shipped" }
        }
    }
}

abstract class CheckKoblasBuildScriptStructureTask : DefaultTask() {
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val scripts: ConfigurableFileCollection

    @TaskAction
    fun checkScripts() {
        val forbidden = listOf("System.getProperty", "gradle.startParameter", "tasks.register", "doLast")
        val offenders = scripts.files.flatMap { script ->
            val lines = script.readLines()
            if ("id(\"koblas.native-library\")" !in lines.joinToString("\n")) {
                listOf("${script.name}: missing koblas.native-library convention plugin")
            } else {
                lines.withIndex()
                    .filter { (_, line) -> forbidden.any(line::contains) }
                    .map { (index, line) -> "${script.name}:${index + 1}: ${line.trim()}" }
            }
        }
        if (offenders.isNotEmpty()) {
            throw GradleException(
                "bundled-native build scripts must declare data through koblas.native-library:\n" +
                    offenders.joinToString("\n").prependIndent("  "),
            )
        }
    }
}

class KoblasNativeLibraryPlugin : Plugin<Project> {
    override fun apply(project: Project) {
        require(project.pluginManager.hasPlugin("com.eignex.jvm")) {
            "koblas.native-library must be applied after com.eignex.jvm"
        }
        val extension = project.extensions.create("koblasNativeLibrary", KoblasNativeLibraryExtension::class.java)
        project.afterEvaluate { configure(project, extension) }
    }

    private fun configure(project: Project, extension: KoblasNativeLibraryExtension) {
        val libraryName = extension.libraryName.get()
        val taskSuffix = libraryName.replaceFirstChar(Char::uppercaseChar)
        val resourceRoot = project.layout.buildDirectory.dir("${libraryName.lowercase()}/resources")
        val hostPlatform = hostPlatform(project.providers)
        val platform = project.providers.gradleProperty(extension.platformProperty.get()).orElse(hostPlatform)
        val lintOnly = project.gradle.startParameter.taskNames.let { names ->
            names.isNotEmpty() && names.all { it.substringAfterLast(':') == "lintDocs" }
        }
        val buildTask = project.tasks.register<Exec>("build$taskSuffix") {
            dependsOn(extension.dependsOnTasks.get())
            val output = resourceRoot.get().asFile.absolutePath
            val platformValue = platform.get()
            inputs.file(extension.lockFile)
            inputs.file(extension.buildScript)
            inputs.file(project.rootProject.layout.projectDirectory.file("scripts/third-party-notices.sh"))
            inputs.files(extension.extraInputFiles)
            inputs.property("platform", platform)
            inputs.property("resourceLayout", "complete-resources-v1")
            inputs.property("os.name", project.providers.systemProperty("os.name"))
            inputs.property("os.arch", project.providers.systemProperty("os.arch"))
            extension.compilerDefaults.get().forEach { (environment, defaultCommand) ->
                val command = project.providers.environmentVariable(environment).orElse(defaultCommand)
                inputs.property(environment, command)
            }
            extension.toolCommands.get().forEach { (name, command) ->
                inputs.property("tool-$name", project.providers.exec { commandLine(command, "--version") }.standardOutput.asText)
            }
            outputs.dir(resourceRoot)
            outputs.cacheIf("the locked source, target platform, and toolchain are declared inputs") { true }
            val extraArguments = if (extension.platformArguments.getOrElse(emptyMap()).isNotEmpty()) {
                extension.platformArguments.get().getValue(platform.get())
            } else {
                extension.extraArguments.get()
            }
            commandLine("bash", extension.buildScript.get().asFile.absolutePath, "--platform", platformValue, "--output", output, *extraArguments.toTypedArray())
            extension.compilerDefaults.get().forEach { (environment, defaultCommand) ->
                environment(environment, project.providers.environmentVariable(environment).orElse(defaultCommand).get())
            }
            environment(
                mapOf(
                    "CFLAGS" to "", "CXXFLAGS" to "", "FFLAGS" to "", "LDFLAGS" to "",
                    "MAKEFLAGS" to "", "LC_ALL" to "C", "TZ" to "UTC", "SOURCE_DATE_EPOCH" to "0",
                ),
            )
        }

        project.extensions.configure<JavaPluginExtension> {
            sourceSets.named("main") { resources.srcDir(resourceRoot) }
        }
        project.tasks.named("processResources") {
            if (!lintOnly) dependsOn(buildTask)
        }
        project.tasks.named<Jar>("sourcesJar") { dependsOn(buildTask) }

        val resourceDirectories = extension.supportedPlatforms.get().associateWith { supportedPlatform ->
            resourceRoot.get().dir("${extension.resourcePackage.get()}/$supportedPlatform").asFile.absolutePath
        }
        val verifyTask = project.tasks.register<VerifyKoblasNativeResourcesTask>("verify${taskSuffix}Resources") {
            dependsOn(buildTask)
            notices.set(resourceRoot.map { it.file("THIRD-PARTY-NOTICES.txt") })
            requiredResources.set(extension.requiredResources)
            this.resourceDirectories.set(resourceDirectories)
            forbiddenResources.set(extension.forbiddenResources)
            forbiddenNoticeText.set(extension.forbiddenNoticeText)
            resourceFiles.from(resourceDirectories.values)
        }
        project.tasks.configureEach {
            if (name.startsWith("publish")) dependsOn(verifyTask)
        }
        project.tasks.withType<Jar>().configureEach {
            manifest { attributes(mapOf("Automatic-Module-Name" to "com.eignex.koblas.${libraryName.lowercase()}")) }
        }
        project.tasks.withType<Test>().configureEach {
            jvmArgs("--enable-native-access=ALL-UNNAMED", *extension.testJvmArgs.get().toTypedArray())
        }
    }

    private fun hostPlatform(providers: org.gradle.api.provider.ProviderFactory): Provider<String> =
        providers.systemProperty("os.name").zip(providers.systemProperty("os.arch")) { osName, osArch ->
            when {
                osName.startsWith("Linux", ignoreCase = true) && osArch in setOf("amd64", "x86_64") -> "linux-x86_64"
                osName.startsWith("Linux", ignoreCase = true) && osArch in setOf("aarch64", "arm64") -> "linux-arm64"
                osName.startsWith("Mac", ignoreCase = true) && osArch in setOf("aarch64", "arm64") -> "macosx-arm64"
                else -> error("unsupported koblas native-library host $osName/$osArch")
            }
        }
}

class KoblasBuildScriptStructurePlugin : Plugin<Project> {
    override fun apply(project: Project) {
        val scripts = listOf(
            "koblas-basiclu/build.gradle.kts",
            "koblas-hfactor/build.gradle.kts",
            "koblas-openblas/build.gradle.kts",
            "koblas-suitesparse/build.gradle.kts",
        ).map { project.layout.projectDirectory.file(it).asFile }
        project.tasks.register<CheckKoblasBuildScriptStructureTask>("checkBuildScriptStructure") {
            group = "verification"
            description = "Checks bundled-native module scripts remain declarative declarations."
            this.scripts.from(scripts)
        }
    }
}
