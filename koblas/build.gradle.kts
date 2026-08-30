import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
    id("com.eignex.kmp") version "1.3.2"
    kotlin("plugin.serialization") version "2.4.10"
}

eignexPublish {
    description.set("Dense/sparse linear algebra for Kotlin/KMP with a pluggable BLAS/LAPACK backend seam.")
    githubRepo.set("Eignex/koblas")
}

kotlin {
    explicitApi()
    applyDefaultHierarchyTemplate()
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
        optIn.add("com.eignex.koblas.UnsafeKoblasApi")
    }
    // The JVM host-BLAS backend binds through java.lang.foreign, finalized in 22 and used here with
    // Linker.Option.critical. 25 is the current LTS-track release; this is the floor for JVM consumers.
    jvmToolchain(25)
    jvm {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_25)
        }
    }
    linuxX64(); linuxArm64()
    macosArm64()

    targets.withType<KotlinNativeTarget>().configureEach {
        compilations.getByName("main").cinterops.create("koblasKernels") {
            definitionFile.set(project.file("src/nativeInterop/cinterop/koblas_kernels.def"))
            includeDirs(project.file("native"))
        }
    }

    sourceSets {
        // Kotlin/Native compiles the shared C level-1 leaves beneath its host backends.
        val scalarMain = create("scalarMain") {
            dependsOn(commonMain.get())
        }
        nativeMain.get().dependsOn(scalarMain)

        // Every supported Native target can resolve OpenBLAS and SuiteSparse independently, so either
        // library may still be absent at runtime and fall back to scalar code.
        val hostMain = create("hostMain") { dependsOn(nativeMain.get()) }
        linuxMain.get().dependsOn(hostMain)
        macosMain.get().dependsOn(hostMain)

        val hostTest = create("hostTest") { dependsOn(nativeTest.get()) }
        linuxTest.get().dependsOn(hostTest)
        macosTest.get().dependsOn(hostTest)

        commonMain.dependencies {
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
        }
        commonTest.dependencies {
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
        }
    }
}

val jvmKernelsPlatform = providers.gradleProperty("koblas.kernels.platform").orElse(
    providers.systemProperty("os.name").zip(providers.systemProperty("os.arch")) { osName, architecture ->
        when {
            osName.startsWith("Linux", ignoreCase = true) && architecture in setOf("amd64", "x86_64") ->
                "linux-x86_64"
            osName.startsWith("Linux", ignoreCase = true) && architecture in setOf("aarch64", "arm64") ->
                "linux-arm64"
            osName.startsWith("Mac", ignoreCase = true) && architecture in setOf("aarch64", "arm64") ->
                "macosx-arm64"
            else -> error("unsupported koblas kernel host $osName/$architecture")
        }
    },
)
val jvmKernelsResources = layout.buildDirectory.dir("kernels/resources")
val buildJvmKernels = tasks.register<Exec>("buildJvmKernels") {
    inputs.files("native/koblas_kernels.c", "native/koblas_kernels.h", "../scripts/build-koblas-kernels.sh")
    inputs.property("platform", jvmKernelsPlatform)
    inputs.property("CC", providers.environmentVariable("CC").orElse("cc"))
    outputs.dir(jvmKernelsResources)
    commandLine(
        "bash",
        rootProject.file("scripts/build-koblas-kernels.sh").absolutePath,
        "--platform",
        jvmKernelsPlatform.get(),
        "--output",
        jvmKernelsResources.get().asFile.absolutePath,
    )
    environment("CC", providers.environmentVariable("CC").orElse("cc").get())
}
kotlin.sourceSets.named("jvmMain") { resources.srcDir(jvmKernelsResources) }
tasks.named("jvmProcessResources") { dependsOn(buildJvmKernels) }

// Dokka site is the canonical user documentation. Module-level and per-package prose live in
// adjacent .md files referenced here.
dokka {
    moduleName.set("koblas")
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
        includes.from(
            "module.md",
            "src/commonMain/kotlin/com/eignex/koblas/package.md",
            "src/commonMain/kotlin/com/eignex/koblas/dense/package.md",
            "src/commonMain/kotlin/com/eignex/koblas/sparse/package.md",
            "src/commonMain/kotlin/com/eignex/koblas/sparse/basis/package.md",
        )
    }
}

// JVM SIMD primitives in Primitives.kt use the incubator Vector API. Make the module visible to the
// Kotlin compiler and at test runtime; downstream JVM consumers need the same flag.
tasks.withType<KotlinJvmCompile>().configureEach {
    compilerOptions.freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector")
}
// FFM downcalls are restricted methods: a warning on 25, an error later. Consumers pass the same flag
// when they want the host-BLAS backend; core koblas needs it only because the backend now ships inside.
tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (project.findProperty("koblas.noSimd") != "true") {
        jvmArgs("--add-modules=jdk.incubator.vector")
    }
    // Tests marked @Category(HostLibraryTest) need a real OpenBLAS or SuiteSparse, so they are out of the
    // default run and opted into with -Pkoblas.hostTests=true. They measure the machine as much as the library,
    // so including them would make the everyday result mean different things on a box with SuiteSparse and one
    // without. Excluding them and pinning every backend role to `reference` keeps that out; the opt-in
    // run accepts the noise in exchange for exercising the bindings, and reports coverage like any other run.
    if (project.findProperty("koblas.hostTests") == "true") return@configureEach
    systemProperty("koblas.backend.dense.kernels", "reference")
    systemProperty("koblas.backend.dense.blas", "reference")
    systemProperty("koblas.backend.dense.decompositions", "reference")
    systemProperty("koblas.backend.sparse.kernels", "reference")
    systemProperty("koblas.backend.sparse.blas", "reference")
    systemProperty("koblas.backend.sparse.general.lu", "reference")
    systemProperty("koblas.backend.sparse.repeated.lu", "reference")
    systemProperty("koblas.backend.sparse.cholesky", "reference")
    systemProperty("koblas.backend.sparse.ldl", "reference")
    systemProperty("koblas.backend.sparse.qr", "reference")
    systemProperty("koblas.backend.basis.factorizations", "reference")
    systemProperty("koblas.backend.basis.solvers", "reference")
    useJUnit {
        excludeCategories("com.eignex.koblas.testutil.host.HostLibraryTest")
    }
}

// Kotlin emits a `$DefaultImpls` holder for every interface with a body, and a bridge for every method with
// a default argument. Neither is reachable from Kotlin call sites, so both count as permanently uncovered and
// make the report read as though tested code were not: `Blas.syr` shows its real body at 98% next to a bridge
// at 0%. Dropping the holders leaves the report describing code a test can actually reach.
kover {
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
