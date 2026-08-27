import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

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

    sourceSets {
        // Kotlin/Native keeps scalar primitive leaves as the fallback beneath its host backends.
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
    // without. Excluding them and pinning dense and sparse backends to `reference` keeps that out; the opt-in
    // run accepts the noise in exchange for exercising the bindings, and reports coverage like any other run.
    if (project.findProperty("koblas.hostTests") == "true") return@configureEach
    systemProperty("koblas.dense.backend", "reference")
    systemProperty("koblas.sparse.backend", "reference")
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
                classes("*\$DefaultImpls")
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
