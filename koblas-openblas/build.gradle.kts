plugins {
    id("com.eignex.jvm") version "1.2.7"
}

eignexPublish {
    description.set("OpenBLAS-backed LinearAlgebra for koblas on the JVM, activated by presence on the classpath.")
    githubRepo.set("Eignex/koblas")
}

// java.lang.foreign downcalls need the finalized API; 25 is the current LTS. Only this module needs
// it - koblas core stays on the lower baseline, so consumers that skip this artifact are unaffected.
kotlin {
    jvmToolchain(25)
}

dependencies {
    api(project(":koblas"))
    // Bundles OpenBLAS natives for every supported OS/arch. Consumers who care about download size can
    // exclude this and depend on org.bytedeco:openblas with just their platform's classifier.
    implementation("org.bytedeco:openblas-platform:0.3.28-1.5.11")
}

// FFM downcalls and the natives' loader are restricted methods: a warning on 25, an error later.
// Consumers of this artifact pass the same flag; see the README.
tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
