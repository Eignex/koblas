plugins {
    id("com.eignex.jvm") version "1.2.7"
}

eignexPublish {
    description.set("OpenBLAS-backed LinearAlgebra for koblas on the JVM, activated by presence on the classpath.")
    githubRepo.set("Eignex/koblas")
}

dependencies {
    api(project(":koblas"))
    // Bundles OpenBLAS natives for every supported OS/arch. Consumers who care about download size can
    // exclude this and depend on org.bytedeco:openblas with just their platform's classifier.
    implementation("org.bytedeco:openblas-platform:0.3.28-1.5.11")
}