plugins {
    id("com.eignex.jvm") version "1.3.1"
}

eignexPublish {
    description.set("Maven-hosted OpenBLAS and LAPACKE loader for koblas on the JVM.")
    githubRepo.set("Eignex/koblas")
}

dependencies {
    api(project(":koblas"))
    // Provides the maintained platform binary jars. koblas owns the extraction and FFM loading path; no
    // JavaCPP classes or JNI entry points are used at runtime.
    runtimeOnly("org.bytedeco:openblas-platform:0.3.34-1.5.14")
}

tasks.withType<Jar>().configureEach {
    manifest {
        attributes("Automatic-Module-Name" to "com.eignex.koblas.openblas")
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}
