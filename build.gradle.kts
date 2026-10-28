plugins {
    // Declared here with apply=false so that two KMP subprojects don't each load
    // the Kotlin plugin into separate classloaders (which conflicts on shared
    // build services like KotlinNativeBundleBuildService).
    id("com.eignex.kmp") apply false
    kotlin("plugin.serialization") apply false
}

allprojects {
    repositories {
        ivy {
            url = uri("https://github.com/Eignex/kbuild/releases/download")
            patternLayout { artifact("v[revision]/[artifact]-[revision](-[classifier]).[ext]") }
            metadataSources { gradleMetadata() }
            content { includeModule("com.eignex", "kbuild-platform") }
        }
    }
}
